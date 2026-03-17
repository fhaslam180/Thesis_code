"""Deterministic hazard-scoring tools and LLM-based supplier discovery.

Hazard scores come from computation, never from an LLM.
Supplier discovery uses GPT-4o with structured output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import ssl
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from bor_risk.models import (
    AlternativeResponse,
    CompanyProfile,
    HazardScore,
    MitigationResponse,
    Supplier,
    TierResponse,
)
from bor_risk.utils import grid_key, load_prompts, load_reference_set, percentile_rank

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency / missing tavily in test env.
_search_module = None

_DEFAULT_SUPPLIERS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tests" / "fixtures" / "mock_suppliers.json"
)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Ensure OPENAI_API_KEY from .env is available for non-CLI code paths too.
load_dotenv()


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def _normalize_score(raw_0_1: float) -> tuple[int, str]:
    """Convert a 0.0-1.0 float to (0-100 int, Low/Medium/High level)."""
    score_100 = round(raw_0_1 * 100)
    score_100 = max(0, min(100, score_100))
    if score_100 <= 33:
        level = "Low"
    elif score_100 <= 66:
        level = "Medium"
    else:
        level = "High"
    return score_100, level


def _build_hazard_score(
    supplier_name: str,
    hazard_type: str,
    raw_score: float,
    metadata: dict,
    raw_value: float = 0.0,
) -> HazardScore:
    """Create a HazardScore with normalised score_100 and level.

    raw_score is the percentile rank H (0–1); raw_value is the physical quantity.
    """
    score = round(raw_score, 4)
    score_100, level = _normalize_score(raw_score)
    return HazardScore(
        supplier_name=supplier_name,
        hazard_type=hazard_type,
        score=score,
        score_100=score_100,
        level=level,
        raw_value=raw_value,
        dataset_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Shared HTTP helper (SSL certificate fallback)
# ---------------------------------------------------------------------------


def _urlopen_with_ssl_fallback(url: str, timeout: int = 15) -> bytes:
    """Fetch URL content, retrying with unverified SSL on cert errors.

    Some local environments (e.g. macOS) fail CA verification for
    certain HTTPS endpoints.  When that happens, retry once with an
    unverified context so the agent still returns real data.
    Re-raises non-SSL errors so callers can apply their own fallback.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
                return resp.read()
        raise


# ---------------------------------------------------------------------------
# SQLite grid helpers (read-only; thread-safe via check_same_thread=False)
# ---------------------------------------------------------------------------

# DB path constants — resolved relative to the repo data/ directory.
_GEM_DB = str(_DATA_DIR / "gem_seismic.db")
_FLOOD_DB = str(_DATA_DIR / "aqueduct_flood.db")
_CYCLONE_DB = str(_DATA_DIR / "storm_cyclone.db")


@lru_cache(maxsize=8)
def _open_grid_db(path: str) -> sqlite3.Connection:
    """Open a read-only grid database, safe for multi-threaded reads.

    Connection policy: one connection per DB path, cached for process lifetime.
    All grid lookups are reads; no INSERT/UPDATE/VACUUM/CREATE occurs at runtime.
    check_same_thread=False is safe for concurrent reads on a single SQLite file.
    If a DB file is regenerated, restart the process or call _open_grid_db.cache_clear()
    (tests that regenerate fixture DBs must call cache_clear() in teardown).
    """
    return sqlite3.connect(path, check_same_thread=False)


def _db_lookup(db_path: str, lat_key: int, lon_key: int) -> float:
    """Point lookup using integer grid keys. Returns 0.0 for absent cells."""
    conn = _open_grid_db(db_path)
    row = conn.execute(
        "SELECT value FROM grid WHERE lat_key=? AND lon_key=?", (lat_key, lon_key)
    ).fetchone()
    return float(row[0]) if row else 0.0


# ---------------------------------------------------------------------------
# API raw-value cache (wildfire, heat_stress, drought)
# Ensures experiment reproducibility: API results are frozen after first fetch.
# ---------------------------------------------------------------------------

_API_CACHE_PATH = _DATA_DIR / "api_hazard_cache.json"
_api_cache_lock = threading.Lock()
_api_cache: dict[str, float] | None = None   # loaded lazily


def _get_api_cache() -> dict[str, float]:
    global _api_cache
    if _api_cache is None:
        with _api_cache_lock:
            if _api_cache is None:
                if _API_CACHE_PATH.exists():
                    try:
                        _api_cache = json.loads(_API_CACHE_PATH.read_text())
                    except (json.JSONDecodeError, OSError):
                        _api_cache = {}
                else:
                    _api_cache = {}
    return _api_cache


def _api_cache_lookup(hazard: str, lat: float, lon: float, method_version: str) -> float | None:
    key = f"{hazard}|{lat}|{lon}|1991-2020|{method_version}"
    return _get_api_cache().get(key)


def _api_cache_store(hazard: str, lat: float, lon: float, method_version: str, value: float) -> None:
    key = f"{hazard}|{lat}|{lon}|1991-2020|{method_version}"
    cache = _get_api_cache()
    with _api_cache_lock:
        cache[key] = value
        try:
            _API_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _API_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
        except OSError as e:
            logger.warning("Failed to write API hazard cache: %s", e)


# ---------------------------------------------------------------------------
# Non-finite value guard (runtime)
# ---------------------------------------------------------------------------

def _guard_raw(hazard: str, supplier_name: str, lat: float, lon: float, value: float) -> float:
    """Return value if finite and non-negative, else 0.0 with a warning."""
    if not math.isfinite(value) or value < 0.0:
        logger.warning(
            "%s raw value non-finite for '%s' at (%.4f, %.4f); using 0.0",
            hazard, supplier_name, lat, lon,
        )
        return 0.0
    return value


# ---------------------------------------------------------------------------
# Earthquake scoring (GEM 2023 Global Seismic Hazard, SQLite)
# ---------------------------------------------------------------------------


def _score_earthquake(supplier: Supplier) -> HazardScore:
    """GEM 2023 PGA at 10%/50yr exceedance (475-yr RP), in g. 0.1-degree grid."""
    if not _DATA_DIR.joinpath("gem_seismic.db").exists():
        logger.warning("gem_seismic.db not found; earthquake score=0.0 for '%s'", supplier.name)
        pga_g = 0.0
    else:
        pga_g = _db_lookup(_GEM_DB, grid_key(supplier.lat, 1), grid_key(supplier.lon, 1))
    ref = load_reference_set()["hazards"]["earthquake"]
    if ref[0] == ref[-1]:
        logger.warning("Earthquake reference array is constant (non-informative)")
    h = percentile_rank(pga_g, ref)
    return _build_hazard_score(
        supplier.name, "earthquake", h, raw_value=pga_g,
        metadata={
            "dataset": "GEM_2023",
            "pga_g": pga_g,
            "return_period_yr": 475,
            "grid_resolution_deg": 0.1,
            "source": "GEM Global Seismic Hazard Model 2023 (Pagani et al. 2018)",
        },
    )


# ---------------------------------------------------------------------------
# Flood scoring (WRI Aqueduct Floods, SQLite, 0.01-degree grid)
# ---------------------------------------------------------------------------


def _score_flood(supplier: Supplier) -> HazardScore:
    """WRI Aqueduct 100-yr river flood inundation depth (m). 0.01-degree grid."""
    if not _DATA_DIR.joinpath("aqueduct_flood.db").exists():
        logger.warning("aqueduct_flood.db not found; flood score=0.0 for '%s'", supplier.name)
        depth_m = 0.0
    else:
        depth_m = _db_lookup(_FLOOD_DB, grid_key(supplier.lat, 2), grid_key(supplier.lon, 2))
    ref = load_reference_set()["hazards"]["flood"]
    if ref[0] == ref[-1]:
        logger.warning("Flood reference array is constant (non-informative)")
    h = percentile_rank(depth_m, ref)
    return _build_hazard_score(
        supplier.name, "flood", h, raw_value=depth_m,
        metadata={
            "dataset": "Aqueduct_Floods",
            "depth_m": depth_m,
            "return_period_yr": 100,
            "grid_resolution_deg": 0.01,
            "source": "WRI Aqueduct Floods (Winsemius et al. 2018)",
        },
    )


# ---------------------------------------------------------------------------
# Heat stress scoring (Open-Meteo ERA5 apparent_temperature_max, API-based)
# Canonical citation: Gasparrini et al. (2015) for heat-health context only.
# 32 deg C is an operational screening threshold used in this study.
# ---------------------------------------------------------------------------

_HEAT_METHOD_VERSION = "apparent_temp_32_v1"


def _fetch_heat_fraction(lat: float, lon: float) -> float:
    """Fraction of days with apparent_temperature_max >= 32 deg C (1991-2020).

    Heat exposure is defined as the fraction of days with apparent_temperature_max
    >= 32 deg C over 1991-2020, derived from Open-Meteo ERA5. The 32 deg C value
    is used as an operational screening threshold in this study; it is not
    presented as a universal biological cutoff. Gasparrini et al. (2015) is
    cited for general heat-health context only. NOT UTCI.
    """
    cached = _api_cache_lookup("heat_stress", lat, lon, _HEAT_METHOD_VERSION)
    if cached is not None:
        return cached

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=apparent_temperature_max"
        "&start_date=1991-01-01&end_date=2020-12-31&timezone=UTC"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=30)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError) as e:
        logger.warning("heat_stress API call failed for (%.4f, %.4f): %s", lat, lon, e)
        return 0.0

    values = [v for v in data.get("daily", {}).get("apparent_temperature_max", []) if v is not None]
    if not values:
        return 0.0

    fraction = sum(1 for v in values if v >= 32.0) / len(values)
    _api_cache_store("heat_stress", lat, lon, _HEAT_METHOD_VERSION, fraction)
    return fraction


def _score_heat_stress(supplier: Supplier) -> HazardScore:
    fraction = _fetch_heat_fraction(supplier.lat, supplier.lon)
    fraction = _guard_raw("heat_stress", supplier.name, supplier.lat, supplier.lon, fraction)
    ref = load_reference_set()["hazards"]["heat_stress"]
    if ref[0] == ref[-1]:
        logger.warning("Heat stress reference array is constant (non-informative)")
    h = percentile_rank(fraction, ref)
    return _build_hazard_score(
        supplier.name, "heat_stress", h, raw_value=fraction,
        metadata={
            "dataset": "Open-Meteo ERA5",
            "heat_metric": "apparent_temperature_max >= 32C",
            "metric_source": "Open-Meteo ERA5",
            "threshold_note": (
                "32 deg C is an operational screening threshold used in this study; "
                "not a universal biological cutoff. See Gasparrini et al. (2015) "
                "for general heat-health context."
            ),
            "period": "1991-01-01 to 2020-12-31",
            "method_version": _HEAT_METHOD_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Drought scoring (SPEI-12 via ERA5 precip + ET0, Open-Meteo, API-based)
# Vicente-Serrano et al. (2010). 3-parameter log-logistic (scipy.stats.fisk).
# ---------------------------------------------------------------------------

_DROUGHT_METHOD_VERSION = "spei12_v1"
_MIN_SPEI_MONTHS = 180   # ~15 years minimum for a 12-month SPEI fit


def _compute_spei12_fraction(monthly_precip_mm: list[float], monthly_et0_mm: list[float]) -> float:
    """Compute fraction of months with SPEI-12 <= -1 (Vicente-Serrano et al. 2010).

    Uses ERA5 monthly precipitation minus potential evapotranspiration (P-ET0),
    aggregated over 12-month windows and standardised with a 3-parameter
    log-logistic distribution (scipy.stats.fisk).
    """
    from scipy import stats as scipy_stats  # type: ignore

    n = min(len(monthly_precip_mm), len(monthly_et0_mm))
    if n < _MIN_SPEI_MONTHS:
        raise ValueError(
            f"Insufficient monthly data for SPEI-12: {n} months "
            f"(need >= {_MIN_SPEI_MONTHS}). Cannot fit 12-month standardised index."
        )

    # Water balance: D_i = P_i - ET0_i
    d = [monthly_precip_mm[i] - monthly_et0_mm[i] for i in range(n)]

    # 12-month running sum (SPEI-12 accumulation period)
    d12 = [sum(d[i - 11: i + 1]) for i in range(11, n)]
    if not d12:
        return 0.0

    # Shift so all values are positive before fitting log-logistic
    shift = min(d12) - 1.0
    d12_shifted = [v - shift for v in d12]

    # Fit 3-parameter log-logistic (fisk) and compute CDF
    try:
        c, loc, scale = scipy_stats.fisk.fit(d12_shifted, floc=0)
        cdf_vals = scipy_stats.fisk.cdf(d12_shifted, c, loc=loc, scale=scale)
    except Exception:
        return 0.0

    # SPEI = Phi^{-1}(F(D)) where Phi is standard normal CDF
    spei = scipy_stats.norm.ppf(cdf_vals)

    drought_months = sum(1 for s in spei if s <= -1.0)
    return drought_months / len(spei)


def _fetch_spei12_fraction(lat: float, lon: float) -> float:
    """Fetch ERA5 precip + ET0 via Open-Meteo and compute SPEI-12 fraction."""
    cached = _api_cache_lookup("drought", lat, lon, _DROUGHT_METHOD_VERSION)
    if cached is not None:
        return cached

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum,et0_fao_evapotranspiration"
        "&start_date=1991-01-01&end_date=2020-12-31&timezone=UTC"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=30)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError) as e:
        logger.warning("drought API call failed for (%.4f, %.4f): %s", lat, lon, e)
        return 0.0

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    precip_daily = daily.get("precipitation_sum", [])
    et0_daily = daily.get("et0_fao_evapotranspiration", [])

    # Aggregate to monthly totals
    monthly_precip: dict[str, float] = {}
    monthly_et0: dict[str, float] = {}
    for i, date_str in enumerate(dates):
        month_key = date_str[:7]
        p = precip_daily[i] if i < len(precip_daily) and precip_daily[i] is not None else 0.0
        e = et0_daily[i] if i < len(et0_daily) and et0_daily[i] is not None else 0.0
        monthly_precip[month_key] = monthly_precip.get(month_key, 0.0) + p
        monthly_et0[month_key] = monthly_et0.get(month_key, 0.0) + e

    months = sorted(monthly_precip.keys())
    p_list = [monthly_precip[m] for m in months]
    e_list = [monthly_et0[m] for m in months]

    try:
        fraction = _compute_spei12_fraction(p_list, e_list)
    except ValueError as e:
        logger.warning("SPEI-12 computation failed for (%.4f, %.4f): %s; using 0.0", lat, lon, e)
        return 0.0

    _api_cache_store("drought", lat, lon, _DROUGHT_METHOD_VERSION, fraction)
    return fraction


def _score_drought(supplier: Supplier) -> HazardScore:
    fraction = _fetch_spei12_fraction(supplier.lat, supplier.lon)
    fraction = _guard_raw("drought", supplier.name, supplier.lat, supplier.lon, fraction)
    ref = load_reference_set()["hazards"]["drought"]
    if ref[0] == ref[-1]:
        logger.warning("Drought reference array is constant (non-informative)")
    h = percentile_rank(fraction, ref)
    return _build_hazard_score(
        supplier.name, "drought", h, raw_value=fraction,
        metadata={
            "dataset": "Open-Meteo ERA5",
            "indicator": "SPEI-12",
            "metric": "fraction months SPEI-12 <= -1",
            "period": "1991-01-01 to 2020-12-31",
            "method": "3-parameter log-logistic (scipy.stats.fisk)",
            "source": "Vicente-Serrano et al. (2010)",
            "method_version": _DROUGHT_METHOD_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Wildfire scoring (Approx. FWI Q95, ERA5 daily stats via Open-Meteo, API-based)
# Label used everywhere: "Approx. FWI Q95 (daily ERA5 stats proxy)"
# Caveat: ERA5 daily stats are an approximation of the standard noon-condition
# Canadian FWI workflow (Van Wagner 1987). Not strict FWI.
# ---------------------------------------------------------------------------

_WILDFIRE_METHOD_VERSION = "fwi_daily_stats_v1"


def _compute_fwi_series(
    temps: list[float],
    rhs: list[float],
    winds: list[float],
    precips: list[float],
) -> list[float]:
    """Approximate daily FWI series using Van Wagner (1987) recurrences.

    Inputs are ERA5 daily max temperature (deg C), min relative humidity (%),
    max wind speed (km/h), and precipitation sum (mm).  This is an approximation
    of the standard Canadian FWI workflow, which requires noon-condition
    observations; ERA5 daily statistics are a proxy.
    """
    n = min(len(temps), len(rhs), len(winds), len(precips))
    if n == 0:
        return []

    # Initial moisture code values (Van Wagner 1987 defaults)
    FFMC = 85.0   # Fine Fuel Moisture Code
    DMC = 6.0     # Duff Moisture Code
    DC = 15.0     # Drought Code

    fwi_series: list[float] = []

    for i in range(n):
        T = temps[i]
        H = max(1.0, min(100.0, rhs[i]))
        W = max(0.0, winds[i])
        ro = max(0.0, precips[i])

        # ---- FFMC (Fine Fuel Moisture Code) ----
        mo = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
        if ro > 0.5:
            rf = ro - 0.5
            if mo <= 150.0:
                mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
            else:
                mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                mr = min(mr, 250.0) if mo > 150.0 else mr
            mo = mr

        Ed = 0.942 * H ** 0.679 + 11.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
        Ew = 0.618 * H ** 0.753 + 10.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))

        if mo > Ed:
            ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * W ** 0.5 * (1.0 - (H / 100.0) ** 8)
            kd = ko * 0.581 * math.exp(0.0365 * T)
            m = Ed + (mo - Ed) * 10.0 ** (-kd)
        elif mo < Ew:
            kl = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * W ** 0.5 * (1.0 - ((100.0 - H) / 100.0) ** 8)
            kw = kl * 0.581 * math.exp(0.0365 * T)
            m = Ew - (Ew - mo) * 10.0 ** (-kw)
        else:
            m = mo

        FFMC = 59.5 * (250.0 - m) / (147.2 + m)
        FFMC = max(0.0, min(101.0, FFMC))

        # ---- DMC (Duff Moisture Code) ----
        L_e = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        month_idx = (i // 30) % 12
        Le = L_e[month_idx]

        K = 1.894 * (T + 1.1) * (100.0 - H) * Le * 1e-6
        if ro > 1.5:
            re = 0.92 * ro - 1.27
            Mo = 20.0 + math.exp(5.6348 - DMC / 43.43)
            if DMC <= 33.0:
                b = 100.0 / (0.5 + 0.3 * DMC)
            elif DMC <= 65.0:
                b = 14.0 - 1.3 * math.log(DMC)
            else:
                b = 6.2 * math.log(DMC) - 17.2
            Mr = Mo + 1000.0 * re / (48.77 + b * re)
            Pr = 43.43 * (5.6348 - math.log(Mr - 20.0)) if Mr > 20.0 else 0.0
            DMC = max(0.0, Pr) + K
        else:
            DMC = DMC + K
        DMC = max(0.0, DMC)

        # ---- DC (Drought Code) ----
        Lf = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        Lf_val = Lf[month_idx]
        V = 0.36 * (T + 2.8) + Lf_val
        if V < 0:
            V = 0.0
        if ro > 2.8:
            rd = 0.83 * ro - 1.27
            Qo = 800.0 * math.exp(-DC / 400.0)
            Qr = Qo + 3.937 * rd
            Dr = 400.0 * math.log(800.0 / Qr) if Qr > 0 else 0.0
            DC = max(0.0, Dr) + V / 2.0
        else:
            DC = DC + V / 2.0
        DC = max(0.0, DC)

        # ---- ISI (Initial Spread Index) ----
        fw = math.exp(0.05039 * W)
        ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
        ISI = 0.208 * fw * ff

        # ---- BUI (Buildup Index) ----
        if DMC <= 0.4 * DC:
            BUI = 0.8 * DMC * DC / (DMC + 0.4 * DC) if (DMC + 0.4 * DC) > 0 else 0.0
        else:
            BUI = DMC - (1.0 - 0.8 * DC / (DMC + 0.4 * DC)) * (0.92 + (0.0114 * DMC) ** 1.7) if (DMC + 0.4 * DC) > 0 else 0.0
        BUI = max(0.0, BUI)

        # ---- FWI (Fire Weather Index) ----
        if BUI <= 80.0:
            fD = 0.626 * BUI ** 0.809 + 2.0
        else:
            fD = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * BUI))
        B = 0.1 * ISI * fD
        if B > 1.0:
            FWI = math.exp(2.72 * (0.434 * math.log(B)) ** 0.647)
        else:
            FWI = B
        FWI = max(0.0, FWI)
        fwi_series.append(FWI)

    return fwi_series


def _fetch_approx_fwi_q95(lat: float, lon: float) -> float:
    """Q95 of approximate daily FWI series (1991-2020) from ERA5 via Open-Meteo.

    Label: 'Approx. FWI Q95 (daily ERA5 stats proxy)'
    Thesis: FWI is computed from ERA5 daily statistics via Van Wagner (1987) recurrences.
    This is an approximation; noon-condition observations are not available.
    """
    cached = _api_cache_lookup("wildfire", lat, lon, _WILDFIRE_METHOD_VERSION)
    if cached is not None:
        return cached

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,relative_humidity_2m_min"
        ",wind_speed_10m_max,precipitation_sum"
        "&start_date=1991-01-01&end_date=2020-12-31&timezone=UTC"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=30)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError) as e:
        logger.warning("wildfire API call failed for (%.4f, %.4f): %s", lat, lon, e)
        return 0.0

    daily = data.get("daily", {})
    temps = [v if v is not None else 20.0 for v in daily.get("temperature_2m_max", [])]
    rhs = [v if v is not None else 50.0 for v in daily.get("relative_humidity_2m_min", [])]
    winds = [v if v is not None else 10.0 for v in daily.get("wind_speed_10m_max", [])]
    precips = [v if v is not None else 0.0 for v in daily.get("precipitation_sum", [])]

    fwi_vals = _compute_fwi_series(temps, rhs, winds, precips)
    if not fwi_vals:
        return 0.0

    fwi_sorted = sorted(fwi_vals)
    q95 = fwi_sorted[int(0.95 * len(fwi_sorted))]
    _api_cache_store("wildfire", lat, lon, _WILDFIRE_METHOD_VERSION, q95)
    return q95


def _score_wildfire(supplier: Supplier) -> HazardScore:
    fwi_q95 = _fetch_approx_fwi_q95(supplier.lat, supplier.lon)
    fwi_q95 = _guard_raw("wildfire", supplier.name, supplier.lat, supplier.lon, fwi_q95)
    ref = load_reference_set()["hazards"]["wildfire"]
    if ref[0] == ref[-1]:
        logger.warning("Wildfire reference array is constant (non-informative)")
    h = percentile_rank(fwi_q95, ref)
    return _build_hazard_score(
        supplier.name, "wildfire", h, raw_value=fwi_q95,
        metadata={
            "dataset": "Open-Meteo ERA5",
            "indicator": "Approx. FWI Q95 (daily ERA5 stats proxy)",
            "fwi_computation": "daily_stats_approximation",
            "period": "1991-01-01 to 2020-12-31",
            "method": "Van Wagner (1987) FWI recurrences on ERA5 daily stats",
            "caveat": (
                "ERA5 daily max/min fields are an approximation of the standard "
                "noon-condition Canadian FWI workflow (Van Wagner 1987). "
                "Strict FWI benchmarks are not applicable to this computation."
            ),
            "method_version": _WILDFIRE_METHOD_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Cyclone scoring (STORM dataset – Bloemendaal et al. 2020, SQLite)
# ---------------------------------------------------------------------------


def _score_cyclone(supplier: Supplier) -> HazardScore:
    """STORM 100-yr return period max sustained wind speed (km/h). 0.1-degree grid."""
    if not _DATA_DIR.joinpath("storm_cyclone.db").exists():
        logger.warning("storm_cyclone.db not found; cyclone score=0.0 for '%s'", supplier.name)
        v100_kmh = 0.0
    else:
        v100_kmh = _db_lookup(_CYCLONE_DB, grid_key(supplier.lat, 1), grid_key(supplier.lon, 1))
    ref = load_reference_set()["hazards"]["cyclone"]
    if ref[0] == ref[-1]:
        logger.warning("Cyclone reference array is constant (non-informative)")
    h = percentile_rank(v100_kmh, ref)
    return _build_hazard_score(
        supplier.name, "cyclone", h, raw_value=v100_kmh,
        metadata={
            "dataset": "STORM_2020",
            "v100_kmh": v100_kmh,
            "return_period_yr": 100,
            "grid_resolution_deg": 0.1,
            "source": "STORM (Bloemendaal et al. 2020)",
        },
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

_HAZARD_SCORERS = {
    "earthquake": _score_earthquake,
    "flood": _score_flood,
    "heat_stress": _score_heat_stress,
    "drought": _score_drought,
    "wildfire": _score_wildfire,
    "cyclone": _score_cyclone,
}


def compute_hazard(supplier: Supplier, hazard_type: str = "earthquake") -> HazardScore:
    """Return a hazard score for *supplier*.

    Dispatches to a real scoring function for each hazard type.
    Falls back to a deterministic SHA-256 hash stub for unknown types.
    """
    scorer = _HAZARD_SCORERS.get(hazard_type)
    if scorer is not None:
        return scorer(supplier)

    # Fallback: deterministic hash stub for unrecognised hazard types
    digest = hashlib.sha256(
        f"{supplier.name}:{hazard_type}".encode()
    ).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF  # 0-1

    return _build_hazard_score(
        supplier.name,
        hazard_type,
        raw,
        {
            "dataset": f"stub_{hazard_type}_v1",
            "version": "0.1.0",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "sha256_hash_stub",
        },
    )


# ---------------------------------------------------------------------------
# Risk aggregation (pure function, used by graph node + sensitivity analysis)
# ---------------------------------------------------------------------------


N_HAZARDS = 6  # SMHEI is always defined over exactly 6 hazards


def compute_risk_summary(
    hazard_scores: list[dict],
    suppliers: list[dict],
    weights: dict[str, float] | None = None,
    aggregation: str = "arithmetic",
) -> dict:
    """Compute Supplier Multi-Hazard Exposure Index (SMHEI) from hazard scores.

    Primary: equal-weight arithmetic mean over N_HAZARDS=6 hazards.
    Sensitivity: geometric mean (aggregation="geometric") or custom weights.

    Parameters
    ----------
    hazard_scores : list[dict]
        HazardScore dicts from the parallel scorer nodes.
    suppliers : list[dict]
        Supplier metadata dicts (name, tier, confidence, evidence_source).
    weights : dict[str, float] | None
        None = equal 1/N_HAZARDS for all hazards (primary SMHEI).
        For sensitivity scenarios, pass explicit weight maps; unknown/negative
        weights raise ValueError immediately.
    aggregation : str
        "arithmetic" (primary) or "geometric" (sensitivity-only).
        Geometric mean uses equal weights and no epsilon smoothing; zero H gives E_s=0.
    """
    reference_set = load_reference_set()
    supplier_thresholds = reference_set["supplier_exposure_thresholds"]
    company_thresholds = reference_set["company_exposure_thresholds"]

    supplier_meta = {s["name"]: s for s in suppliers}

    # Group hazard scores by supplier
    by_supplier: dict[str, dict[str, float]] = {}
    for hs in hazard_scores:
        by_supplier.setdefault(hs["supplier_name"], {})[hs["hazard_type"]] = float(hs["score"])

    supplier_risks: list[dict] = []

    for supplier_name, hazard_h in by_supplier.items():
        # --- Validate completeness ---
        if weights is None and len(hazard_h) != N_HAZARDS:
            raise ValueError(
                f"Expected {N_HAZARDS} hazard scores for supplier '{supplier_name}', "
                f"got {len(hazard_h)}: {sorted(hazard_h)}"
            )

        # --- Build weight map ---
        if weights is None:
            w = {h: 1.0 / N_HAZARDS for h in hazard_h}
        else:
            unknown = set(weights) - set(hazard_h)
            if unknown:
                raise ValueError(
                    f"Weight map contains unknown hazard names: {sorted(unknown)}. "
                    f"Valid names: {sorted(hazard_h)}"
                )
            if any(v < 0 for v in weights.values()):
                raise ValueError(
                    "Weights must be non-negative; got: "
                    + ", ".join(f"{k}={v}" for k, v in weights.items() if v < 0)
                )
            total_w = sum(weights.get(h, 0.0) for h in hazard_h)
            if total_w == 0.0:
                raise ValueError(
                    "All weights sum to zero — cannot compute weighted mean. "
                    "At least one hazard must have a positive weight."
                )
            w = weights

        # --- Aggregate ---
        if aggregation == "geometric":
            # True geometric mean — no epsilon. Zero H gives E_s=0 (intentional).
            E_s = math.prod(h for h in hazard_h.values()) ** (1.0 / N_HAZARDS)
        else:
            total_w = sum(w.get(h, 0.0) for h in hazard_h)
            E_s = sum(hazard_h[h] * w.get(h, 0.0) for h in hazard_h) / total_w

        M_s = max(hazard_h.values())
        dominant_hazard = max(hazard_h, key=hazard_h.get)

        # --- Supplier exposure band ---
        if E_s >= supplier_thresholds["high"]:
            exposure_band = "High"
        elif E_s >= supplier_thresholds["medium"]:
            exposure_band = "Medium"
        else:
            exposure_band = "Low"

        info = supplier_meta.get(supplier_name, {})
        tier = int(info.get("tier", 1))
        confidence = float(info.get("confidence", 1.0))

        supplier_risks.append({
            "supplier_name": supplier_name,
            "exposure_index": round(E_s, 4),
            "exposure_max": round(M_s, 4),
            "dominant_hazard": dominant_hazard,
            "exposure_band": exposure_band,
            "tier": tier,
            "confidence": confidence,
        })

    supplier_risks.sort(key=lambda x: x["exposure_index"], reverse=True)

    # --- Company-level aggregate ---
    company_score = 0.0
    if supplier_risks:
        company_score = sum(s["exposure_index"] for s in supplier_risks) / len(supplier_risks)
    company_score = round(company_score, 4)

    if company_score >= company_thresholds["high"]:
        company_band = "High"
    elif company_score >= company_thresholds["medium"]:
        company_band = "Medium"
    else:
        company_band = "Low"

    return {
        "company_score": company_score,
        "company_band": company_band,
        "supplier_risks": supplier_risks,
    }


# ---------------------------------------------------------------------------
# Supplier loading (JSON fixtures)
# ---------------------------------------------------------------------------


def load_suppliers(
    company: str,
    tier_depth: int,
    suppliers_path: Path | None = None,
) -> tuple[list[Supplier], list[dict], list[dict]]:
    """Load suppliers for *company* from a JSON file.

    Returns ``(suppliers, edges, evidence)`` filtered to *tier_depth*.
    Raises :class:`KeyError` if *company* is not found in the file.
    """
    path = Path(suppliers_path) if suppliers_path is not None else _DEFAULT_SUPPLIERS_PATH
    data = json.loads(path.read_text())
    source_label = "Supplier relationship dataset"

    if company not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Company '{company}' not found in {path}. "
            f"Available companies: {available}"
        )

    entry = data[company]
    now = datetime.now(timezone.utc).isoformat()

    # Build evidence and edges from raw edge descriptions
    evidence: list[dict] = []
    edges: list[dict] = []
    child_evidence_map: dict[str, list[str]] = {}

    for i, raw_edge in enumerate(entry["edges"], start=1):
        eid = f"E{i}"
        child_name = raw_edge["child"]

        evidence.append({
            "evidence_id": eid,
            "source": source_label,
            "description": raw_edge["description"],
            "retrieved_at": now,
        })
        edges.append({
            "parent": raw_edge["parent"],
            "child": child_name,
            "evidence_ids": [eid],
        })
        child_evidence_map.setdefault(child_name, []).append(eid)

    # Build Supplier objects filtered by tier_depth
    suppliers: list[Supplier] = []
    for raw in entry["suppliers"]:
        if raw["tier"] > tier_depth:
            continue
        suppliers.append(Supplier(
            name=raw["name"],
            lat=raw["lat"],
            lon=raw["lon"],
            tier=raw["tier"],
            confidence=raw["confidence"],
            evidence_ids=child_evidence_map.get(raw["name"], []),
            industry=raw.get("industry", ""),
            product_category=raw.get("product_category", ""),
            location_description=raw.get("location_description", ""),
            relationship_type=raw.get("relationship_type", ""),
            evidence_source="fixture",
        ))

    # Prune edges and evidence to only surviving suppliers
    surviving = {s.name for s in suppliers}
    edges = [e for e in edges if e["child"] in surviving]
    surviving_eids = {eid for e in edges for eid in e["evidence_ids"]}
    evidence = [ev for ev in evidence if ev["evidence_id"] in surviving_eids]

    return suppliers, edges, evidence


# ---------------------------------------------------------------------------
# LLM-based company resolution
# ---------------------------------------------------------------------------


def resolve_company_profile(company: str) -> dict:
    """Resolve a company name to a structured profile using GPT-4o.

    Returns a dict with canonical_name, industry, products, headquarters,
    and description.  Used to provide context for supplier discovery.
    """
    prompts = load_prompts()
    template = prompts["company_profile_prompt"]

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=60,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(CompanyProfile)

    prompt_text = template.format(company=company)
    response: CompanyProfile = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return response.model_dump()


# ---------------------------------------------------------------------------
# LLM-based supplier discovery
# ---------------------------------------------------------------------------


def _deduplicate_suppliers(
    suppliers: list[Supplier],
) -> list[Supplier]:
    """Merge duplicate suppliers (same name, case-insensitive).

    Keeps the entry with the highest confidence and merges evidence_ids
    from all occurrences.
    """
    seen: dict[str, Supplier] = {}

    for s in suppliers:
        key = s.name.strip().lower()
        if key in seen:
            existing = seen[key]
            merged_eids = list(
                dict.fromkeys(existing.evidence_ids + s.evidence_ids)
            )
            if s.confidence > existing.confidence:
                seen[key] = s.model_copy(update={"evidence_ids": merged_eids})
            else:
                seen[key] = existing.model_copy(update={"evidence_ids": merged_eids})
        else:
            seen[key] = s

    return list(seen.values())


def discover_suppliers_llm(
    company: str,
    tier_depth: int,
    company_profile: dict | None = None,
) -> tuple[list[Supplier], list[dict], list[dict]]:
    """Discover suppliers for *company* using GPT-4o structured output.

    Makes one LLM call per (parent, tier) pair.  Returns
    ``(suppliers, edges, evidence)`` with evidence citing the LLM.

    Parameters
    ----------
    company_profile : dict | None
        If provided, injects industry and product context into the
        tier expansion prompt for more accurate discovery.
    """
    prompts = load_prompts()
    template = prompts["tier_expansion_prompt"]

    # Build context from company profile if available.
    industry = "Unknown"
    products_str = "Unknown"
    if company_profile:
        industry = company_profile.get("industry", "Unknown")
        products = company_profile.get("products", [])
        products_str = ", ".join(products) if products else "Unknown"

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(TierResponse)

    all_suppliers: list[Supplier] = []
    all_edges: list[dict] = []
    all_evidence: list[dict] = []
    evidence_counter = 0

    # Tier 1 parents = [company]; tier N parents = tier N-1 supplier names
    parents: list[str] = [company]

    for tier in range(1, tier_depth + 1):
        next_parents: list[str] = []

        for parent in parents:
            prompt_text = template.format(
                company=parent,
                tier=tier,
                industry=industry,
                products=products_str,
            )
            response: TierResponse = structured_llm.invoke(
                [HumanMessage(content=prompt_text)]
            )

            now = datetime.now(timezone.utc).isoformat()

            for llm_sup in response.suppliers:
                evidence_counter += 1
                eid = f"E{evidence_counter}"

                all_suppliers.append(Supplier(
                    name=llm_sup.name,
                    lat=llm_sup.lat,
                    lon=llm_sup.lon,
                    tier=tier,
                    confidence=llm_sup.confidence,
                    evidence_ids=[eid],
                    industry=llm_sup.industry,
                    product_category=llm_sup.product_category,
                    location_description=llm_sup.location_description,
                    relationship_type=llm_sup.relationship_type,
                    evidence_source="llm_only",
                ))
                all_edges.append({
                    "parent": parent,
                    "child": llm_sup.name,
                    "evidence_ids": [eid],
                })
                all_evidence.append({
                    "evidence_id": eid,
                    "source": "LLM:gpt-4o",
                    "description": llm_sup.rationale,
                    "retrieved_at": now,
                })
                next_parents.append(llm_sup.name)

        parents = next_parents

    all_suppliers = _deduplicate_suppliers(all_suppliers)

    return all_suppliers, all_edges, all_evidence


# ---------------------------------------------------------------------------
# LLM-based mitigation generation
# ---------------------------------------------------------------------------


def _build_risk_context(
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> str:
    """Build a plain-text context block for the mitigation LLM prompt."""
    lines: list[str] = []

    # Company-level summary
    lines.append(f"Company exposure score: {summary.get('company_score', 0):.4f}")
    lines.append(f"Company exposure band: {summary.get('company_band', 'unknown')}")
    lines.append("")

    # Supplier details
    supplier_map = {s["name"]: s for s in suppliers}
    lines.append("Suppliers:")
    for sr in summary.get("supplier_risks", []):
        name = sr["supplier_name"]
        info = supplier_map.get(name, {})
        lines.append(
            f"  - {name}: lat={info.get('lat', '?')}, lon={info.get('lon', '?')}, "
            f"tier={sr['tier']}, confidence={sr['confidence']:.2f}, "
            f"exposure_index={sr['exposure_index']:.4f}, "
            f"dominant_hazard={sr.get('dominant_hazard', '?')}"
        )
    lines.append("")

    # Medium/High hazard scores with metadata
    lines.append("Hazard scores (Medium and High only):")
    for h in hazard_scores:
        if h.get("level", "Low") not in ("Medium", "High"):
            continue
        meta = h.get("dataset_metadata", {})
        meta_parts = []
        for k, v in meta.items():
            if k in ("dataset", "source", "method", "version", "retrieved_at"):
                continue
            meta_parts.append(f"{k}={v}")
        lines.append(
            f"  - {h['supplier_name']} / {h['hazard_type']}: "
            f"score={h.get('score_100', 0)}/100 ({h.get('level', '')})"
            + (f" [{', '.join(meta_parts)}]" if meta_parts else "")
        )
    lines.append("")

    # High exposure band suppliers
    high_exposure = [
        sr for sr in summary.get("supplier_risks", [])
        if sr.get("exposure_band") == "High"
    ]
    if high_exposure:
        lines.append("High-exposure suppliers:")
        for sr in high_exposure:
            lines.append(
                f"  - {sr['supplier_name']}: E_s={sr['exposure_index']:.4f}, "
                f"dominant={sr.get('dominant_hazard', '?')}"
            )

    return "\n".join(lines)


def generate_mitigations_llm(
    company: str,
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> list[dict]:
    """Generate mitigation strategies using GPT-4o structured output."""
    prompts = load_prompts()
    template = prompts["mitigation_prompt"]

    risk_context = _build_risk_context(suppliers, hazard_scores, summary)
    prompt_text = template.format(company=company, risk_context=risk_context)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(MitigationResponse)

    response: MitigationResponse = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return [m.model_dump() for m in response.mitigations]


# ---------------------------------------------------------------------------
# LLM-based alternative supplier suggestions
# ---------------------------------------------------------------------------


def suggest_alternatives_llm(
    company: str,
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> list[dict]:
    """Suggest lower-risk alternative suppliers using GPT-4o structured output."""
    prompts = load_prompts()
    template = prompts["suggest_alternatives_prompt"]

    risk_context = _build_risk_context(suppliers, hazard_scores, summary)
    prompt_text = template.format(company=company, risk_context=risk_context)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(AlternativeResponse)

    response: AlternativeResponse = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return [a.model_dump() for a in response.alternatives]


# ---------------------------------------------------------------------------
# Pipeline web verification (Condition B — deterministic order)
# ---------------------------------------------------------------------------

_RELATIONSHIP_CUES = frozenset({
    "supplier", "vendor", "manufactures", "supplies",
    "partner", "contract", "sources from", "procures from",
    "supply chain", "component supplier",
})


def verify_suppliers_batch(
    suppliers: list[dict],
    company: str,
    budget: "BudgetTracker",  # noqa: F821 — forward ref
    snapshot_mode: bool = False,
) -> list[dict]:
    """Verify suppliers against web evidence in descending confidence order.

    This is the deterministic pipeline verification (Condition B).
    Suppliers are verified in a fixed order (highest confidence first)
    until the web budget is exhausted. Uses the same co-mention +
    relationship-cue matching as the agent's ``verify_supplier`` tool.

    Parameters
    ----------
    suppliers : list[dict]
        Supplier dicts from GraphState.
    company : str
        The target company name.
    budget : BudgetTracker
        Budget tracker; stops when web budget is exhausted.
    snapshot_mode : bool
        If *True*, only cached search results are used.

    Returns
    -------
    list[dict]
        Updated supplier dicts with ``evidence_source`` set to
        ``"web_verified"`` where relationship evidence was found.
    """
    from bor_risk.search import search_web, search_web_snapshot

    search_fn = search_web_snapshot if snapshot_mode else search_web

    # Sort by descending confidence (deterministic order).
    ordered = sorted(suppliers, key=lambda s: s.get("confidence", 0), reverse=True)
    updated = {s["name"]: dict(s) for s in suppliers}
    evidence_items: list[dict] = []

    company_lower = company.lower()
    web_evidence_counter = 1

    for s in ordered:
        if budget.web_budget_remaining <= 0:
            break

        supplier_name = s["name"]
        supplier_lower = supplier_name.lower()
        query = (
            f'"{company}" "{supplier_name}" '
            f"supplier OR vendor OR manufactures OR supplies"
        )
        budget.record_web_query(query=query)

        try:
            results = search_fn(query, max_results=3)
        except Exception:
            continue

        verified = False
        for r in results:
            content_lower = r["content"].lower()
            has_co_mention = (
                company_lower in content_lower
                and supplier_lower in content_lower
            )
            has_relationship = any(
                cue in content_lower for cue in _RELATIONSHIP_CUES
            )

            if has_co_mention and has_relationship:
                updated[supplier_name]["evidence_source"] = "web_verified"
                updated[supplier_name]["verification_url"] = r["url"]
                updated[supplier_name]["verification_snippet"] = r["content"][:300]
                updated[supplier_name]["confidence"] = min(
                    1.0, updated[supplier_name].get("confidence", 0.5) + 0.1
                )
                eid = f"W{web_evidence_counter}"
                web_evidence_counter += 1
                evidence_items.append({
                    "evidence_id": eid,
                    "source": f"web:{r['url']}",
                    "description": r["content"][:300],
                    "retrieved_at": r.get("retrieved_at", ""),
                })
                verified = True
                break

        if not verified:
            # Supplier remains llm_only.
            updated[supplier_name].setdefault("evidence_source", "llm_only")

    return list(updated.values()), evidence_items


# ---------------------------------------------------------------------------
# VCG: URL content fetching
# ---------------------------------------------------------------------------


def fetch_url_content(url: str) -> tuple[str, str, str, str, int]:
    """Download URL and return (plain_text, sha256_hex, mime_type, final_url, http_status).

    Uses ``_urlopen_with_ssl_fallback()``.  Strips HTML tags and extracts the
    page ``<title>``.  Limits output to 50 000 characters.

    Returns ``("", "", "", url, 0)`` on any error.
    """
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.texts: list[str] = []
            self.title: str = ""
            self._in_title = False
            self._skip = False

        def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
            if tag in ("script", "style", "noscript"):
                self._skip = True
            elif tag == "title":
                self._in_title = True

        def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
            if tag in ("script", "style", "noscript"):
                self._skip = False
            elif tag == "title":
                self._in_title = False

        def handle_data(self, data: str) -> None:  # type: ignore[override]
            stripped = data.strip()
            if not stripped:
                return
            if self._in_title:
                self.title += stripped
            elif not self._skip:
                self.texts.append(stripped)

    try:
        raw_bytes = _urlopen_with_ssl_fallback(url, timeout=15)
    except Exception:
        return "", "", "", url, 0

    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    # Detect PDFs by magic bytes first (more reliable than URL suffix).
    if raw_bytes[:4] == b"%PDF":
        mime_type = "application/pdf"
    elif url.lower().endswith(".pdf"):
        mime_type = "application/pdf"
    else:
        mime_type = "text/html"

    try:
        if mime_type == "application/pdf":
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_bytes))
                plain_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages[:20]
                )[:50_000]
            except Exception:
                # Fallback: strip non-printable chars from raw UTF-8 decode.
                raw_decoded = raw_bytes.decode("utf-8", errors="replace")
                plain_text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e]", " ", raw_decoded)[:50_000]
            title = ""
        else:
            html_text = raw_bytes.decode("utf-8", errors="replace")
            parser = _TextExtractor()
            parser.feed(html_text)
            plain_text = " ".join(parser.texts)[:50_000]
            title = parser.title.strip()
    except Exception:
        plain_text = ""
        title = ""

    return plain_text, content_hash, mime_type, url, 200


# ---------------------------------------------------------------------------
# VCG: LLM-based mention extraction
# ---------------------------------------------------------------------------


def extract_mentions_from_doc_llm(
    doc_text: str,
    company: str,
    evidence_id: str,
) -> list[dict]:
    """Extract supplier mentions from *doc_text* using a structured LLM call.

    Returns a list of dicts with keys:
    ``{object_entity_id, supporting_quote, evidence_id, start_char, end_char}``

    Post-verifies each span: if ``start_char:end_char`` does not match
    ``supporting_quote`` in ``doc_text``, the offsets are cleared to ``None``.

    Returns ``[]`` on any LLM failure.
    """
    from bor_risk.models import ClaimExtractionResponse

    if not doc_text.strip():
        return []

    prompts = load_prompts()
    template = prompts.get("claim_extraction_prompt", "")
    if not template:
        return []

    # Truncate document to stay within context limits.
    truncated = doc_text[:8_000]
    prompt_text = template.format(
        company=company,
        evidence_id=evidence_id,
        doc_text=truncated,
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)
        # Use function_calling to avoid strict JSON-schema issues with list[dict] fields.
        structured_llm = llm.with_structured_output(
            ClaimExtractionResponse, method="function_calling"
        )
        response: ClaimExtractionResponse = structured_llm.invoke(
            [HumanMessage(content=prompt_text)]
        )
        mentions = response.mentions_found
    except Exception as exc:
        import sys
        print(f"  [WARNING] extract_mentions LLM call failed: {exc}", file=sys.stderr, flush=True)
        return []

    # Post-verify char offsets: check quote is a substring at the claimed offset.
    verified: list[dict] = []
    for m in mentions:
        quote = m.get("supporting_quote", "")
        start = m.get("start_char")
        end = m.get("end_char")

        if quote and start is not None and end is not None:
            # Verify the span matches the quote
            span_text = doc_text[start:end] if 0 <= start < end <= len(doc_text) else ""
            if span_text != quote:
                # Offsets are wrong; clear them but keep quote if it is in text
                m = dict(m)
                m["start_char"] = None
                m["end_char"] = None
                if quote not in doc_text:
                    m["supporting_quote"] = ""

        m = dict(m)
        m["evidence_id"] = evidence_id  # ensure correct evidence_id
        verified.append(m)

    return verified


# ---------------------------------------------------------------------------
# VCG: Mention-to-claim linking
# ---------------------------------------------------------------------------


def _normalize_for_link(name: str) -> str:
    """Lowercase + strip punctuation for linking comparison."""
    import re
    return re.sub(r"[.,\-'\"()]", "", name.strip().lower())


def link_mentions_to_claims(
    mentions: list[dict],
    claims: list["Claim"],  # noqa: F821
) -> list["Claim"]:  # noqa: F821
    """Attach evidence spans to PROPOSED claims by fuzzy name matching.

    Matching priority:
    1. Exact ``normalized_name`` match
    2. Any alias match
    3. Fuzzy match via ``rapidfuzz`` (token_set_ratio >= 85) if available,
       otherwise substring match fallback

    Creates a new PROPOSED Claim only when no existing claim matches AND
    the mention's evidence word_count context is non-trivial (quote length > 20).

    Returns the updated claims list.
    """
    from bor_risk.models import Claim, EvidenceSpan, LocationCandidate

    try:
        from rapidfuzz.fuzz import token_set_ratio as _tsr
        def _fuzzy_match(a: str, b: str) -> bool:
            return _tsr(a, b) >= 85
    except ImportError:
        def _fuzzy_match(a: str, b: str) -> bool:  # type: ignore[misc]
            return a in b or b in a

    updated = [c.model_copy(deep=True) for c in claims]
    claim_index = {i: updated[i].normalized_name for i in range(len(updated))}

    for mention in mentions:
        obj = mention.get("object_entity_id", "")
        if not obj:
            continue
        quote = mention.get("supporting_quote", "")
        eid = mention.get("evidence_id", "")
        start_char = mention.get("start_char")
        end_char = mention.get("end_char")
        norm_obj = _normalize_for_link(obj)

        # Find matching claim
        best_idx: int | None = None
        for i, claim in enumerate(updated):
            if claim.normalized_name == norm_obj:
                best_idx = i
                break
            if any(_normalize_for_link(a) == norm_obj for a in claim.aliases):
                best_idx = i
                break
            if _fuzzy_match(claim.normalized_name, norm_obj):
                best_idx = i
                # Don't break — prefer exact match

        if best_idx is not None:
            c = updated[best_idx]
            if eid and eid not in c.evidence_refs:
                c.evidence_refs.append(eid)
            if quote and eid:
                span = EvidenceSpan(
                    evidence_id=eid,
                    start_char=start_char,
                    end_char=end_char,
                    quote=quote,
                )
                c.supporting_spans.append(span)
            if c.status == "PROPOSED":
                c = c.model_copy(update={"status": "RETRIEVED"})
                updated[best_idx] = c
        else:
            # Create new PROPOSED claim only for substantial quotes
            if quote and len(quote) > 20:
                from bor_risk.models import Claim as _Claim
                import hashlib as _hs
                new_norm = _normalize_for_link(obj)
                cid = _hs.sha256(
                    f"{mention.get('company', '')}|{obj}|SUPPLIES_TO|1".encode()
                ).hexdigest()[:12]
                display_id = f"C{len(updated) + 1:03d}"
                span = EvidenceSpan(
                    evidence_id=eid,
                    start_char=start_char,
                    end_char=end_char,
                    quote=quote,
                )
                new_claim = _Claim(
                    claim_id=cid,
                    display_id=display_id,
                    subject_entity_id=mention.get("company", ""),
                    object_entity_id=obj,
                    normalized_name=new_norm,
                    confidence=0.3,  # low initial confidence for evidence-only claims
                    evidence_refs=[eid] if eid else [],
                    supporting_spans=[span],
                    status="RETRIEVED",
                )
                updated.append(new_claim)

    return updated


# ---------------------------------------------------------------------------
# VCG: Two-stage claim verification
# ---------------------------------------------------------------------------


def _name_variants(name: str, aliases: list[str] | None = None) -> set[str]:
    """Return a set of lowercased name variants for fuzzy entity matching.

    Generates: base name, name without parenthetical suffix, and an acronym
    from capitalized words (only when ≥3 words to avoid short false positives
    like 'LG'). Curated ``aliases`` from ``claim.aliases`` are always included.
    """
    name = name.strip()
    variants: set[str] = {name.lower()}
    # Strip parenthetical qualifiers like "Co. (Taiwan)"
    paren_stripped = re.sub(r"\s*\(.*?\)\s*$", "", name).strip().lower()
    if paren_stripped:
        variants.add(paren_stripped)
    # Acronym from ≥3 capitalized words only (avoids "LG", "3M" false positives)
    cap_words = [w for w in name.split() if w and w[0].isupper()]
    if len(cap_words) >= 3:
        variants.add("".join(w[0] for w in cap_words).lower())
    for alias in aliases or []:
        alias = alias.strip().lower()
        if alias:
            variants.add(alias)
    return variants


def _any_variant_in_text(variants: set[str], text_lower: str) -> bool:
    """Return True if any variant appears in *text_lower*.

    Short tokens (≤4 chars) use word-boundary matching to prevent substring
    false positives (e.g. 'lg' matching 'algae').
    """
    for v in variants:
        if not v:
            continue
        if len(v) <= 4:
            if re.search(r"\b" + re.escape(v) + r"\b", text_lower):
                return True
        elif v in text_lower:
            return True
    return False


def verify_claim_against_evidence(
    claim: "Claim",  # noqa: F821
    evidence_text: str,
    budget: "BudgetTracker | None" = None,  # noqa: F821
) -> "VerificationVerdict":  # noqa: F821
    """Verify a Claim against evidence text using two-stage verification.

    Stage 1 — heuristic pre-filter (no LLM cost):
      - co-mention check: both company and supplier names present
      - relationship-cue check: ``_RELATIONSHIP_CUES`` near both names
      - negation check: termination/dispute language → DISPUTED

    Stage 2 — LLM entailment verdict (structured output):
      - Uses ``claim_verification_prompt`` from ``prompts.yaml``
      - Returns :class:`VerificationVerdict` with verdict, rationale, quote, direction

    Stage 3 — Substring guard (always runs after Stage 2):
      - If ``supporting_quote`` is not an exact substring of ``evidence_text``,
        downgrades SUPPORTED → WEAK and clears the quote.

    Returns :class:`VerificationVerdict`.
    """
    from bor_risk.models import VerificationVerdict

    company = claim.subject_entity_id
    supplier = claim.object_entity_id
    text_lower = evidence_text.lower()
    company_lower = company.lower()
    supplier_lower = supplier.lower()

    # Build alias-aware variant sets for both entities.
    company_variants = _name_variants(company)
    supplier_variants = _name_variants(supplier, aliases=getattr(claim, "aliases", None))

    # --- Stage 1: heuristic pre-filter ---

    has_co_mention = (
        _any_variant_in_text(company_variants, text_lower)
        and _any_variant_in_text(supplier_variants, text_lower)
    )
    if not has_co_mention:
        return VerificationVerdict(
            verdict="UNKNOWN",
            rationale="Co-mention check failed: neither company nor supplier found in evidence.",
            supporting_quote="",
            direction="unclear",
        )

    _NEGATION_CUES = frozenset({
        "no longer", "terminated", "dispute", "divest", "ended supply",
        "removed from", "cut ties", "dissolved", "bankruptcy", "ceased",
        "dropped", "parted ways",
    })
    has_negation = any(cue in text_lower for cue in _NEGATION_CUES)
    if has_negation:
        return VerificationVerdict(
            verdict="DISPUTED",
            rationale="Evidence contains termination or dispute language.",
            supporting_quote="",
            direction="unclear",
        )

    has_relationship = any(cue in text_lower for cue in _RELATIONSHIP_CUES)

    # --- Stage 2: LLM entailment ---
    prompts = load_prompts()
    template = prompts.get("claim_verification_prompt", "")

    if not template:
        # No prompt configured — fall back to heuristic
        verdict = "SUPPORTED" if has_relationship else "WEAK"
        return VerificationVerdict(
            verdict=verdict,
            rationale="Heuristic verification only (no LLM prompt configured).",
            supporting_quote="",
            direction="unclear",
        )

    if budget is not None and budget.llm_budget_remaining <= 0:
        verdict = "SUPPORTED" if has_relationship else "WEAK"
        return VerificationVerdict(
            verdict=verdict,
            rationale="LLM budget exhausted; heuristic fallback used.",
            supporting_quote="",
            direction="unclear",
        )

    # Build a focused snippet centred on the supplier mention.
    idx = text_lower.find(supplier_lower)
    if idx >= 0:
        start = max(0, idx - 200)
        end = min(len(evidence_text), idx + 400)
        snippet = evidence_text[start:end]
    else:
        snippet = evidence_text[:600]

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)
        structured_llm = llm.with_structured_output(VerificationVerdict)
        prompt_text = template.format(
            company=company,
            supplier=supplier,
            evidence_snippet=snippet[:2_000],
        )
        result: VerificationVerdict = structured_llm.invoke(
            [HumanMessage(content=prompt_text)]
        )
        if budget is not None:
            budget.record_llm_call(purpose=f"verify_claim_{claim.claim_id}")
    except Exception:
        verdict = "SUPPORTED" if has_relationship else "WEAK"
        return VerificationVerdict(
            verdict=verdict,
            rationale="LLM verification failed; heuristic fallback used.",
            supporting_quote="",
            direction="unclear",
        )

    # --- Stage 3: substring guard ---
    quote = result.supporting_quote
    if quote and quote not in evidence_text:
        # Quote is not verbatim in evidence — downgrade
        if result.verdict == "SUPPORTED":
            result = VerificationVerdict(
                verdict="WEAK",
                rationale=result.rationale + " [Quote not verified as exact substring]",
                supporting_quote="",
                direction=result.direction,
            )
        else:
            result = VerificationVerdict(
                verdict=result.verdict,
                rationale=result.rationale,
                supporting_quote="",
                direction=result.direction,
            )

    return result


# ---------------------------------------------------------------------------
# VCG: Location resolution
# ---------------------------------------------------------------------------

# Simple lookup table for common location phrases → lat/lon
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "shenzhen": (22.543, 114.057),
    "taipei": (25.033, 121.565),
    "shanghai": (31.228, 121.474),
    "beijing": (39.905, 116.391),
    "tokyo": (35.689, 139.692),
    "osaka": (34.693, 135.502),
    "seoul": (37.566, 126.978),
    "munich": (48.137, 11.576),
    "berlin": (52.520, 13.405),
    "frankfurt": (50.110, 8.682),
    "london": (51.508, -0.128),
    "paris": (48.857, 2.347),
    "amsterdam": (52.370, 4.895),
    "san jose": (37.338, -121.886),
    "san francisco": (37.774, -122.419),
    "austin": (30.267, -97.743),
    "new york": (40.713, -74.006),
    "singapore": (1.352, 103.820),
    "hong kong": (22.319, 114.170),
    "bangalore": (12.972, 77.594),
    "chennai": (13.082, 80.270),
    "pune": (18.520, 73.856),
    "hsinchu": (24.813, 120.967),
    "eindhoven": (51.441, 5.479),
    "stockholm": (59.333, 18.065),
}


def _geocode_simple(place_name: str) -> tuple[float, float] | None:
    """Return (lat, lon) for well-known city names using a lookup table."""
    key = place_name.lower().strip()
    for city, coords in _CITY_COORDS.items():
        if city in key:
            return coords
    return None


def resolve_locations_for_claims(
    claims: list["Claim"],  # noqa: F821
    evidence_texts: dict[str, str],
    company: str = "",
    budget: "BudgetTracker | None" = None,  # noqa: F821
) -> list["Claim"]:
    """Enrich ``location_candidates`` for each claim from evidence snippets.

    For each claim, scans associated evidence texts for city/country mentions
    using a simple geocode lookup table.  When a location is found with higher
    confidence than the existing LLM seed, it is prepended to
    ``location_candidates``.

    Falls back to the existing LLM seed (source="llm") if no evidence-based
    location is found.
    """
    from bor_risk.models import LocationCandidate

    updated = [c.model_copy(deep=True) for c in claims]

    for i, claim in enumerate(updated):
        found_loc: LocationCandidate | None = None

        for eid in claim.evidence_refs:
            ev_text = evidence_texts.get(eid, "")
            if not ev_text:
                continue

            # Look for city names in the evidence text
            ev_lower = ev_text.lower()
            for city, coords in _CITY_COORDS.items():
                if city in ev_lower:
                    found_loc = LocationCandidate(
                        lat=coords[0],
                        lon=coords[1],
                        confidence=0.75,
                        source="evidence",
                        place_name=city.title(),
                    )
                    break
            if found_loc:
                break

        if found_loc:
            # Prepend evidence-based candidate (higher confidence than LLM seed)
            new_candidates = [found_loc] + [
                c for c in claim.location_candidates if c.source != "evidence"
            ]
            updated[i] = claim.model_copy(update={"location_candidates": new_candidates})

    return updated
