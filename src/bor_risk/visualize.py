"""Optional visualization: Folium interactive map + NetworkX static graph.

Install dependencies:
    pip install -e ".[viz]"
    # or: pip install folium networkx matplotlib
"""

from __future__ import annotations

from pathlib import Path

# Risk colour scale aligned with HIGH_RISK_THRESHOLD=0.4 / CRITICAL_EXCEEDANCE_MARGIN=0.2
_RISK_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.25, "#2ecc71", "Low (<0.25)"),
    (0.45, "#f1c40f", "Medium-Low (0.25–0.45)"),
    (0.60, "#e67e22", "Medium-High (0.45–0.60)"),
    (1.01, "#e74c3c", "High (≥0.60)"),
]


def _risk_color(score: float) -> str:
    for threshold, color, _ in _RISK_THRESHOLDS:
        if score < threshold:
            return color
    return _RISK_THRESHOLDS[-1][1]


def _risk_label(score: float) -> str:
    for threshold, _, label in _RISK_THRESHOLDS:
        if score < threshold:
            return label
    return _RISK_THRESHOLDS[-1][2]


def _build_risk_lookup(state: dict) -> dict[str, float]:
    """Return supplier_name → max hazard score across all hazard types."""
    lookup: dict[str, float] = {}
    for hs in state.get("hazard_scores", []):
        name = hs["supplier_name"]
        score = float(hs["score"])
        if score > lookup.get(name, 0.0):
            lookup[name] = score
    return lookup


def _is_valid_coord(lat: float, lon: float) -> bool:
    """Reject (0, 0) LLM-hallucination sentinel and out-of-range values."""
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


# ---------------------------------------------------------------------------
# Folium interactive HTML map
# ---------------------------------------------------------------------------

_LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 30px; right: 30px;
    z-index: 1000;
    background: white;
    padding: 12px 16px;
    border: 2px solid #ccc;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
">
<b>Risk Score</b><br>
{rows}
</div>
"""

_LEGEND_ROW = (
    '<span style="display:inline-block;width:14px;height:14px;'
    'background:{color};border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
    "{label}<br>"
)


def render_supplier_map(state: dict, company: str, out_path: Path) -> None:
    """Write an interactive Folium HTML map to *out_path*.

    Suppliers are shown as circle markers coloured by max hazard score.
    Supply-chain edges are drawn as polylines.
    A legend and focal-company label are included.
    """
    try:
        import folium
        from branca.element import MacroElement
        from jinja2 import Template
    except ImportError:
        print(
            "  [map] folium not installed — skipping HTML map.\n"
            "  Install with: pip install -e '.[viz]'"
        )
        return

    suppliers: list[dict] = state.get("suppliers", [])
    edges: list[dict] = state.get("edges", [])
    risk_lookup = _build_risk_lookup(state)
    hazard_alerts: list[dict] = (
        state.get("company_risk_summary", {}).get("hazard_alerts", [])
    )

    # Build alert index: supplier_name → list of alert strings
    alert_index: dict[str, list[str]] = {}
    for a in hazard_alerts:
        name = a["supplier_name"]
        alert_index.setdefault(name, []).append(
            f"{a['hazard_type']} {a['score']:.2f} (threshold {a['threshold']:.2f})"
        )

    # Build supplier lookup by name for edge drawing
    supplier_coords: dict[str, tuple[float, float]] = {}
    for s in suppliers:
        lat, lon = float(s["lat"]), float(s["lon"])
        if _is_valid_coord(lat, lon):
            supplier_coords[s["name"]] = (lat, lon)

    # Map centre: mean of valid coords, fallback to (20, 0)
    valid_coords = list(supplier_coords.values())
    if valid_coords:
        center_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        center_lon = sum(c[1] for c in valid_coords) / len(valid_coords)
    else:
        center_lat, center_lon = 20.0, 0.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    # Draw supplier→supplier edges (skip edges involving the focal company to avoid (0,0))
    company_lower = company.lower()
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        if parent.lower() == company_lower or child.lower() == company_lower:
            continue
        if parent in supplier_coords and child in supplier_coords:
            p_lat, p_lon = supplier_coords[parent]
            c_lat, c_lon = supplier_coords[child]
            folium.PolyLine(
                [(p_lat, p_lon), (c_lat, c_lon)],
                color="#888888",
                weight=1.5,
                opacity=0.6,
            ).add_to(m)

    # Draw supplier markers
    for s in suppliers:
        lat, lon = float(s["lat"]), float(s["lon"])
        if not _is_valid_coord(lat, lon):
            continue

        name = s["name"]
        tier = int(s.get("tier", 1))
        score = risk_lookup.get(name, 0.0)
        color = _risk_color(score)
        radius = 10 if tier == 1 else 7

        alerts = alert_index.get(name, [])
        alert_html = (
            "<b>Alerts:</b><br>" + "<br>".join(f"&nbsp;• {a}" for a in alerts)
            if alerts
            else "<i>No threshold alerts</i>"
        )
        popup_html = (
            f"<b>{name}</b><br>"
            f"Tier: {tier}<br>"
            f"Risk score: {score:.3f} ({_risk_label(score)})<br>"
            f"Confidence: {s.get('confidence', 0):.0%}<br>"
            f"Industry: {s.get('industry', '—')}<br>"
            f"Product: {s.get('product_category', '—')}<br>"
            f"Location: {s.get('location_description', '—')}<br>"
            f"Evidence: {s.get('evidence_source', '—')}<br>"
            + (
                f"<a href='{s['verification_url']}' target='_blank'>Source</a><br>"
                if s.get("verification_url")
                else ""
            )
            + alert_html
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{name} (T{tier}): {score:.2f}",
        ).add_to(m)

    # Focal company label at map centre
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:13px;font-weight:bold;'
                f'color:#1a1a2e;white-space:nowrap;'
                f'text-shadow:1px 1px 2px white,-1px -1px 2px white;">'
                f'{company}</div>'
            ),
            icon_size=(150, 20),
            icon_anchor=(75, 10),
        ),
    ).add_to(m)

    # Legend
    legend_rows = "".join(
        _LEGEND_ROW.format(color=color, label=label)
        for _, color, label in _RISK_THRESHOLDS
    )

    class LegendControl(MacroElement):
        def __init__(self, html: str) -> None:
            super().__init__()
            self._template = Template(html)

    legend = LegendControl(_LEGEND_HTML.format(rows=legend_rows))
    m.get_root().html.add_child(legend)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# NetworkX + matplotlib static PNG
# ---------------------------------------------------------------------------


def render_network_graph(state: dict, company: str, out_path: Path) -> None:
    """Write a static hierarchical supply-chain PNG to *out_path*.

    Nodes are coloured by max hazard score.  Tiers are laid out on separate
    horizontal bands.  Suitable for thesis inclusion.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print(
            "  [map] networkx/matplotlib not installed — skipping network PNG.\n"
            "  Install with: pip install -e '.[viz]'"
        )
        return

    suppliers: list[dict] = state.get("suppliers", [])
    edges: list[dict] = state.get("edges", [])
    risk_lookup = _build_risk_lookup(state)

    if not suppliers:
        print("  [map] No suppliers found — skipping network PNG.")
        return

    # Build graph
    G = nx.DiGraph()
    # Focal company is tier 0
    G.add_node(company, tier=0)

    for s in suppliers:
        G.add_node(s["name"], tier=s.get("tier", 1))

    # Add edges
    company_lower = company.lower()
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        # Normalise focal company references
        p = company if parent.lower() == company_lower else parent
        c = company if child.lower() == company_lower else child
        if G.has_node(p) and G.has_node(c):
            G.add_edge(p, c)

    # Hierarchical layout
    tier_nodes: dict[int, list[str]] = {}
    for node, data in G.nodes(data=True):
        t = data.get("tier", 1)
        tier_nodes.setdefault(t, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    tier_y = {0: 1.0, 1: 0.5, 2: 0.0}
    for tier, nodes in tier_nodes.items():
        y = tier_y.get(tier, -0.5 * tier)
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2.0) * (1.0 / max(n, 1)) * n
            pos[node] = (x, y)

    # Node styling
    node_colors: list[str] = []
    node_sizes: list[int] = []
    labels: dict[str, str] = {}
    for node in G.nodes():
        if node == company:
            node_colors.append("white")
            node_sizes.append(2400)
            labels[node] = node
        else:
            score = risk_lookup.get(node, 0.0)
            node_colors.append(_risk_color(score))
            tier = G.nodes[node].get("tier", 1)
            node_sizes.append(1200 if tier == 1 else 600)
            short = node[:18] + ("…" if len(node) > 18 else "")
            labels[node] = f"{short}\n{score:.2f}"

    n_tier2 = len(tier_nodes.get(2, []))
    fig_width = max(16, n_tier2 * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.set_axis_off()

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        labels=labels,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=7,
        font_color="#1a1a1a",
        arrows=True,
        arrowsize=12,
        edge_color="#aaaaaa",
        width=1.2,
        with_labels=True,
    )

    # Tier labels on left margin
    for tier, y in tier_y.items():
        label = {0: company, 1: "Tier 1", 2: "Tier 2"}.get(tier)
        if label and tier_nodes.get(tier):
            ax.text(
                ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -2,
                y,
                label if tier > 0 else "",
                fontsize=9,
                color="#555555",
                va="center",
                style="italic",
            )

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, color, label in _RISK_THRESHOLDS
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        title="Risk Score",
        framealpha=0.9,
    )

    ax.set_title(f"Supply Chain Network — {company}", fontsize=13, pad=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
