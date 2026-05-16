
import argparse
import csv
import math
import rasterio

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--value-column", required=True)
parser.add_argument("--skip-zero", action="store_true")
args = parser.parse_args()

with rasterio.open(args.src) as ds, open(args.out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["lon", "lat", args.value_column])
    writer.writeheader()

    nodata = ds.nodata
    for _, window in ds.block_windows(1):
        arr = ds.read(1, window=window, masked=True)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if getattr(val, "mask", False):
                    continue
                val = float(val)
                if not math.isfinite(val):
                    continue
                if nodata is not None and val == nodata:
                    continue
                if args.skip_zero and val <= 0:
                    continue

                row = window.row_off + r
                col = window.col_off + c
                lon, lat = ds.xy(row, col)
                writer.writerow({"lon": lon, "lat": lat, args.value_column: val})

