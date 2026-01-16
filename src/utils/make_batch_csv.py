import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    inp = os.path.expanduser(args.input_dir)
    out_csv = os.path.expanduser(args.out_csv)

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(inp, "**", f"*{e}"), recursive=True)

    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No images found in {inp}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path"])
        for p in files:
            rel = os.path.relpath(p, inp).replace("\\", "/")
            w.writerow([f"/input/{rel}"])

    print(f"Wrote {len(files)} rows to {out_csv}")


if __name__ == "__main__":
    main()
