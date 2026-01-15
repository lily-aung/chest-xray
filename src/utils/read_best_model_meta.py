# src/utils/read_best_model_meta.py
from __future__ import annotations
import json
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--field", type=str, required=True,
                   help="mlflow_run_id")
    args = p.parse_args()
    meta_path = Path(args.meta)
    obj = json.loads(meta_path.read_text())
    best = obj["best_model"]
    if args.field not in best:
        raise SystemExit(f"Field '{args.field}' not found in best_model_meta.json. Available: {list(best.keys())}")

    print(best[args.field])

if __name__ == "__main__":
    main()
