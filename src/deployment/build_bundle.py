import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from typing import Any, Optional
import mlflow
import torch
import mlflow.pytorch

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return "unknown"

def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    files = [p for p in path.rglob("*") if p.is_file()]
    for p in sorted(files, key=lambda x: str(x.relative_to(path))):
        rel = str(p.relative_to(path)).encode()
        h.update(rel)
        h.update(_sha256_file(p).encode())
    return h.hexdigest()

def _mlflow_list_artifacts(run_id: str, path: str = "") -> List[str]:
    client = mlflow.tracking.MlflowClient()
    items = client.list_artifacts(run_id, path)
    return [it.path for it in items]

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def _find_first(meta: Dict[str, Any], keys) -> Optional[Any]:
    for k in keys:
        if k in meta:
            return meta[k]
    for v in meta.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return v[k]
    return None

def _try_download_mlflow_model(run_id: str, artifact_path: str, dst_dir: Path) -> bool:
    """
    Try downloading MLflow artifact directory runs:/run_id/artifact_path into dst_dir.
    Returns True on success, False if not found / fails.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        shutil.copytree(local_path, dst_dir, dirs_exist_ok=True)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow_run_id", required=True)
    ap.add_argument("--run_dir", required=True, help="BEST_RUN_DIR from best_model_meta")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--class_names", default="Normal,Pneumonia,Tuberculosis")
    ap.add_argument("--cnn_cfg", default=None, help="Optional: path to config to snapshot")
    ap.add_argument("--model_artifact_path", default=None,help="MLflow artifact path where the model is logged (default: auto-detect, prefers 'model')")

    ap.add_argument( "--extra_files",default="", help="Comma-separated relative paths under run_dir to copy into bundle if they exist "
             "(e.g. 'calibration.json,metrics.json')" )
    #semver tag for deployment bundles
    ap.add_argument("--semver", default="0.0.0", help="Deployment semantic version (e.g., 1.2.0)")
    ap.add_argument("--best_meta", default=None, help="Path to best_model_meta.json for fallback model resolution")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    run_dir = Path(args.run_dir)
    _safe_mkdir(out_dir)

    policy_src = run_dir / "thresholds_policy.json"
    if not policy_src.exists():
        raise FileNotFoundError(f"Missing policy file: {policy_src}")
    policy_dst = out_dir / "thresholds_policy.json"
    shutil.copy2(policy_src, policy_dst)
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    if not class_names:
        raise ValueError("class_names parsed empty; check --class_names")
    _write_json(out_dir / "class_names.json", {"class_names": class_names})

    if args.cnn_cfg and Path(args.cnn_cfg).exists():
        shutil.copy2(args.cnn_cfg, out_dir / "preprocessing_source_config.yaml")
    if args.extra_files.strip():
        for rel in [x.strip() for x in args.extra_files.split(",") if x.strip()]:
            src = run_dir / rel
            if src.exists() and src.is_file():
                shutil.copy2(src, out_dir / Path(rel).name)
    model_out = out_dir / "model"
    if model_out.exists():
        shutil.rmtree(model_out)
    _safe_mkdir(model_out)

    ok = _try_download_mlflow_model(args.mlflow_run_id, "model", model_out)

    if not ok and args.best_meta:
        meta_path = Path(args.best_meta)
        if meta_path.exists():
            meta = _read_json(meta_path)
            model_uri = _find_first(meta, ["model_uri", "modelURI", "modelUri"])
            if isinstance(model_uri, str) and model_uri.startswith("runs:/"):
                try:
                    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
                    shutil.copytree(local_path, model_out, dirs_exist_ok=True)
                    ok = True
                except Exception:
                    ok = False

            #run_id
            if not ok:
                train_run_id = _find_first(meta, ["train_run_id", "training_run_id", "train_mlflow_run_id"])
                if isinstance(train_run_id, str) and len(train_run_id) >= 8:
                    for apath in ["model", "models", "pytorch_model", "artifacts/model", "data/model"]:
                        if _try_download_mlflow_model(train_run_id, apath, model_out):
                            ok = True
                            break

            if not ok:
                ckpt = _find_first(meta, ["model_path", "checkpoint_path", "ckpt_path", "best_ckpt_path"])
                if isinstance(ckpt, str) and ckpt:
                    ckpt_path = Path(ckpt)
                    if not ckpt_path.exists():
                        rel1 = run_dir / ckpt
                        rel2 = Path.cwd() / ckpt
                        if rel1.exists():
                            ckpt_path = rel1
                        elif rel2.exists():
                            ckpt_path = rel2
                    if ckpt_path.exists() and ckpt_path.is_file():
                        shutil.copy2(ckpt_path, model_out / ckpt_path.name)
                        ok = True

    if not ok:
        try:
            from src.utils.mlflow_utils import load_run_model_pytorch
            resolved_id = args.mlflow_run_id
            if args.best_meta and Path(args.best_meta).exists():
                meta = _read_json(Path(args.best_meta))
                cand = _find_first(meta, ["train_run_id", "training_run_id", "train_mlflow_run_id"])
                if isinstance(cand, str) and len(cand) >= 8:
                    resolved_id = cand
            model = load_run_model_pytorch(resolved_id)
            model.eval()
            mlflow.pytorch.save_model(model, path=str(model_out))
            ok = True
            print(f"[Fallback] Exported model via load_run_model_pytorch(run_id={resolved_id})")
        except Exception as e:
            ok = False
            print(f"[Fallback] load_run_model_pytorch failed: {e}")

    if not ok:
        raise FileNotFoundError(
            "Failed to bundle model. Could not find MLflow model artifact directory "
            "and could not resolve a model location from best_meta.\n"
            "Fix today: ensure best_model_meta.json includes one of: model_uri, train_run_id, model_path/checkpoint_path.\n"
            "Fix long-term: log the model as an MLflow artifact under 'model/'." )


    # --- Compute hashes for traceability ---
    policy_hash = _sha256_file(policy_dst)
    model_hash = _sha256_dir(model_out)

    manifest = {
        "bundle_format_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "semver": args.semver, "mlflow_run_id": args.mlflow_run_id,
        "mlflow_model_uri": model_uri, "source_run_dir": str(run_dir),
        "git_commit": _git_sha(),
        "class_names": class_names,
        "hashes": {
            "thresholds_policy_sha256": policy_hash,
            "model_dir_sha256": model_hash},
        "artifacts": {
            "model": "model/",
            "policy": "thresholds_policy.json", "class_names": "class_names.json",
            "preprocessing_source_config": "preprocessing_source_config.yaml"
            if (out_dir / "preprocessing_source_config.yaml").exists()
            else None,
        },
    }
    manifest["artifacts"] = {k: v for k, v in manifest["artifacts"].items() if v is not None}
    #Todo: lily hardcode here  => future improvement
    manifest["model_family"] = "cnn"
    manifest["gradcam_target_layer"] = "features.8"
    manifest["cam_method"] = "gradcam-pp-smooth"
    _write_json(out_dir / "manifest.json", manifest)
    print(f"[OK] Bundle written to: {out_dir}")

if __name__ == "__main__":
    main()
