import os
import argparse
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_gray01(path: str, img_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def hflip(img01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img01[:, ::-1])

def apply_clahe(img01: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray:
    img_u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    out = clahe.apply(img_u8)
    return out.astype(np.float32) / 255.0

def hog_feat(img01: np.ndarray, orientations=9, ppc=16, cpb=2) -> np.ndarray:
    return hog( img01, orientations=orientations,
        pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), block_norm="L2-Hys",
        transform_sqrt=True, feature_vector=True ).astype(np.float32)

def build_split(csv_path: str, img_root: str | None, img_col: str,label_col: str,img_size: int,
                clip: float,grid: tuple[int, int], orientations: int,ppc: int,cpb: int,
                augment_train: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(">>> Build Features for HOG " , str)
    df = pd.read_csv(csv_path)
    X, y, paths = [], [], []

    for _, row in df.iterrows():
        rel = str(row[img_col])
        lab = int(row[label_col])
        full = rel if img_root is None else os.path.join(img_root, rel)

        img = load_gray01(full, img_size)
        img_c = apply_clahe(img, clip, grid)
        X.append(hog_feat(img_c, orientations, ppc, cpb))
        y.append(lab)
        paths.append(full)

        #CLAHE(flip(img)) for training
        if augment_train:
            img_cf = apply_clahe(hflip(img), clip, grid)
            X.append(hog_feat(img_cf, orientations, ppc, cpb))
            y.append(lab)
            paths.append(full + "::flip")
    return np.stack(X), np.array(y, dtype=np.int64), paths

def save_cm_plots(cm: np.ndarray, class_names: list[str], out_dir: str, prefix: str):
    ensure_dir(out_dir)
    # counts
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{prefix} confusion (counts)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_cm_counts.png"), dpi=200)
    plt.close(fig)
    # normalized
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format=".2f")
    ax.set_title(f"{prefix} confusion (row-normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_cm_norm.png"), dpi=200)
    plt.close(fig)

    # misclassification matrix (diag=0)
    mis = cm.copy()
    np.fill_diagonal(mis, 0)
    disp = ConfusionMatrixDisplay(mis, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{prefix} misclassification (diag=0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_misclass.png"), dpi=200)
    plt.close(fig)

    # top confusions csv
    rows = []
    for i in range(mis.shape[0]):
        for j in range(mis.shape[1]):
            if i != j and mis[i, j] > 0:
                rows.append((class_names[i], class_names[j], int(mis[i, j])))
    rows.sort(key=lambda x: x[2], reverse=True)
    pd.DataFrame(rows, columns=["true", "pred", "count"]).to_csv(
        os.path.join(out_dir, f"{prefix}_top_misclass.csv"), index=False )


def save_preds_csv(paths, y_true, y_pred, proba, class_names, out_path):
    df = pd.DataFrame({ "path": paths,
        "y_true": y_true, "y_pred": y_pred,
        "true_name": [class_names[i] for i in y_true],
        "pred_name": [class_names[i] for i in y_pred],
        "correct": (y_true == y_pred).astype(int)})
    if proba is not None:
        for k, name in enumerate(class_names):
            df[f"proba_{name}"] = proba[:, k]
        df["pred_confidence"] = proba.max(axis=1)
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--img_root", default="null")
    ap.add_argument("--img_col", default="path")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--class_names", default="Normal,Pneumonia,Tuberculosis")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=str, default="8,8")  # "8,8"
    ap.add_argument("--hog_orientations", type=int, default=9)
    ap.add_argument("--hog_ppc", type=int, default=16)
    ap.add_argument("--hog_cpb", type=int, default=2)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="artifacts/hog_baselines")
    args = ap.parse_args()

    np.random.seed(args.seed)

    img_root = None if str(args.img_root).lower() == "null" else args.img_root
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    grid = tuple(int(x) for x in args.clahe_grid.split(","))

    ensure_dir(args.out_dir)

    Xtr, ytr, ptr = build_split(args.train_csv, img_root, args.img_col, args.label_col,
                                args.img_size, args.clahe_clip, grid,
                                args.hog_orientations, args.hog_ppc, args.hog_cpb,
                                augment_train=True)

    Xva, yva, pva = build_split(args.val_csv, img_root, args.img_col, args.label_col,
                                args.img_size, args.clahe_clip, grid,
                                args.hog_orientations, args.hog_ppc, args.hog_cpb,
                                augment_train=False)

    Xte, yte, pte = build_split(args.test_csv, img_root, args.img_col, args.label_col,
                                args.img_size, args.clahe_clip, grid,
                                args.hog_orientations, args.hog_ppc, args.hog_cpb,
                                augment_train=False)

    num_classes = int(max(ytr.max(), yva.max(), yte.max())) + 1
    if len(class_names) != num_classes:
        class_names = class_names[:num_classes] + [f"class_{i}" for i in range(len(class_names), num_classes)]

    models = {}

    models["mlp"] = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier( hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3,  max_iter=80,
            early_stopping=True, n_iter_no_change=10, random_state=args.seed ))   ])

    models["rf"] = RandomForestClassifier( n_estimators=500,
        n_jobs=-1, random_state=args.seed, class_weight="balanced_subsample" )

    if HAS_XGB:
        models["xgb"] = XGBClassifier( n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9,reg_lambda=1.0, objective="multi:softprob",
            num_class=num_classes, eval_metric="mlogloss", random_state=args.seed, n_jobs=-1  )

    summary = []
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        # val
        yv = clf.predict(Xva)
        val_acc = accuracy_score(yva, yv)
        val_f1 = f1_score(yva, yv, average="macro")
        # test
        yt = clf.predict(Xte)
        test_acc = accuracy_score(yte, yt)
        test_f1 = f1_score(yte, yt, average="macro")
        proba = clf.predict_proba(Xte) if hasattr(clf, "predict_proba") else None

        cm = confusion_matrix(yte, yt, labels=list(range(num_classes)))
        save_cm_plots(cm, class_names, args.out_dir, prefix=f"{name}_test")
        save_preds_csv(pte, yte, yt, proba, class_names, os.path.join(args.out_dir, f"{name}_test_predictions.csv"))

        joblib.dump(clf, os.path.join(args.out_dir, f"{name}_model.joblib"))
        summary.append({
            "model":name,"val_acc":val_acc,"val_macro_f1":val_f1,""
            "test_acc":test_acc,"test_macro_f1":test_f1,"hog_dim":int(Xtr.shape[1]),
            "seed":args.seed,"clahe_clip":args.clahe_clip,"clahe_grid":str(grid),
            "hog_ppc":args.hog_ppc,"hog_cpb":args.hog_cpb,"hog_orientations":args.hog_orientations})
    pd.DataFrame(summary).sort_values("val_macro_f1", ascending=False).to_csv(
        os.path.join(args.out_dir, "summary.csv"), index=False)
    print(f"✅✅✅ Saved outputs to: {args.out_dir}")

if __name__ == "__main__":
    main()
