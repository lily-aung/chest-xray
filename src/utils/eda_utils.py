from collections import Counter
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import os
from src.utils.image_utils import (clahe_safe, img_metrics, load_gray, global_normalize, gamma_correction)

sns.set_theme(style="whitegrid", context="talk")
CLASS_ORDER =["normal", "pneumonia", "tuberculosis"] 
CLASS_COLORS = { "normal": "#83F87B",  "pneumonia": "#F7B986",  "tuberculosis": "#F78174"}

# Dataset construction
def collect_metadata(root_dir, cache_path="metadata.csv", refresh=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        print(f"Loading metadata from {cache_path}")
        return pd.read_csv(cache_path)
    print("Collecting metadata from disk...")
    records = [
        {"path": str(p), "filename": p.name, "split": split, "label": cls}
        for split in ["train", "val", "test"]
        for cls in ["normal", "pneumonia", "tuberculosis"]
        for p in (Path(root_dir) / split / cls).glob("*")
        if (Path(root_dir) / split / cls).exists() ]
    df = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Metadata saved to {cache_path}")
    return df

# CLASS DISTRIBUTION
def class_distribution(df, split="train"):
    return Counter(df[df.split == split].label)
def imbalance_ratio(df, split):
    counts = df[df.split == split].label.value_counts()
    return counts.max() / counts.min()
def plot_class_distribution(counter, title, ax, class_order=None, show_values=True):
    labels = class_order or list(counter)
    counts = [counter.get(l, 0) for l in labels]
    total = sum(counts)
    bars = ax.bar([l.capitalize() for l in labels], counts, width=0.4,
                  color=[CLASS_COLORS.get(l, "gray") for l in labels], edgecolor="none")
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45, colors=(0,0,0,0.7), labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.grid(axis="x", visible=False)
    ymax = ax.get_ylim()[1]
    base_offset, gap = ymax*0.01, ymax*0.03
    if show_values:
        for b in bars:
            c = b.get_height()
            pct = 100*c/total if total else 0
            ax.text(b.get_x()+b.get_width()/2, c+base_offset, f"({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8, color="dimgray")
            ax.text(b.get_x()+b.get_width()/2, c+base_offset+gap, f"{int(c):,}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.5, -0.22, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.margins(x=0.4)

def plot_imbalance_ratio(imb_df, class_order, class_colors, title="Class Imbalance Across Splits", figsize=(5,3)):
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=imb_df, x="label", y="imbalance_ratio", hue="split",
                     order=class_order, width=0.7, palette=["#063D95","#679EF7","#96B2F8"], edgecolor="none")
    
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30, colors=(0,0,0,0.7), labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.axhline(1.0, ls="--", lw=0.8, c="gray", alpha=0.7)
    ax.set_ylabel("Imbalance Ratio (max / class)", fontsize=10)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)

    ymax = ax.get_ylim()[1]
    offset = ymax * 0.02
    for p in ax.patches:
        h = p.get_height()
        if h>0:
            ax.text(p.get_x()+p.get_width()/2, h+offset, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, color="dimgray")
    ax.margins(x=0.05)
    ax.legend(title="", fontsize=9)
    plt.tight_layout()
    #plt.savefig("imbalance_ratio.png", dpi=300, bbox_inches='tight')
    plt.show()

# EXIF METADATA
def inspect_exif_presence(df, max_samples=None):
    df = df.sample(max_samples, random_state=42) if max_samples else df
    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        try:
            exif = Image.open(r.path).getexif()
            rows.append({"has_exif": bool(exif),
                "num_exif_tags": len(exif) if exif else 0,
                "label": r.label, "split": r.split})
        except Exception:
            pass
    return pd.DataFrame(rows)

# DATA QUALITY CHECKS
def find_corrupt_images(df):
    bad = []
    for p in df.path:
        try:
            with Image.open(p) as img: img.verify()
        except (UnidentifiedImageError, OSError):
            bad.append(p)
    return df[df.path.isin(bad)]

def compute_size_flags(df, max_samples=None, min_size=224, max_ar=2.0):
    df = df.sample(min(len(df), max_samples), random_state=42) if max_samples else df
    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        try:
            with Image.open(r.path) as img:
                w, h = img.size
                ar = max(w / h, h / w)
                rows.append({
                    "width": w, "height": h, "aspect_ratio": ar,
                    "label": r.label, "split": r.split,
                    "small_image": (w < min_size) or (h < min_size),
                    "extreme_aspect_ratio": ar > max_ar
                })
        except Exception:
            continue
    return pd.DataFrame(rows)

def plot_size_distributions(size_df, classorder, save_path=None):
    CLASS_COLORS_DISPLAY = {"Normal":"#15FD04","Pneumonia":"#FA7100","Tuberculosis":"#F71900"}
    df_plot = size_df.copy()
    df_plot['display_label'] = df_plot['label'].str.capitalize()
    hue_order = [c.capitalize() for c in classorder] if classorder else list(CLASS_COLORS_DISPLAY.keys())
    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(data=df_plot, x="width", y="height", hue="display_label",
                         hue_order=hue_order, palette=CLASS_COLORS_DISPLAY, alpha=0.5, s=30, edgecolor=None)
    plt.axvline(224, ls="--", c="red", lw=0.5)
    plt.axhline(224, ls="--", c="red", lw=0.5)
    ax.set_title("Image Resolution Distribution", fontsize=10)
    ax.set_xlabel("Width (pixels)", fontsize=10)
    ax.set_ylabel("Height (pixels)", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.grid(True, linestyle="--", alpha=0.2)
    plt.legend(title="Class", bbox_to_anchor=(0.1,0.97), loc="upper left", fontsize=10, title_fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_aspect_ratio_distribution(size_df, class_colors, classorder=None, save_path=None):
    hue_order = classorder if classorder else list(class_colors.keys())
    plt.figure(figsize=(6,4))
    ax = sns.kdeplot( data=size_df,
        x="aspect_ratio", hue="label",hue_order=hue_order,palette=class_colors,
        common_norm=False, fill=True, alpha=0.4,legend=False )
    plt.axvline(2.0, ls="--", c="red", lw=0.4)
    plt.xlim(0.8, 3.5)
    ax.set_xlabel("Aspect Ratio", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Aspect Ratio Distribution by Class", fontsize=14)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    handles = [plt.Line2D([0], [0], color=class_colors[l], lw=1) for l in hue_order]
    labels = [l.capitalize() for l in hue_order]
    ax.legend(handles, labels, title="Class", fontsize=11, title_fontsize=12,
              loc="upper right", frameon=False)
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def aspect_ratio_box_plot(size_df):
    plt.figure(figsize=(5,6))
    sns.boxplot(data=size_df, x="label", y="aspect_ratio",
                palette=CLASS_COLORS)
    plt.axhline(2.0, ls="--", c="red", lw=1)
    plt.ylabel("Aspect Ratio")
    plt.title("Aspect Ratio by Class")
    plt.tight_layout()
    plt.show()


# PIXEL STATISTICS
def intensity_statistics(df, max_samples=200):
    df = df.sample(min(len(df), max_samples), random_state=42)
    vals = []
    for p in df.path:
        arr = np.array(Image.open(p).convert("L")) / 255.0
        vals.append(arr.mean())
    return dict(
        mean_intensity=np.mean(vals), std_intensity=np.std(vals),
        min_intensity=np.min(vals), max_intensity=np.max(vals))

def plot_intensity_distribution_by_class(df, class_order, figsize=(6, 4),
                                         xlabel="Mean Pixel Intensity (0-1)", ylabel="Density",
                                         title="Mean Pixel Intensity Distribution by Class"):
    from src.utils.image_utils import intensity_stats
    plt.figure(figsize=figsize)
    for lbl in class_order:
        v = intensity_stats(df[df.label == lbl])
        sns.kdeplot(v, label=lbl.capitalize(), fill=True, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# VISUAL SANITY CHECK
def viz_images(df, title, n=6):
    fig, axes = plt.subplots(1, n, figsize=(12,3))
    sample = df.sample(n, random_state=42)

    for ax, (idx, r) in zip(axes, sample.iterrows()):
        img = Image.open(r.path).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(r.label, fontsize=9)
        ax.text(5, 15, f"idx={idx}", color="yellow",
                fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=2))
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def viz_exposure_images(df, exp_df, class_colors, title, n=6):
    samples = df.sample(n, random_state=42).merge(
        exp_df[["path","black_frac","white_frac","p99"]], on="path", how="left")
    fig, ax = plt.subplots(1, n, figsize=(12,3))
    for i, r in enumerate(samples.itertuples()):
        img = Image.open(r.path).convert("L")
        ax[i].imshow(img, cmap="gray")
        label_color = class_colors.get(r.label, "black")
        idx = os.path.basename(r.path).split('-')[-1].split('.')[0]
        ax[i].set_title(
            f"{r.label.capitalize()} : idx {idx}\nblack={r.black_frac:.2f}, white={r.white_frac:.2f}, p99={r.p99:.2f}",
            fontsize=9, color=label_color)
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_exposure_set(df, title, class_colors=None):
    class_colors = class_colors or {}
    fig, axes = plt.subplots(len(df), 4, figsize=(12, 2.4 * len(df)))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(df.iterrows()):
        orig = load_gray(r.path)
        imgs = {
            "Original": orig,
            "Global": global_normalize(orig),
            "Gamma": gamma_correction(orig),
            "CLAHE": clahe_safe(orig)} #Contrast-limited adaptive histogram equalization
        label_color = class_colors.get(r.label, "black")
        for j, (name, im) in enumerate(imgs.items()):
            axes[i, j].imshow(im, cmap="gray")
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(name, fontsize=10)
            if name != "Original":
                m, s, c = img_metrics(orig, im)
                axes[i, j].text( 0.01, 0.99, f"μ={m:.2f} σ={s:.2f}\nρ={c:.2f}",
                    transform=axes[i, j].transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        axes[i, 0].text( 0.01, 0.01,
            f"{r.label.capitalize()}\n" f"black={r.black_frac:.2f} white={r.white_frac:.2f} p99={r.p99:.2f}",
            transform=axes[i, 0].transAxes, ha="left", va="bottom",
            fontsize=9, color=label_color, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def viz_blackborder(
    df, title, class_colors=None,
    n_cols=12, thumb_size=1.5,
    show_metrics=False, flag_col=None,
    flag_text="BLACK BORDER", flag_color="red"):
    """
    Visualize grayscale thumbnails with optional labels and flags.
    
    flag_col: column name in df with boolean flag (e.g., 'has_black_borders')
    """
    class_colors = class_colors or {}
    n, n_rows = len(df), (len(df) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * thumb_size, n_rows * thumb_size))
    axes = np.array(axes).reshape(-1)

    for ax, (_, r) in zip(axes, df.iterrows()):
        img = load_gray(r.path)
        ax.imshow(img, cmap="gray"); ax.axis("off")

        if flag_col and bool(r.get(flag_col, False)):
            ax.text(0.5, 0.05, flag_text, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color="white",
                    bbox=dict(facecolor=flag_color, alpha=0.85, edgecolor="none"))
        if hasattr(r, "label"):
            ax.text(0.02, 0.98, r.label, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color=class_colors.get(r.label, "white"),
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"))
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_black_border_removal(df, n_per_page=20, thumb_height=1.2, thumb_width=2.0):
    from src.utils.image_utils import remove_black_borders, black_fraction
    for start in range(0, len(df), n_per_page):
        batch = df.iloc[start:start+n_per_page]
        stats = [] 
        fig, axes = plt.subplots(len(batch), 2, figsize=(thumb_width*2, thumb_height*len(batch)))
        axes = np.atleast_2d(axes)
        for i, (_, r) in enumerate(batch.iterrows()):
            img = load_gray(r.path)
            cropped = remove_black_borders(img)
            print('img shape {} cropped img shape {} '.format(img.shape, cropped.shape))
            before = black_fraction(img)
            after = black_fraction(cropped)
            stats.append({ "before": before, "after": after,
                "reduction_pct": 100 * (before - after) / (before + 1e-6)})
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title("Original", fontsize=7)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(cropped, cmap="gray")
            axes[i, 1].set_title("After Border Removal", fontsize=7)
            axes[i, 1].axis("off")
        fig.suptitle( f"Black Border Removal (images {start+1}–{start+len(batch)})",fontsize=10)
        plt.tight_layout()
        plt.show()
        stats_df = pd.DataFrame(stats)
    return stats_df
