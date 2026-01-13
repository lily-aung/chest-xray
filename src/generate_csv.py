from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_csv_from_directory(base_dir: Path, output_csv: Path):
    data = []

    class_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    for label, class_dir in enumerate(class_dirs):
        for image_path in class_dir.iterdir():
            if image_path.is_file():
                data.append([str(image_path), label])

    df = pd.DataFrame(data, columns=["image_path", "label"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"{output_csv.name}: {len(df)} samples")


def generate_csv_files():
    base = PROJECT_ROOT / "data" / "raw" / "archive"

    create_csv_from_directory(base / "train", PROJECT_ROOT / "data" / "train.csv")
    create_csv_from_directory(base / "val", PROJECT_ROOT / "data" / "val.csv")
    create_csv_from_directory(base / "test", PROJECT_ROOT / "data" / "test.csv")


if __name__ == "__main__":
    generate_csv_files()


