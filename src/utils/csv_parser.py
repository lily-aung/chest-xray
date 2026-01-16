import pandas as pd
def resolve_csv_columns(df: pd.DataFrame) -> tuple[str, str]:
    for c in ["path", "image_path", "filepath", "filename", "image", "img_path"]:
        if c in df.columns:
            img_col = c
            break
    else:
        raise ValueError(f"No image path column found. Columns={list(df.columns)}")
    for c in ["label", "target", "class", "y"]:
        if c in df.columns:
            label_col = c
            break
    else:
        raise ValueError(f"No label column found. Columns={list(df.columns)}")
    return img_col, label_col
