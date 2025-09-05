import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    # Load metadata
    df = pd.read_csv("images/metadata/images.csv.gz")

    # Filter by size
    df = df[(df["width"] >= 1000) & (df["height"] >= 1000)]

    base_dir = Path("images/small")
    valid_paths = []

    for rel_path in tqdm(df["path"], desc="Checking files"):
        full_path = base_dir / rel_path
        if full_path.is_file():
            valid_paths.append("../" + full_path.as_posix())

    print(f"Found {len(valid_paths)} images ≥1000px")

    out_file = Path("valid_images.txt")
    out_file.write_text("\n".join(valid_paths), encoding="utf-8")


if __name__ == "__main__":
    main()
