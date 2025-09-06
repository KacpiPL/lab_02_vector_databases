import joblib
import torch
from PIL import Image
from itertools import islice, batched
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from db_init import engine, Img  # 👈 reuse your existing engine + model

# ---- config ----
MAX_IMAGES = 50   # (I use macbook air m1, so limit to 50 for demo)
BATCH_SIZE = joblib.cpu_count(only_physical_cores=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32", device=device)


def insert_images(engine, rows):
    """Insert batch of images with embeddings into DB, skip duplicates."""
    if not rows:
        return
    stmt = (
        pg_insert(Img.__table__)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["image_path"])
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def vectorize_images(engine, model, image_paths):
    """Encode images in batches and insert embeddings into DB."""
    limited_paths = list(islice(image_paths, MAX_IMAGES))

    with tqdm(total=len(limited_paths), desc="Vectorizing images") as pbar:
        for paths_batch in batched(limited_paths, BATCH_SIZE):
            paths_batch = list(paths_batch)

            # Load images
            images = [Image.open(p).convert("RGB") for p in paths_batch]

            # Compute embeddings
            emb = model.encode(
                images,
                batch_size=len(images),
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            if torch.is_tensor(emb):
                emb = emb.detach().cpu().tolist()

            # Prepare rows for DB insert
            rows = [
                {"image_path": str(p), "embedding": e}
                for p, e in zip(paths_batch, emb)
            ]

            insert_images(engine, rows)
            pbar.update(len(rows))


# --- Run vectorization when script executed directly ---
if __name__ == "__main__":
    print(">>> image_vectorization.py starting...", flush=True)

    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    paths = []
    root = Path("images/small")
    for pat in exts:
        paths.extend(root.rglob(pat))

    image_paths = sorted(set(paths))
    print(f">>> Found {len(image_paths)} images under {root.resolve()}", flush=True)

    if not image_paths:
        print(">>> No images found to vectorize. Check the folder and patterns.", flush=True)
    else:
        vectorize_images(engine, model, image_paths)
        print(">>> Done. Image embeddings inserted into DB.", flush=True)
        