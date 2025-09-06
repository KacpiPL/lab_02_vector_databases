import matplotlib.pyplot as plt
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

class ImageSearch:
    def __init__(self, engine, model):
        self.engine = engine
        self.model = model

    def __call__(self, image_description: str, k: int = 5):
        paths = self.find_similar_images(image_description, k)
        self.display_images(paths)
        return paths

    def find_similar_images(self, image_description: str, k: int):
        query_vec = self.model.encode(
            [image_description],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].tolist()

        stmt = text("""
            SELECT image_path
            FROM images
            ORDER BY embedding <=> :qvec
            LIMIT :k
        """).bindparams(qvec=query_vec, k=int(k))

        with Session(self.engine) as sess:
            rows = sess.execute(stmt).scalars().all()
        return rows

    def display_images(self, image_paths):
        if not image_paths:
            print("No images to display.")
            return

        k = len(image_paths)
        fig, axes = plt.subplots(1, k, figsize=(4*k, 4))
        if k == 1:
            axes = [axes]

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                axes[i].imshow(img)
                axes[i].axis("off")
                axes[i].set_title(f"{i+1}")
            except Exception as e:
                axes[i].axis("off")
                axes[i].set_title(f"Failed: {e}")

        plt.tight_layout()
        plt.show()
