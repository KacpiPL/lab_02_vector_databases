from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from typing import List
from sqlalchemy import String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="mlops",
)

engine = create_engine(DB_URL, echo=True, future=True)

# --- Base class ---
class Base(DeclarativeBase):
    pass

# --- Model ---
class Img(Base):
    __tablename__ = "images"
    __table_args__ = {"extend_existing": True}

    VECTOR_LENGTH: int = 512  # CLIP ViT-B/32

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_path: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH), nullable=False)

# --- Create the table in the database ---
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Table 'images' ensured in DB.")
    