import faiss
import numpy as np
import sys
from pathlib import Path


DIMENSION = 512


def default_index_path() -> Path:
    """
    Resolve one shared persistent index for every app process.
    Frozen builds store writable data beside the executable, never in
    PyInstaller's per-process temporary _MEIPASS extraction directory.
    Source runs keep the project-relative data path.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data" / "face_index.faiss"
    return Path("data/face_index.faiss")


DEFAULT_INDEX_PATH = default_index_path()


class VectorStore:
    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
    ):
        """
        Initialize the persistent FAISS vector store.

        If a FAISS index already exists, it is loaded from disk.
        Otherwise, a new 512-dimensional IndexIDMap is created.
        """

        self.index_path = Path(index_path)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.index_path.exists():

            self.index = faiss.read_index(
                str(self.index_path)
            )

            print("FAISS index loaded.")

        else:

            base_index = faiss.IndexFlatIP(# IP = Inner product(cosine similarity), performs the vectore comparison
                DIMENSION
            )

            self.index = faiss.IndexIDMap(# allows us to map the IDs with the vectors store
                base_index
            )

            print("New FAISS index created.")

    def save(self):

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

    def add_vector(
        self,
        vector_id: int,
        embedding: np.ndarray,
    ):

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(
            1,
            DIMENSION,
        )

        faiss.normalize_L2(
            vector
        )

        # A student ID must identify exactly one face vector.
        # Never let FAISS silently accumulate duplicate IDs.
        existing_ids = (
            faiss.vector_to_array(self.index.id_map).astype(np.int64)
            if hasattr(self.index, "id_map")
            else np.array([], dtype=np.int64)
        )
        if int(vector_id) in existing_ids:
            raise ValueError(
                f"FAISS already contains student ID {int(vector_id)}; "
                "refusing to create a duplicate mapping."
            )

        ids = np.array([int(vector_id)], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        try:
            self.save()
        except Exception:
            # Do not leave an in-memory orphan if persistence fails.
            self.index.remove_ids(ids)
            try:
                self.save()
            except Exception:
                pass
            raise

    def remove_vector(self, vector_id: int):

         ids = np.array(
        [vector_id],
        dtype=np.int64,
    )

         removed = self.index.remove_ids(
        ids
    )

         self.save()

         return removed
    def search(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ):

        if self.index.ntotal == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
            )

        k = min(
            k,
            self.index.ntotal,
        )

        query = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(
            1,
            DIMENSION,
        )

        faiss.normalize_L2(
            query
        )

        scores, ids = self.index.search(
            query,
            k,
        )

        return (
            ids[0],
            scores[0],
        )

    def find_existing(
        self,
        embedding: np.ndarray,
        threshold: float = 0.60,
    ):

        if self.index.ntotal == 0:
            return None

        ids, scores = self.search(
            embedding,
            k=1,
        )

        if len(ids) == 0:
            return None

        best_id = int(ids[0])
        best_score = float(scores[0])

        if best_score >= threshold:

            return {
                "vector_id": best_id,
                "similarity": best_score,
            }

        return None

    @property
    def count(self):


        return self.index.ntotal

