import os
import numpy as np
from typing import Dict, Optional, List, Union


class FaceDatabase:

    def __init__(self, filepath: Optional[str] = None):

        self._data: Dict[str, np.ndarray] = {}
        self.filepath = filepath
        if filepath and os.path.isfile(filepath):
            self.load(filepath)

    def add_person(self, name: str, embedding: np.ndarray) -> None:

        if name in self._data:

            raise ValueError(f"Person '{name}' already exists in the database.")
        
        if not isinstance(embedding, np.ndarray):

            raise TypeError("Embedding must be a numpy array.")
        
        if embedding.shape != (512,):

            raise ValueError("Embedding must have shape (512,).")
        
        if embedding.dtype != np.float32:

            embedding = embedding.astype(np.float32)

        self._data[name] = embedding.copy()

    def has_person(self, name: str) -> bool:
     
        return name in self._data

    def get_embedding(self, name: str) -> Optional[np.ndarray]:

        return self._data.get(name)

    def get_all_people(self) -> Dict[str, np.ndarray]:

        return self._data.copy()

    def delete_person(self, name: str) -> None:

        if name not in self._data:
            raise KeyError(f"Person '{name}' not found.")
        del self._data[name]

    def clear(self) -> None:
        return 
        
    def save(self, filepath=None):

        if filepath is None:
          filepath = self.filepath

        if filepath is None:
            raise ValueError(
            "No filepath specified for saving."
        )

        os.makedirs(
        os.path.dirname(filepath) or ".",
        exist_ok=True,
    )

        names = list(self._data.keys())

        embeddings = np.array(
        [
            self._data[name]
            for name in names
        ],
        dtype=np.float32,
    )

        np.savez_compressed(
        filepath,
        names=names,
        embeddings=embeddings,
    )
    def load(self, filepath: Optional[str] = None) -> None:

        if filepath is None:
            filepath = self.filepath
        if filepath is None:
            raise ValueError("No filepath specified for loading.")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Database file '{filepath}' not found.")
        with np.load(filepath, allow_pickle=True) as data:
            names = data["names"].tolist()  # list of strings
            embeddings = data["embeddings"]  # shape (N, 512)
        # Rebuild internal dict
        self._data = {name: embeddings[i].astype(np.float32) for i, name in enumerate(names)}

    def __len__(self) -> int:
        """Return the number of registered people."""
        return len(self._data)

    def __repr__(self) -> str:
        return f"FaceDatabase(filepath={self.filepath!r}, entries={len(self._data)})"