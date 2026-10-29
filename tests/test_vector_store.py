import numpy as np

from src.vector_store import VectorStore


def test_add_and_search():
    store = VectorStore()

    # Student 101 ka sample 512-D embedding
    embedding = np.random.rand(512).astype(np.float32)

    # Student ID = 101
    store.add_vector(101, embedding)

    # Same embedding ko query kar rahe hain
    ids, scores = store.search(embedding, k=1)

    assert ids[0] == 101
    assert scores[0] > 0.99

    print("FAISS test passed!")
    print("Student ID:", ids[0])
    print("Cosine similarity:", scores[0])


if __name__ == "__main__":
    test_add_and_search()