import cv2
import numpy as np

from app.embedding.scrfd import SCRFD
from src.app.alignment.alignment import FaceAligner
from src.app.embedding.arc_face import ArcFace
from face_db import FaceDatabase


def capture_embedding(
    detector,
    aligner,
    arcface,
    camera,
    required_samples=10,
):
    embeddings = []

    print("\nLook at the camera.")
    print("Keep your face reasonably steady.")
    print("Capturing embeddings...\n")

    while len(embeddings) < required_samples:
        success, frame = camera.read()

        if not success:
            continue

        detections = detector.detect(frame)

        if len(detections) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        elif len(detections) == 1:
            detection = detections[0]

            landmarks = detection["landmarks"]

            aligned_face = aligner.align(
                frame,
                landmarks,
            )

            embedding = arcface.get_embedding(
                aligned_face
            )

            embeddings.append(embedding)

            cv2.putText(
                frame,
                f"Capturing: {len(embeddings)}/{required_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        else:
            cv2.putText(
                frame,
                "Multiple faces detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow(
            "Face Registration",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    average_embedding = np.mean(
        embeddings,
        axis=0,
    )

    average_embedding /= (
        np.linalg.norm(average_embedding)
        + 1e-12
    )

    return average_embedding.astype(
        np.float32
    )


def main():
    name = input(
        "Enter person's name: "
    ).strip()

    if not name:
        raise ValueError(
            "Name cannot be empty."
        )

    detector = SCRFD(
        "models/scrfd/2.5g_bnkps.onnx"
    )

    aligner = FaceAligner()

    arcface = ArcFace(
        "models/arcface/w600k_r50.onnx"
    )

    database = FaceDatabase(
        "data/face_database.npz"
    )

    if database.has_person(name):
        raise ValueError(
            f"Person '{name}' already exists."
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    try:
        embedding = capture_embedding(
            detector,
            aligner,
            arcface,
            camera,
        )

        if embedding is None:
            print(
                "Registration cancelled."
            )
            return

        database.add_person(
            name,
            embedding,
        )

        database.save()

        print("\nRegistration successful.")
        print(f"Name: {name}")
        print(
            f"Embedding shape: {embedding.shape}"
        )
        print(
            f"L2 norm: {np.linalg.norm(embedding)}"
        )
        print(
            "Database saved successfully."
        )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()