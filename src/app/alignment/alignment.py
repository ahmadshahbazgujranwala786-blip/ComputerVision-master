import cv2
import numpy as np


class FaceAligner:
    def __init__(self):
        self.output_size = (112, 112)

        self.reference_points = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def align(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        landmarks = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if landmarks.shape != (5, 2):
            raise ValueError(
                "Landmarks must have shape (5, 2)"
            )

        transformation_matrix, _ = cv2.estimateAffinePartial2D(# Combines the landmarks and the actual reference we want
            landmarks,
            self.reference_points,
            method=cv2.LMEDS,
        )

        if transformation_matrix is None:
            raise RuntimeError(
                "Could not calculate face alignment transformation"
            )

        aligned_face = cv2.warpAffine(# Actually peforms the transformation for the ArcFace
            frame,
            transformation_matrix,
            self.output_size,
        )

        return aligned_face