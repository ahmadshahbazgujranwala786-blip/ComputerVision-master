from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ArcFace:
    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = Path(model_path)

        # Four intra-op threads benchmark faster on this laptop than the default.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name

        self.output_info = self.session.get_outputs()[0]
        self.output_name = self.output_info.name

    def preprocess(
        self,
        aligned_face: np.ndarray,
    ) -> np.ndarray:

        face = cv2.cvtColor(
            aligned_face,
            cv2.COLOR_BGR2RGB,
        )

        face = face.astype(
            np.float32
        )

        face = (face - 127.5) / 127.5

        face = np.transpose(
            face,
            (2, 0, 1),
        )

        face = np.expand_dims(
            face,
            axis=0,
        )

        return face

    def get_embedding(
        self,
        aligned_face: np.ndarray,
    ) -> np.ndarray:

        input_tensor = self.preprocess(
            aligned_face
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor,
            },
        )[0]

        embedding = output[0]

        embedding = embedding / (
            np.linalg.norm(embedding)
            + 1e-12
        )

        return embedding