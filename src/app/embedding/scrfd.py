from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class SCRFD:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Limit ONNX Runtime thread fan-out. On this laptop, four intra-op
        # threads substantially reduce ArcFace latency versus the default.
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

        self.input_height = self.input_info.shape[2]
        self.input_width = self.input_info.shape[3]

        self.strides = [8, 16, 32]
        self.num_anchors = 2

    def preprocess(self, frame: np.ndarray):
        original_height, original_width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
        1.0 / 128.0,
        (
        self.input_width,
        self.input_height,
        ),
        (
            127.5,
            127.5,
            127.5,
        ),
        swapRB=True,
    )

        return (
        blob,
        original_width,
        original_height,
    )

    def generate_centers(self, stride: int):
        feature_height = self.input_height // stride
        feature_width = self.input_width // stride

        centers = []

        for y in range(feature_height):
            for x in range(feature_width):
                center_x = x * stride# reference points
                center_y = y * stride

                for _ in range(self.num_anchors):
                    centers.append(
                        [center_x, center_y]
                    )

        return np.array(
            centers,
            dtype=np.float32,
        )

    def decode_level(
        self,
        scores,
        bbox_predictions,
        kps_predictions,
        stride,
    ):
        centers = self.generate_centers(stride)

        scores = scores.reshape(-1)

        bbox_predictions = bbox_predictions.reshape(
            -1,
            4,
        )

        landmarks = kps_predictions.reshape(
            -1,
            5,
            2,
        )

        valid_indices = np.where(
            scores >= self.confidence_threshold
        )[0]

        scores = scores[valid_indices]
        bbox_predictions = bbox_predictions[valid_indices]
        landmarks = landmarks[valid_indices]
        centers = centers[valid_indices]

        if len(scores) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
            )

        bbox_predictions = bbox_predictions * stride

        x1 = centers[:, 0] - bbox_predictions[:, 0]# distances from the center which are we using to find the coordinates
        y1 = centers[:, 1] - bbox_predictions[:, 1]

        x2 = centers[:, 0] + bbox_predictions[:, 2]
        y2 = centers[:, 1] + bbox_predictions[:, 3]

        boxes = np.stack(
            [x1, y1, x2, y2],
            axis=1,
        )

        landmarks = (
            landmarks * stride
            + centers[:, None, :]
        )

        return boxes, scores, landmarks

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (
            (x2 - x1)
            * (y2 - y1)
        )

        order = scores.argsort()[::-1]

        keep = []

        while len(order) > 0:
            current = order[0]

            keep.append(current)

            remaining = order[1:]

            if len(remaining) == 0:
                break

            xx1 = np.maximum(
                x1[current],
                x1[remaining],
            )

            yy1 = np.maximum(
                y1[current],
                y1[remaining],
            )

            xx2 = np.minimum(
                x2[current],
                x2[remaining],
            )

            yy2 = np.minimum(
                y2[current],
                y2[remaining],
            )

            width = np.maximum(
                0,
                xx2 - xx1,
            )

            height = np.maximum(
                0,
                yy2 - yy1,
            )

            intersection = width * height

            union = (
                areas[current]
                + areas[remaining]
                - intersection
            )

            iou = intersection / np.maximum(
                union,
                1e-6,
            )

            order = remaining[
                iou <= self.nms_threshold
            ]

        return keep

    def detect(self, frame: np.ndarray):
        (
            input_tensor,
            original_width,
            original_height,
        ) = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
            },
        )

        all_boxes = []
        all_scores = []
        all_landmarks = []

        for level, stride in enumerate(self.strides):
            score_predictions = outputs[level]
            bbox_predictions = outputs[level + 3]
            kps_predictions = outputs[level + 6]

            boxes, scores, landmarks = self.decode_level(
                score_predictions,
                bbox_predictions,
                kps_predictions,
                stride,
            )

            if len(scores) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_landmarks.append(landmarks)

        if not all_boxes:
            return []

        boxes = np.concatenate(
            all_boxes,
            axis=0,
        )

        scores = np.concatenate(
            all_scores,
            axis=0,
        )

        landmarks = np.concatenate(
            all_landmarks,
            axis=0,
        )

        keep_indices = self.nms(
            boxes,
            scores,
        )

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        landmarks = landmarks[keep_indices]

        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height

        boxes[:, [0, 2]] *= scale_x# x1 and x2 coordinates of boxes being scaled
        boxes[:, [1, 3]] *= scale_y# y1 and y2 coordinates of boxes being scaled

        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y

        detections = []

        for box, score, landmark in zip(
            boxes,
            scores,
            landmarks,
        ):
            detections.append(
                {
                    "box": box,
                    "score": float(score),
                    "landmarks": landmark,
                }
            )

        return detections