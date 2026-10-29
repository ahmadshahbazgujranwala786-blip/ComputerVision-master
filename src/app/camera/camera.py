import mmap
import struct
import threading
import time

import cv2
import numpy as np

SOURCE_WIDTH = 1280
SOURCE_HEIGHT = 720

# Preserve the existing Imou live-video crop used by Smart Attendance.
CROP_X1 = 261
CROP_Y1 = 85
CROP_X2 = 1279
CROP_Y2 = 663

# Camera Emulator publishes the complete Imou window through a Windows
# memory-mapped frame buffer. Smart Attendance reads the newest complete
# BGRA frame directly; DirectShow is no longer part of this path.
_MAP_NAME = "SmartAttendance.ImouFrame"
_MAP_CAPACITY = 32 * 1024 * 1024
_HEADER_SIZE = 32
_MAGIC = 0x53414631  # SAF1


class Camera:
    def __init__(self, camera_index=None, retries=20):
        self._map = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._sequence = 0
        self._open(retries)

    def _crop_frame(self, frame):
        height, width = frame.shape[:2]
        x1 = max(0, min(CROP_X1, width - 1))
        y1 = max(0, min(CROP_Y1, height - 1))
        x2 = max(x1 + 1, min(CROP_X2, width))
        y2 = max(y1 + 1, min(CROP_Y2, height))
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        return cv2.resize(cropped, (SOURCE_WIDTH, SOURCE_HEIGHT), interpolation=cv2.INTER_LINEAR)

    def _open(self, retries):
        last_error = None

        for _ in range(retries):
            try:
                self._map = mmap.mmap(
                    -1,
                    _MAP_CAPACITY,
                    tagname=_MAP_NAME,
                    access=mmap.ACCESS_WRITE,
                )
                magic = struct.unpack_from("<I", self._map, 20)[0]
                if magic != _MAGIC:
                    self._map.close()
                    self._map = None
                    raise RuntimeError(
                        "Camera Emulator shared-frame publisher is not ready"
                    )

                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    frame = self._read_latest()
                    if frame is not None:
                        with self._lock:
                            self._frame = frame

                        self._thread = threading.Thread(
                            target=self._reader_loop,
                            daemon=True,
                            name="camera-shared-frame-reader",
                        )
                        self._thread.start()

                        print(
                            "Camera Emulator opened successfully "
                            "through direct WGC shared-memory frame bridge",
                            flush=True,
                        )
                        return

                    time.sleep(0.01)

                last_error = RuntimeError(
                    "Camera Emulator shared frame exists but produced no frame"
                )

            except Exception as exc:
                last_error = exc

            if self._map is not None:
                try:
                    self._map.close()
                except Exception:
                    pass
                self._map = None
            time.sleep(0.2)

        raise RuntimeError(
            f"Could not initialize direct Imou camera bridge: {last_error}"
        )

    def _read_latest(self):
        if self._map is None:
            return None

        for _ in range(3):
            seq1 = struct.unpack_from("<Q", self._map, 0)[0]
            if seq1 == 0 or (seq1 & 1):
                time.sleep(0.001)
                continue

            # Skip an unchanged publication instead of copying and resizing it repeatedly.
            if seq1 == self._sequence:
                return None

            width = struct.unpack_from("<I", self._map, 8)[0]
            height = struct.unpack_from("<I", self._map, 12)[0]
            channels = struct.unpack_from("<I", self._map, 16)[0]

            if (
                width <= 0
                or height <= 0
                or channels != 3
                or width > 4096
                or height > 4096
            ):
                return None

            size = width * height * channels
            raw = np.frombuffer(
                self._map,
                dtype=np.uint8,
                count=size,
                offset=_HEADER_SIZE,
            ).copy()

            seq2 = struct.unpack_from("<Q", self._map, 0)[0]
            if seq1 == seq2 and not (seq2 & 1):
                bgr = raw.reshape((height, width, 3))
                self._sequence = seq2
                return self._crop_frame(bgr)

        return None

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                frame = self._read_latest()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(0.002)
            except Exception:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._stop.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        if self._map is not None:
            try:
                self._map.close()
            except Exception:
                pass

        self._map = None
        self._thread = None

        with self._lock:
            self._frame = None
