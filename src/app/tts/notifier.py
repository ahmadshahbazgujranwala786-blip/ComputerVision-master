from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from src.app.tts import TTSEngine
from src.app.tts.service import TTSService


class TTSNotifier:
    """Queue voice notifications and cancel all playback on session shutdown."""

    def __init__(self, voice: str = "F5", language: str = "en") -> None:
        self.language = language
        self.queue: Queue[str | None] = Queue()
        self._stopping = Event()
        self._shutdown_lock = __import__("threading").Lock()

        self.service = TTSService(
            engine=TTSEngine(voice=voice),
            output_path=Path("data/tts_output.wav"),
        )
        self.worker = Thread(
            target=self._worker,
            name="tts-worker",
            daemon=True,
        )
        self.worker.start()

    def speak(self, text: str) -> None:
        if not text.strip() or self._stopping.is_set():
            return
        self.queue.put(text)

    def _worker(self) -> None:
        while True:
            text = self.queue.get()
            try:
                if text is None or self._stopping.is_set():
                    return
                try:
                    self.service.speak(text=text, language=self.language)
                except Exception as exc:
                    if not self._stopping.is_set():
                        print("TTS playback error:", exc)
            finally:
                self.queue.task_done()

    def shutdown(self, wait: bool = True) -> None:
        with self._shutdown_lock:
            if not self._stopping.is_set():
                self._stopping.set()

                # Stop currently playing audio first, then discard queued speech.
                try:
                    self.service.stop()
                except Exception as exc:
                    print("TTS stop warning:", exc)

                while True:
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        break
                    else:
                        self.queue.task_done()

                self.queue.put(None)

        if wait and self.worker.is_alive():
            self.worker.join(timeout=3.0)
