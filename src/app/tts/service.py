from __future__ import annotations

from pathlib import Path

import sounddevice as sd

from src.app.tts.engine import TTSEngine


class TTSService:
    def __init__(self, engine: TTSEngine, output_path: str | Path) -> None:
        self.engine = engine
        self.output_path = Path(output_path)

    def process(self, text: str, language: str = "en") -> Path:
        wav, _ = self.engine.synthesize(text, language)
        self.engine.save(wav, str(self.output_path))
        return self.output_path

    def speak(self, text: str, language: str = "en") -> None:
        wav, _ = self.engine.synthesize(text, language)
        sd.play(wav[0], samplerate=44100)
        sd.wait()

    def stop(self) -> None:
        """Immediately stop active sounddevice playback."""
        sd.stop()
