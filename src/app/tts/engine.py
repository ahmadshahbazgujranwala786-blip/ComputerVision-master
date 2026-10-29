from __future__ import annotations

import soundfile as sf

from supertonic import TTS


class TTSEngine:

    def __init__(
        self,
        voice: str = "F5",
    ) -> None:

        self.tts = TTS(
            auto_download=False,
        )

        self.voice_style = (
            self.tts.get_voice_style(
                voice_name=voice,
            )
        )

    def synthesize(
        self,
        text: str,
        language: str,
    ) -> tuple:

        if not text.strip():
            raise ValueError(
                "TTS text cannot be empty."
            )

        return self.tts.synthesize(
            text,
            voice_style=self.voice_style,
            lang=language,
        )

    def save(
        self,
        wav,
        output_path: str,
    ) -> None:


        sf.write(
            output_path,
            wav[0],
            44100,
        )