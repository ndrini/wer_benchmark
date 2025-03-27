# asr_whisper.py   Ejemplo con OpenAI Whisper (local o API).

import whisper

from asr.asr_base import AsrBase


class ASRWhisper(AsrBase):
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        return result["text"]
