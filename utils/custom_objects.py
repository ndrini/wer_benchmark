from typing import Optional

from pydantic import BaseModel


class Trace(BaseModel):
    audio: bytes
    transcription: str
    language: Optional[str] = None
