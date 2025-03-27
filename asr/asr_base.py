import time
from abc import ABC, abstractmethod

# from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TranscriptionResult:
    service: str
    transcription: str
    transcription_itn: Optional[str] = None
    confidence: Optional[float] = None  # Azure a volte fornisce confidenza, Whisper no
    processing_time: float = 0.0
    error: bool = False
    wer: Optional[float] = None


@dataclass
class Transcription:
    audio: bytes  # binary file i.e. wav audio file
    audio_mp3: Optional[bytes] = None  # mp3 audio file
    duration: Optional[float] = None
    transcription_ground_truth: str = ""
    transcription_ground_truth_itn: Optional[str] = None
    # transcriptionResults: list[TranscriptionResult] = []
    transcriptionResults: list[TranscriptionResult] = field(default_factory=list)


class AsrBase(ABC):
    """Interfaz base para sistemas ASR."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe un archivo de audio y devuelve un objeto TranscriptionResult."""
        pass
