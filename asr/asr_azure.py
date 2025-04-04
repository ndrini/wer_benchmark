import datetime
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

from . import root_path

# from .asr_whisper import ASRWhisper

# from .asr_whisper import ASRWhisper

# Carica le variabili d'ambiente dal file .env
load_dotenv(root_path / ".env")

# Recupera le variabili d'ambiente necessarie
AZURE_AI_SPEECH_SERVICE_KEY = os.getenv("AZURE_AI_SPEECH_SERVICE_KEY", "")
AZURE_AI_SPEECH_SERVICE_REGION = os.getenv("AZURE_AI_SPEECH_SERVICE_REGION", "")
AZURE_AI_SPEECH_SERVICE_URL = os.getenv(
    "AZURE_AI_SPEECH_SERVICE_URL", "https://centralindia.api.cognitive.microsoft.com/"
)


class AsrAzure(AsrBase):

    def __init__(self):
        subscription_key: str = AZURE_AI_SPEECH_SERVICE_KEY
        region: str = AZURE_AI_SPEECH_SERVICE_REGION
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        self.audio_config = None  # defined in each transcription

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe a audio file. Returns the object TranscriptionResult."""

        start_time = datetime.datetime.now()

        self.audio_config = speechsdk.audio.AudioConfig(
            filename=str(transcript_obj.audio_wav_path)
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=self.audio_config
        )
        result = recognizer.recognize_once()

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        print(
            "\n\n",
            "+++++ AsrAzure +++++",
            f"*** during transcribe function, \n\t\t\tresult.reason: {result.reason}, \n\tresult.text:\n\t\t {result.text}",
            f"\n\ttranscript_obj.transcription_ground_truth:\n\t\t{transcript_obj.transcription_ground_truth}",
        )

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return TranscriptionResult(
                service="azure",
                transcription=result.text,
                # confidence=result.confidence if result.confidence else None,
                processing_time=processing_time,
                error=False,
                wer=wer(result.text, transcript_obj.transcription_ground_truth),
            )
        else:
            return TranscriptionResult(service="azure", transcription="", error=True)
