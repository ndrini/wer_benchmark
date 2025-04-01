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
        self.audio_config = None  # Se define en cada transcripción

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe a audio file. Returns the object TranscriptionResult."""

        start_time = datetime.datetime.now()

        self.audio_config = speechsdk.audio.AudioConfig(
            filename=transcript_obj.audio_wav_path
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=self.audio_config
        )
        result = recognizer.recognize_once()

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        print(
            "\n\n\n\n\n",
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            f"*** during transcribe function, \n\t\t\tresult.reason: {result.reason}, \n\t\t\tresult.text: {result.text}",
            transcript_obj.transcription_ground_truth,
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
