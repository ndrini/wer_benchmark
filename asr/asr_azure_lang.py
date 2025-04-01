import datetime
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

from . import root_path

load_dotenv(root_path / ".env")

# Recupera le variabili d'ambiente necessarie
AZURE_AI_SPEECH_SERVICE_KEY = os.getenv("AZURE_AI_SPEECH_SERVICE_KEY", "")
AZURE_AI_SPEECH_SERVICE_REGION = os.getenv("AZURE_AI_SPEECH_SERVICE_REGION", "")
AZURE_AI_SPEECH_SERVICE_URL = os.getenv(
    "AZURE_AI_SPEECH_SERVICE_URL", "https://centralindia.api.cognitive.microsoft.com/"
)


class AsrAzureLang(AsrBase):

    def __init__(self):
        self.subscription_key: str = AZURE_AI_SPEECH_SERVICE_KEY
        self.region: str = AZURE_AI_SPEECH_SERVICE_REGION

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe a audio file. Returns the object TranscriptionResult."""

        start_time = datetime.datetime.now()

        speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key, region=self.region
        )
        # speech_config.speech_recognition_language =  transcript_obj.language  # Specifica la lingua dalla trascrizione

        speech_config.speech_recognition_language = (
            "es-ES"  # Specifica la lingua dalla trascrizione
        )

        audio_config = speechsdk.audio.AudioConfig(
            filename=transcript_obj.audio_wav_path
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
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
                service="azure_lang",
                transcription=result.text,
                processing_time=processing_time,
                error=False,
                wer=wer(result.text, transcript_obj.transcription_ground_truth),
            )
        else:
            return TranscriptionResult(
                service="azure_lang", transcription="", error=True
            )
