import datetime

import azure.cognitiveservices.speech as speechsdk
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult


class AsrAzure(AsrBase):
    def __init__(self, subscription_key: str, region: str):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        self.audio_config = None  # Se define en cada transcripción

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe a audio file . Returns the object TranscriptionResult."""

        start_time = datetime.datetime.now()

        self.audio_config = speechsdk.audio.AudioConfig(filename=transcript_obj.audio)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=self.audio_config
        )
        result = recognizer.recognize_once()

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return TranscriptionResult(
                service="azure",
                transcription=result.text,
                confidence=result.confidence,
                processing_time=processing_time,
                error=False,
                wer=wer(result.text, transcript_obj.transcription_ground_truth),
            )
        else:
            return TranscriptionResult(
                service_name="azure", transcription="", error=True
            )
