import datetime

from google.cloud import speech
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult


class AsrGoogle(AsrBase):
    def __init__(self):
        self.client = speech.SpeechClient()

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe an audio file using Google Cloud Speech-to-Text. Returns the TranscriptionResult object."""

        start_time = datetime.datetime.now()

        with open(transcript_obj.audio, "rb") as f:
            audio_content = f.read()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        response = self.client.recognize(config=config, audio=audio)

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        if response.results:
            transcription = response.results[0].alternatives[0].transcript
            return TranscriptionResult(
                service="google",
                transcription=transcription,
                confidence=None,  # Google non fornisce un punteggio di confidenza
                processing_time=processing_time,
                error=False,
                wer=wer(transcription, transcript_obj.transcription_ground_truth),
            )
        else:
            return TranscriptionResult(service="google", transcription="", error=True)


# Esempio di utilizzo
# asr_google = AsrGoogle()
# result = asr_google.transcribe(transcript_obj)
# print(result)
