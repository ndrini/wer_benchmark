import datetime
import logging
import os
import time
from typing import Optional

import requests
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

# Configurazione delle variabili di ambiente
SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY", "")
SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION", "eastus2")
BASE_URL = f"https://{SERVICE_REGION}.api.cognitive.microsoft.com/speechtotext/v3.2"

if not SUBSCRIPTION_KEY:
    raise ValueError(
        "The Azure subscription key is missing. Set it as an environment variable."
    )


class AzAiSpeechBatch(AsrBase):
    """ASR engine for Azure AI Speech Batch."""

    def __init__(self):
        self.subscription_key = SUBSCRIPTION_KEY
        self.base_url = BASE_URL
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe an audio file using Azure AI Speech Batch."""
        start_time = datetime.datetime.now()

        operation_url = self._start_transcription(transcript_obj.audio_wav_path)
        if not operation_url:
            return TranscriptionResult(service="azure", transcription="", error=True)

        # Monitora lo stato della trascrizione
        transcription_result = self._poll_transcription_status(operation_url)

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        if transcription_result:
            azure_transcription = transcription_result.get(
                "combinedRecognizedPhrases", [{}]
            )[0].get("display", "")
            return TranscriptionResult(
                service="azure",
                transcription=azure_transcription,
                confidence=None,  # Azure Speech Batch non fornisce un punteggio di confidenza
                processing_time=processing_time,
                error=False,
                wer=wer(azure_transcription, transcript_obj.transcription_ground_truth),
            )
        else:
            return TranscriptionResult(service="azure", transcription="", error=True)

    def _start_transcription(self, audio_url: str) -> Optional[str]:
        """Invia la richiesta per creare una trascrizione e restituisce l'URL per monitorare lo stato."""
        payload = {
            "displayName": "Batch Transcription",
            "description": "Transcription of audio file",
            "locale": "en-US",
            "contentUrls": [audio_url],
            "properties": {
                "diarizationEnabled": False,
                "wordLevelTimestampsEnabled": True,
            },
        }

        response = requests.post(
            f"{self.base_url}/transcriptions", headers=self.headers, json=payload
        )

        if response.status_code != 202:
            logging.error(
                f"Failed to create transcription: {response.status_code} - {response.text}"
            )
            return None

        return response.headers.get("Operation-Location")

    def _poll_transcription_status(
        self, operation_url: str, timeout: int = 300, interval: int = 10
    ) -> Optional[dict]:
        """Monitora lo stato della trascrizione fino al completamento con timeout."""
        start_time = time.time()

        while True:
            response = requests.get(operation_url, headers=self.headers)
            if response.status_code != 200:
                logging.error(f"Failed to get transcription status: {response.text}")
                return None

            result = response.json()
            status = result.get("status")

            if status == "Succeeded":
                return result
            elif status in {"Failed", "Cancelled"}:
                logging.error(f"Transcription failed with status: {status}")
                return None

            # Controlla il timeout
            if time.time() - start_time > timeout:
                logging.error("Transcription polling timed out.")
                return None

            logging.info(
                f"Transcription status: {status}, retrying in {interval} sec..."
            )
            time.sleep(interval)  # Aspetta prima del prossimo tentativo
