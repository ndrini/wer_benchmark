import datetime
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv
from jiwer import wer
from openai import OpenAI

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

from . import root_path

# Carica le variabili d'ambiente dal file .env
load_dotenv(root_path / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise ValueError(
        "The OpenAI API key is missing. Set it as an environment variable."
    )


class AsrOpenaiGpt4oTr(AsrBase):
    """ASR engine for OpenAI GPT-4o Transcription."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe an audio file using OpenAI GPT-4o Transcription. Returns the TranscriptionResult object."""
        start_time = datetime.datetime.now()

        try:
            with open(transcript_obj.audio_wav_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                )
            openai_transcription = transcription.text
            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            print("\n\n", "+++++ OpenAI GPT-4o Transcription +++++")
            print("OpenAI Transcription: ", openai_transcription)

            # Calcola il WER se è disponibile la trascrizione di riferimento
            wer_score = None
            if transcript_obj.transcription_ground_truth:
                wer_score = wer(
                    transcript_obj.transcription_ground_truth, openai_transcription
                )
                print("++++++ WER +++++ ", wer_score)

            return TranscriptionResult(
                service="openai_gpt_4o_tr",
                transcription=openai_transcription,
                confidence=None,  # OpenAI non fornisce un punteggio di confidenza diretto per questo modello
                processing_time=processing_time,
                error=False,
                wer=wer_score,
            )

        except Exception as e:
            logging.error(f"Failed to transcribe with OpenAI GPT-4o: {e}")
            return TranscriptionResult(
                service="openai_gpt_4o_tr",
                transcription="",
                error=True,
                processing_time=(datetime.datetime.now() - start_time).total_seconds(),
            )
        except FileNotFoundError:
            logging.error(f"Audio file not found: {transcript_obj.audio_wav_path}")
            return TranscriptionResult(
                service="openai_gpt_4o_tr",
                transcription="",
                error=True,
                processing_time=(datetime.datetime.now() - start_time).total_seconds(),
            )
