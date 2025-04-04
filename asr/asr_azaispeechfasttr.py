import datetime
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

from . import root_path

# from .asr_whisper import ASRWhisper

# from .asr_whisper import ASRWhisper

# Carica le variabili d'ambiente dal file .env
load_dotenv(root_path / ".env")

SUBSCRIPTION_KEY = os.getenv("AZAISPEECH_KEY", "")
SERVICE_REGION = os.getenv("AZAISPEECH_REGION", "")
SERVICE_URL = os.getenv("AZAISPEECH_ENDOPOINT", "")


if not SUBSCRIPTION_KEY:
    raise ValueError(
        "The Azure subscription key is missing. Set it as an environment variable."
    )


class AsrAzAiSpeechFastTr(AsrBase):
    """ASR engine for Azure AI Speech Fast Transcription."""

    def __init__(self):
        self.subscription_key = SUBSCRIPTION_KEY
        self.base_url = (
            SERVICE_URL
            + "speechtotext/transcriptions:transcribe?api-version=2024-11-15"
        )
        self.headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe an audio file using Azure AI Speech Fast Transcription. Returns the TranscriptionResult object."""
        start_time = datetime.datetime.now()

        try:
            with open(transcript_obj.audio_wav_path, "rb") as audio_file:
                files = {
                    "audio": audio_file,
                    "definition": (
                        None,
                        # ["de-DE", "en-GB", "en-IN", "en-US", "es-ES", "es-MX", "fr-FR", "hi-IN", "it-IT", "ja-JP", "ko-KR", "pt-BR", "zh-CN"]
                        '{ "locales": ["es-ES","en-US","ja-JP"], "profanityFilterMode": "Masked", "channels": [0, 1] }',
                    ),
                }

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    files=files,
                )

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            transcription_result = response.json()

            # print("\n\n", "+++++ Full Response +++++", response.text)
            """
            {"duration":9209,
             "combinedPhrases":[
                {"channel":0,"text":"A todas estas las campanas seguían tocando con igual furia, prueba evidente de que el entusiasmo de los cuatro muchachos no había disminuido."}],
             "phrases":[
                {"channel":0,"offset":80,"duration":7480,
                  "text":"A todas estas las campanas seguían tocando con igual furia, prueba evidente de que el entusiasmo de los cuatro muchachos no había disminuido.","words":[
                    {"text":"A","offset":80,"duration":240},{"text":"todas","offset":320,"duration":320},{"text":"estas","offset":640,"duration":480},
                    {"text":"las","offset":1120,"duration":200},{"text":"campanas","offset":1320,"duration":520},{"text":"seguían","offset":1840,"duration":440},{"text":"tocando","offset":2280,"duration":400},{"text":"con","offset":2680,"duration":160},{"text":"igual","offset":2840,"duration":280},{"text":"furia,","offset":3120,"duration":720},{"text":"prueba","offset":3840,"duration":400},{"text":"evidente","offset":4240,"duration":440},
                    {"text":"de","offset":4680,"duration":80},{"text":"que","offset":4760,"duration":80},{"text":"el","offset":4840,"duration":160},
                    {"text":"entusiasmo","offset":5000,"duration":520},{"text":"de","offset":5520,"duration":80},{"text":"los","offset":5600,"duration":160},
                    {"text":"cuatro","offset":5760,"duration":240},
                    {"text":"muchachos","offset":6000,"duration":560},
                    {"text":"no","offset":6560,"duration":80},{"text":"había","offset":6640,"duration":200},
                    {"text":"disminuido.","offset":6840,"duration":720}
                   ],
                "locale":"es-ES",
                "confidence":0.9384509}
                ]
            }
            """
            azaispeech_fast_tr = transcription_result.get("combinedPhrases", [{}])[
                0
            ].get("text", "")

            # azaispeechfast_transcription = transcription_result.get("recognizedPhrases", [{}])[
            #     0
            # ].get("display", "")

            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            print("\n\n", "+++++ AzAiSpeechFastTrLang +++++")
            print("Azure Fast Transcription: ", azaispeech_fast_tr)
            azaispeech_fast_confidence = transcription_result.get(
                "phrases", [{"confidence": -999}]
            )[0].get("confidence")
            print("++++++ confidence +++++ ", azaispeech_fast_confidence)

            return TranscriptionResult(
                service="azaispeech_fast",
                transcription=azaispeech_fast_tr,
                confidence=None,  # azaispeech_fast_confidence # Azure Fast Transcription does not provide a confidence score
                processing_time=processing_time,
                error=False,
                wer=wer(azaispeech_fast_tr, transcript_obj.transcription_ground_truth),
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to transcribe: {e}")
            return TranscriptionResult(
                service="azaispeechfast_fast",
                transcription="",
                error=True,
                processing_time=(datetime.datetime.now() - start_time).total_seconds(),
            )
        except (KeyError, IndexError) as e:
            logging.error(f"Failed to parse transcription result: {e}")
            return TranscriptionResult(
                service="azaispeechfast_fast",
                transcription="",
                error=True,
                processing_time=(datetime.datetime.now() - start_time).total_seconds(),
            )
        except FileNotFoundError:
            logging.error(f"Audio file not found: {transcript_obj.audio_wav_path}")
            return TranscriptionResult(
                service="azure_fast",
                transcription="",
                error=True,
                processing_time=(datetime.datetime.now() - start_time).total_seconds(),
            )
