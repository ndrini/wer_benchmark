import asyncio
import base64
import json
import os

import azure.cognitiveservices.speech as speechsdk
import ffmpeg
from dotenv import load_dotenv

from . import root_path

load_dotenv(root_path)

AZURE_AI_SPEECH_SERVICE_KEY = os.getenv("AZURE_AI_SPEECH_SERVICE_KEY")
AZURE_AI_SPEECH_SERVICE_REGION = os.getenv("AZURE_AI_SPEECH_SERVICE_REGION")


async def recognize_speech_from_wav(
    wav_file_path: str = "", language: str = "es-ES"
) -> str:
    """
    Asynchronous speech recognition from a WAV file using Azure Cognitive Services.

    Args:
        wav_file_path (str): Path to the WAV file.
        language (str, optional): Language of the speech. Defaults to "es-ES".


    Returns:
        str: The transcribed text if successful, or None otherwise.

    TODO  RETURN the timespan of the process (how long it takes)

    """
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_AI_SPEECH_SERVICE_KEY, region=AZURE_AI_SPEECH_SERVICE_REGION
    )
    speech_config.speech_recognition_language = language

    audio_config = speechsdk.audio.AudioConfig(filename=wav_file_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    if wav_file_path[-4:] != ".wav":
        raise ValueError("File must be a .wav file")

    # Call the async method without await
    result_future = recognizer.recognize_once_async()

    # Use asyncio to get the result from ResultFuture
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, result_future.get)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
        return None
    return None


async def transcribe_twl_media(
    twl_media_transcribed: str, twl_media_content: dict, twl_msg: dict
):
    """
    It:
    - given a pause (detects the pause in the input media stream)
    - transcribes the Twilio media messages using Azure and
    """

    return await twl_media_transcribed, twl_media_content, False


if __name__ == "__main__":
    path = os.path.abspath(__file__)

    current_dir = os.path.dirname(path)
    print("Current directory", current_dir)
    # Example usage
    convert_g711_ulaw_to_mp3(
        os.path.join(current_dir, "..", "data", "call_example_06_hotel.json"),
        os.path.join(current_dir, "..", "data", "output_a_06_hotel.wav"),
    )
