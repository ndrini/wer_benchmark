import os

from asr.asr_base import Transcription, TranscriptionResult
from asr.asr_factory import get_asr_engine, get_available_asr_engines

from . import asr_utils

# list possible asr engines, like ["whispers", "gemini", "speech"]
engines: list = get_available_asr_engines()


AUDIO_DIR = "data"
# create the dataset composed of objects of type Transcription


def get_asr_info() -> list[str]:
    return engines


def create_dataset(item_number: int = 10) -> list[Transcription]:
    """Create a dataset of Transcription objects from the audio files in the AUDIO_DIR directory."""

    dataset = []

    # sample from first source: the short audio files
    df_short = asr_utils.extract_random_short(item_number)

    for idx, row in df_short.iterrows():
        dataset.append(asr_utils.create_transcription_instance_from_short(idx, row))

    # # sample from second source: the long audio files
    # df_short = asr_utils.extract_random_short(10)
    # for idx, row in df_short.iteritems():
    #     print(idx, row)
    #     dataset.append(asr_utils.create_transcription_instance_from_long(idx, row))

    return dataset


def add_transcription(service_name: str, transcript_obj: Transcription):
    """ "populate the transcriptionResults list of the Transcription object with the transcription obtained from the ASR engine"""

    asr = get_asr_engine(service_name)
    transcript_obj.transcriptionResults.append(asr.transcribe(transcript_obj))
    # return TranscriptionResult(service_name, transcription, None, 0.0, None)


def get_wer(
    item_number: int = 2, engines: list[str] = engines
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute the WER for all the services in the transcriptionResults, and return a tuple of dictionaries with the WER and the processing time for each service"""

    # for all the services in the transcriptionResults, compute the WER

    print("Computing WER for all services...")

    dataset = create_dataset(item_number)

    for engine in engines:
        for transcript_obj in dataset:
            add_transcription(engine, transcript_obj)
            # aka
            # add_transcription("azure", transcript_obj)
            # add_transcription("gemini", transcript_obj)
            # add_transcription("azure_lang", transcript_obj)

    wer_result = (
        asr_utils.compute_mean_wer_for_each_service(dataset),
        asr_utils.compute_mean_processing_time_for_each_service(dataset),
    )

    asr_utils.plot_and_store_results_for_each_service(*wer_result)

    return wer_result
