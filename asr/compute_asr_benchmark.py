import os

import asr_utils

from asr.asr_base import Transcription, TranscriptionResult
from asr.asr_factory import get_asr_engine

# list possible asr engines
engines = ["whispers", "gemini", "speech"]

print("Possible ASR engines:", get_asr_engine.__name__)

AUDIO_DIR = "data"
# create the dataset composed of objects of type Transcription


def create_dataset() -> list[Transcription]:

    dataset = []

    # sample from first source: the short audio files
    df_short = asr_utils.extract_random_short(10)

    for idx, row in df_short.iteritems():
        print(idx, row)
        dataset.append(asr_utils.create_transcription_instance_from_short(idx, row))

    # # sample from second source: the long audio files
    # df_short = asr_utils.extract_random_short(10)
    # for idx, row in df_short.iteritems():
    #     print(idx, row)
    #     dataset.append(asr_utils.create_transcription_instance_from_long(idx, row))

    return dataset


def add_transcription(service_name, transcript_obj: Transcription) -> Transcription:
    """ "populate the transcriptionResults list of the Transcription object with the transcription obtained from the ASR engine"""

    asr = get_asr_engine(service_name)
    transcript_obj.transcriptionResults.append(asr.transcribe(transcript_obj.audio))
    # return TranscriptionResult(service_name, transcription, None, 0.0, None)


def get_wer():
    # for all the services in the transcriptionResults, compute the WER
    pass


if __name__ == "__main__":
    dataset = create_dataset()
    for transcript_obj in dataset:
        add_transcription("whisper", transcript_obj)

        # add further services here
        print(transcript_obj)

    # analyze and plot the results
    pass


# # Configurar el motor ASR (Azure o Whisper)
# asr = get_asr_engine("whisper")  # Cambia a "azure" si quieres usar Azure

# # Directorio con audios
# AUDIO_DIR = "data/ds_short"


# # select 10 short audio files and 2 large audio files
# audio_files = os.listdir(AUDIO_DIR)
# audio_files = [f for f in audio_files if f.endswith(".wav") or f.endswith(".mp3")]
# audio_files = audio_files[:10] + audio_files[-2:]


# # Procesar cada archivo de audio en el directorio
# for audio_file in os.listdir(AUDIO_DIR):
#     if audio_file.endswith(".wav") or audio_file.endswith(".mp3"):

#         audio_path = os.path.join(AUDIO_DIR, audio_file)
#         transcript = asr.transcribe(audio_path)
#         print(f"{audio_file}: {transcript}")
