import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# from audio_tools import recognize_speech_from_wav
from datasets import load_dataset
from jiwer import wer

from . import root_path
from .asr_base import Transcription, TranscriptionResult

# from utils import custom_objects

# ds_long = load_dataset("Nexdata/Spanish_Conversational_Speech_Data_by_Telephone")

# from main import root_path


def extract_random_short(files_number: int) -> pd.DataFrame:
    """extract a random sample of files from the short audio files samples"""

    # import in a dataframe the data/ds_short_transcript.txt txt file, fields separated by | pipe
    # transcript = pd.read_csv(
    #     os.path.join(root_path, "data", "ds_short_transcript.txt"), sep="\\|"
    # )

    df_transcript: pd.DataFrame = pd.read_csv(
        os.path.join(root_path, "data", "ds_short_transcript.csv"),
        dtype={
            "file": str,
            "transcript_itn": str,
            "transcript": str,
            "duration": float,
        },
    )

    # create a new dataframe with only the file name and transcript columns
    # df_short_all = transcript[["file", "transcript_itn"]]

    # give types to the columns
    df_transcript["file"] = df_transcript["file"].astype(str)
    df_transcript["transcript_itn"] = df_transcript["transcript_itn"].astype(str)
    df_transcript["transcript"] = df_transcript["transcript"].astype(str)
    df_transcript["duration"] = df_transcript["duration"].astype(float)

    print(
        "*** during extract_random_short function, \n\t\t\ttranscript.head():",
        df_transcript.head(),
    )

    # print(
    #     "*******************************************", df_transcript["transcript_itn"]
    # )
    # randomly sample from the dataframe
    df_short = df_transcript.sample(n=files_number)

    df_short["file"] = df_short["file"].astype(str)
    df_short["transcript_itn"] = df_short["transcript_itn"].astype(str)
    df_short["transcript"] = df_short["transcript"].astype(str)
    df_short["duration"] = df_short["duration"].astype(float)

    print(
        "*** during extract_random_short function, , \n\t\t\tdf_short.head():",
        df_short.head(),
    )

    for sentence in df_short["transcript"]:
        print("*******************************************", sentence)

    return df_short


def extract_random_long(files_number: int) -> pd.DataFrame:
    return pd.DataFrame()


def create_transcription_instance_from_short(
    # idx: int, row: pd.Series[Any]
    idx: int,
    row: pd.Series,
) -> Transcription:
    audio_file_path = os.path.join(root_path, "data", "ds_short", row["file"])

    return Transcription(
        # audio_wav_path=open(audio_file_path, "rb").read(),
        audio_wav_path=audio_file_path,
        transcription_ground_truth=row["transcript"],
        transcription_ground_truth_itn=row["transcript_itn"],
        duration=row["duration"],
    )


def compute_wer(reference: str, hypothesis: str):

    return wer(reference, hypothesis)


def compute_mean_wer_for_each_service(dataset: list[Transcription]) -> dict[str, float]:
    """
    Computa il WER medio per ciascun servizio.
    Gestisce i casi in cui trans_result.wer è None.
    """
    mean_wers: dict[str, list[float]] = {}
    for transcription in dataset:
        for trans_result in transcription.transcriptionResults:
            if trans_result.wer is not None:
                mean_wers.setdefault(trans_result.service, []).append(trans_result.wer)

    # Compute the mean WER for each service, skipping services with no valid WER values.
    result_mean_wers: dict[str, float] = {}
    for service, wers in mean_wers.items():
        if wers:  # Check if the list is not empty
            print("*************** mean_wers", wers)
            result_mean_wers[service] = sum(wers) / len(wers)

    return result_mean_wers


# def compute_mean_wer_for_each_service(dataset: list[Transcription]) -> dict[str, float]:
#     # compute the mean wer for each service
#     mean_wers = {}
#     for transcription in dataset:
#         # for each service
#         for trans_result in transcription.transcriptionResults:
#             # Ensure the service key exists and is a list
#             mean_wers.setdefault(trans_result.service, []).append(trans_result.wer)


#     """ prevent possible missing of wer values in the transcriptionResults list
#         to avoid *************** mean_wers [None, 1.5, 1.1428571428571428]
# Traceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code
#   File "/home/lillo/workspace/b_ins/wer_benchmark/main.py", line 31, in <module>
#     wer_result = compute_asr_benchmark.get_wer(item_number)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/lillo/workspace/b_ins/wer_benchmark/asr/compute_asr_benchmark.py", line 72, in get_wer
#     asr_utils.compute_mean_wer_for_each_service(dataset),
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/lillo/workspace/b_ins/wer_benchmark/asr/asr_utils.py", line 110, in compute_mean_wer_for_each_service
#     mean_wers[service] = sum(mean_wers[service]) / len(mean_wers[service])
#                          ^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

#     """

#     # Compute the mean WER for each service
#     for service in mean_wers:
#         print("*************** mean_wers", mean_wers[service])
#         mean_wers[service] = sum(mean_wers[service]) / len(mean_wers[service])


#     return mean_wers


def compute_mean_processing_time_for_each_service(
    dataset: list[Transcription],
) -> dict[str, float]:
    # compute the mean wer for each service
    mean_process_time = {}
    for transcription in dataset:
        # for each service
        for trans_result in transcription.transcriptionResults:
            # Ensure the service key exists and is a list
            mean_process_time.setdefault(trans_result.service, []).append(
                trans_result.processing_time
            )

    # Compute the mean WER for each service
    for service in mean_process_time:

        print("*************** mean_process_time", mean_process_time[service])
        mean_process_time[service] = sum(mean_process_time[service]) / len(
            mean_process_time[service]
        )

    return mean_process_time


# def display_mean_wer_for_each_service(dataset: list[Transcription]):
#     """plot the mean WER for each service"""
#     mean_wers = compute_mean_wer_for_each_service(dataset)


def plot_and_store_results_for_each_service(
    mean_wers: dict[str, float], mean_process_time: dict[str, float]
):
    """normalize and plot the mean WER and mean processing time for each service"""
    # normalize the mean WER and mean processing time
    # plot the mean WER and mean processing time for each service
    # Create the plot
    max_process_time = max(mean_process_time.values())
    max_wer = max(mean_wers.values())
    # Normalize the values
    normalized_process_time = {
        k: v / max_process_time for k, v in mean_process_time.items()
    }
    normalized_wer = {k: v / max_wer for k, v in mean_wers.items()}
    # Create the plot
    data = {
        "WER": normalized_wer,
        "Processing Time": normalized_process_time,
    }

    # Creiamo un DataFrame
    df = pd.DataFrame(data)

    # Creiamo il grafico a barre
    df.plot(kind="bar", figsize=(10, 6))

    # Aggiungiamo etichette
    plt.xlabel("Categorie")
    plt.ylabel("Valori Normalizzati")
    plt.title("Confronto tra WER e Tempo di Elaborazione")
    plt.legend(title="Metrica")

    # Add some styling
    plt.title("Mean Word Error Rate (WER) and Mean Processing Time by Service")
    plt.xlabel("Service")
    plt.ylabel("Normalized Mean (0-1)")
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.legend()  # Show legend
    plt.tight_layout()  # Adjust layout to make room for rotated labels

    results_dir = Path(root_path) / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Creazione di un percorso per un file
    filename = f"mean_services_{datetime.today()}.png"
    filepath = results_dir / filename

    # Salva il grafico come immagine
    plt.savefig(filepath)


def plot_mean_wer_for_each_service(mean_wers: dict[str, float]):
    """plot the mean WER for each service"""

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set an appropriate figure size
    plt.bar(mean_wers.keys(), mean_wers.values())

    # Add some styling
    plt.title("Mean Word Error Rate (WER) by Service")
    plt.xlabel("Service")
    plt.ylabel("Mean WER")
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()  # Adjust layout to make room for rotated labels

    print("********************************", root_path)
    # Create results directory if it doesn't exist
    results_dir = os.path.join(root_path, "results")
    print("********************************", results_dir)
    # results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with current date
    filename = f"mean_wer_{datetime.now().strftime('%Y-%m-%d')}.png"
    filepath = os.path.join(results_dir, filename)
    print("********************************", filepath)

    # Save the figure first
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


# def run_asr(files_number):

#     # select 30 short traces and 6 long traces
#     df_short = extract_random(files_number)

#     df_results = pandas.DataFrame(
#         columns=[
#             "idx",
#             "transcript_asr",
#             "transcript_ground_truth",
#             "time_spent_transcribing",
#         ]
#     )
#     # use tqdm to run in parallel the speach recognition, return also the speed of the process
#     for idx, row in df_short.iterrows():
#         audio_file_path = os.path.join(root_path, "data", "ds_short", row["file"])
#         extracted_text = recognize_speech_from_wav(audio_file_path)

#         # add wer columns to the df_results
#         df_results["idx"] = df_short["idx"]
#         df_short.at[idx, "transcript_asr"] = extracted_text
#         df_results["time_spent_transcribing"] = (
#             0.0  # TODO add time spent transcribing here
#         )
#         df_results["transcript_ground_truth"] = df_short["transcript_itn"]
#         print("**********", row["transcript_ground_truth"], extracted_text)

#     # compute wer
#     df_results["wer"] = df_results.apply(
#         lambda row: compute_wer(row["transcript_ground_truth"], row["transcript_asr"]),
#         axis=1,
#     )

#     # print wer
#     print(df_results.head())

#     return df_results


#     pass

if __name__ == "__main__":
    # extract_random(2)
    # plot_mean_wer_for_each_service({"whispers": 0.1, "gemini": 0.2, "speech": 0.15})

    plot_and_store_results_for_each_service(
        {"whispers": 0.1, "gemini": 0.2, "speech": 0.15},
        {"whispers": 580, "gemini": 874, "speech": 511},
    )
