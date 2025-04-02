import os
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable

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
    idx: Hashable,
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
    Computes the average WER for each service.
    Handles cases where trans_result.wer is None.
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


def plot_and_store_results_for_each_service(
    mean_wers: dict[str, float], mean_process_time: dict[str, float]
):
    """normalize and plot the mean WER and mean processing time for each service"""
    # Normalize the values
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

    # create un DataFrame
    df = pd.DataFrame(data)

    # create il grafico a barre
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


def store_results_in_csv(
    wer_result: tuple[dict[str, float], dict[str, float]], item_number: int
):
    """store the results in a file"""

    # read the CSV file with previous results and if not present, create it
    filepath = os.path.join(root_path, "results", "wer_all_time_results.csv")
    if os.path.exists(filepath):
        df_all_time_results = pd.read_csv(filepath)

        # update columns if necessary
        for col in wer_result[0].keys():
            if col not in df_all_time_results.columns:
                df_all_time_results[col] = None
    else:
        # Create a DataFrame from the results (first time or after cancellation)
        df_all_time_results = pd.DataFrame(
            columns=["item_number"]
            # "item_number": wer_result[0].keys(),
            # "WER": wer_result[0].values(),
            # "Processing Time": wer_result[1].values(),
        )
        # Add the new columns
        for col in wer_result[0].keys():
            df_all_time_results["WER_" + col] = None
            df_all_time_results["time_" + col] = None
        # set types, set all columns starting with "WER_" to float
        for col in df_all_time_results.columns:
            if col.startswith("WER_") or col.startswith("time_"):
                df_all_time_results[col] = df_all_time_results[col].astype(float)
        df_all_time_results["item_number"] = df_all_time_results["item_number"].astype(
            int
        )
        print("****new df ***** ", df_all_time_results.head())

    # Add values to the DataFrame
    # add a new line in the DataFrame with the results
    # Create a new row to append
    new_row = {"item": item_number}
    for col, value in wer_result[0].items():
        new_row["WER_" + str(col)] = value
    for col, value in wer_result[1].items():
        new_row["time_" + str(col)] = value

    # Append the new row to the DataFrame
    df_all_time_results = pd.concat(
        [df_all_time_results, pd.DataFrame([new_row])], ignore_index=True
    )

    print("**** populated df_all_time_results ***** ", df_all_time_results.head())

    # Save the updated DataFrame to CSV
    df_all_time_results.to_csv(filepath, index=False)

    # for col in wer_result[0].keys():
    #     df_all_time_results.at[item_number, "WER_" + col] = wer_result[0][col]
    # for col in wer_result[1].keys():
    #     df_all_time_results.at[item_number, "time_" + col] = wer_result[1][col]

    # # Save to CSV
    # filename = f"wer_results_{item_number}_{datetime.today()}.csv"
    # filepath = os.path.join(root_path, "results", filename)
    # df_all_time_results.to_csv(filepath, index=False)


if __name__ == "__main__":

    plot_and_store_results_for_each_service(
        {"whispers": 0.1, "gemini": 0.2, "speech": 0.15},
        {"whispers": 580, "gemini": 874, "speech": 511},
    )
