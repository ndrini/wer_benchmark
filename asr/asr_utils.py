import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import pandas

# from audio_tools import recognize_speech_from_wav
from datasets import load_dataset
from jiwer import wer

from utils import custom_objects

ds_long = load_dataset("Nexdata/Spanish_Conversational_Speech_Data_by_Telephone")

# from main import root_path

from . import root_path
from .asr_base import Transcription, TranscriptionResult


def extract_random_short(files_number) -> pandas.DataFrame:

    # import in a dataframe the data/ds_short_transcript.txt txt file, fields separated by | pipe
    transcript = pandas.read_csv(
        os.path.join(root_path, "data", "ds_short_transcript.txt"), sep="\\|"
    )

    # create a new dataframe with only the file name and transcript columns
    df_short_all = transcript[["file", "transcript_itn"]]
    print(df_short_all.head())

    # randomly sample from the dataframe
    df_short = df_short_all.sample(n=files_number)

    print(df_short.head())
    return df_short


def extract_random_long(files_number) -> pandas.DataFrame:
    pass


def create_transcription_instance_from_short(idx, row: pandas.Series):
    audio_file_path = os.path.join(root_path, "data", "ds_short", row["file"])

    return Transcription(
        audio=open(audio_file_path, "rb").read(),
        transcription_ground_truth=row["transcript"],
        transcription_ground_truth_itn=row["transcript_itn"],
        duration=row["duration"],
    )


def compute_wer(reference, hypothesis):

    return wer(reference, hypothesis)


def compute_mean_wer_for_each_service(dataset: list[Transcription]) -> dict:
    # compute the mean wer for each service
    mean_wers = {}
    # for transcription in dataset:
    #     # for each service
    #     for trans_result in transcription.transcriptionResults:
    #         if trans_result.service not in mean_wer:
    #             mean_wer[trans_result.service] = []
    #         mean_wer[trans_result.service].append(trans_result.wer)
    #         mean_wer[trans_result.service] = sum(mean_wer[trans_result.service]) / len(
    #             mean_wer[trans_result.service]
    #         )
    # return mean_wer

    for transcription in dataset:
        # for each service
        for trans_result in transcription.transcriptionResults:
            # Ensure the service key exists and is a list
            mean_wers.setdefault(trans_result.service, []).append(trans_result.wer)

    # Compute the mean WER for each service
    for service in mean_wers:
        mean_wers[service] = sum(mean_wers[service]) / len(mean_wers[service])

    return mean_wers


# def display_mean_wer_for_each_service(dataset: list[Transcription]):
#     """plot the mean WER for each service"""
#     mean_wers = compute_mean_wer_for_each_service(dataset)


def display_mean_wer_for_each_service(mean_wers: dict):
    """plot the mean WER for each service"""

    # plt.bar(mean_wers.keys(), mean_wers.values())

    # # display the image
    # plt.show()

    # # store the plot into output folder with the date of the creation
    # plt.savefig(
    #     f"{root_path}/results/mean_wer_{datetime.datetime.now().strftime('%Y-%m-%d')}.png"
    # )

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
    display_mean_wer_for_each_service(
        {"Service1": 0.1, "Service2": 0.2, "Service3": 0.15}
    )
