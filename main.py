import os
import pathlib

import azure.cognitiveservices.speech as speechsdk
import ffmpeg

from asr import compute_asr_benchmark

# # get abs root path
# root_path = pathlib.Path(__file__).parent.resolve()

# print(root_path)


def main():
    print("Starting WER benchmark creation...")
    # Add your code here to create the benchmark
    pass


if __name__ == "__main__":
    #  main()

    # read input from the command line
    args = os.sys.argv[1:]
    if len(args) > 0:
        item_number = int(args[0])
    else:
        item_number = 2

    wer_result = compute_asr_benchmark.get_wer(item_number)
    print(wer_result)
