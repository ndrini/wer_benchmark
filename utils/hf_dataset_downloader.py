import os


def download_audio_files():

    # store in data/ds_long the files from 697.mp3 to till 230 files more
    base_url = "https://huggingface.co/datasets/portafolio/llamadas-celular-05/blob/main/train/audio/"
    for i in range(500, 1000):
        url = base_url + str(i) + ".mp3"
        print(url)
        # download the file
        os.system("wget " + url + " -P data/ds_long/")


# https://huggingface.co/datasets/portafolio/llamadas-celular-05/blob/main/test/audio/697.mp3


if __name__ == "__main__":
    download_audio_files()
