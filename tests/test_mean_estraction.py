from asr.asr_base import Transcription, TranscriptionResult
from asr.asr_utils import compute_mean_wer_for_each_service

# populate with mock information for testing a Transcription object

transcription_0 = Transcription(
    audio=b"...",  # binary data for the audio file
    audio_mp3=b"...",  # binary data for the mp3 audio file
    duration=120.0,  # duration of the audio in seconds
    transcription_ground_truth="This is the ground truth transcription.",
    transcription_ground_truth_itn="This is the ground truth transcription with ITN.",
    transcriptionResults=[
        TranscriptionResult(
            service="Service1",
            transcription="This is the transcription from Service1.",
            transcription_itn="This is the transcription from Service1 with ITN.",
            confidence=0.95,
            processing_time=1.2,
            error=False,
            wer=0.1,
        ),
        TranscriptionResult(
            service="Service2",
            transcription="This is the transcription from Service2.",
            transcription_itn="This is the transcription from Service2 with ITN.",
            confidence=0.90,
            processing_time=0.8,
            error=False,
            wer=0.05,
        ),
    ],
)

transcription_1 = Transcription(
    audio=b"...",  # binary data for the audio file
    audio_mp3=b"...",  # binary data for the mp3 audio file
    duration=120.0,  # duration of the audio in seconds
    transcription_ground_truth="This is the ground truth transcription.",
    transcription_ground_truth_itn="This is the ground truth transcription with ITN.",
    transcriptionResults=[
        TranscriptionResult(
            service="Service1",
            transcription="This is the transcription from Service1.",
            transcription_itn="This is the transcription from Service1 with ITN.",
            confidence=0.95,
            processing_time=1.2,
            error=False,
            wer=0.3,
        ),
        TranscriptionResult(
            service="Service2",
            transcription="This is the transcription from Service2.",
            transcription_itn="This is the transcription from Service2 with ITN.",
            confidence=0.90,
            processing_time=0.8,
            error=False,
            wer=0.15,
        ),
    ],
)

dataset = [transcription_0, transcription_1]


def test_compute_mean_wer_for_each_service():
    mean_wer = compute_mean_wer_for_each_service(dataset)
    assert mean_wer == {"Service1": 0.2, "Service2": 0.1}


def test_display_mean_wer_for_each_service():
    compute_mean_wer_for_each_service(dataset)
    # assert the plot is displayed
