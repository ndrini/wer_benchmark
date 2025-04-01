# https://ai.google.dev/gemini-api/docs/audio?lang=python
import datetime
import os

from google import genai

# from google.cloud import speech
from jiwer import wer

from asr.asr_base import AsrBase, Transcription, TranscriptionResult

api_key = os.getenv("GOOGLE_API_KEY", "")

assert api_key[-2:] == "G0", "The Gemini API key is missing. Set it as env variable."

if not api_key:
    raise ValueError("The Gemini API key is missing. Set it as env variable.")


class AsrGemini(AsrBase):
    """ASR engine for Google Gemini.
    Gemini supports the following audio format MIME types:

    - WAV - audio/wav
    - MP3 - audio/mp3
    """

    def __init__(self):
        self.client = genai.Client()

    def transcribe(self, transcript_obj: Transcription) -> TranscriptionResult:
        """Transcribe an audio file using Google Cloud Speech-to-Text. Returns the TranscriptionResult object."""

        start_time = datetime.datetime.now()

        myfile = self.client.files.upload(file=transcript_obj.audio_wav_path)
        # myfile = client.files.upload(file='media/sample.mp3')
        prompt = "Generate a transcript of the speech."

        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt, myfile]
        )

        # response = self.client.models.generate_content(
        #     model="gemini-2.0-flash", contents=["Describe this audio clip", myfile]
        # )

        # response = self.client.recognize(config=config, audio=audio)

        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        print("Gemini transcript: response.text: ", response.text)
        # print("Gemini transcript: response:", response)

        """
        candidates=[Candidate(content=Content(parts=[Part(video_metadata=None, thought=None, code_execution_result=None, executable_code=None, file_data=None, function_call=None, function_response=None, inline_data=None, text='Un fuerte ronquido anunció el aplanamiento de aquel elevado espíritu, conturbado por el vino de la conjuración.')], role='model'), citation_metadata=None, finish_message=None, token_count=None, finish_reason=<FinishReason.STOP: 'STOP'>, avg_logprobs=-0.004714108407497406, grounding_metadata=None, index=None, logprobs_result=None, safety_ratings=None)] create_time=None response_id=None model_version='gemini-2.0-flash' prompt_feedback=None usage_metadata=GenerateContentResponseUsageMetadata(cache_tokens_details=None, cached_content_token_count=None, candidates_token_count=25, candidates_tokens_details=[ModalityTokenCount(modality=<MediaModality.TEXT: 'TEXT'>, token_count=25)], prompt_token_count=157, prompt_tokens_details=[ModalityTokenCount(modality=<MediaModality.AUDIO: 'AUDIO'>, token_count=150), ModalityTokenCount(modality=<MediaModality.TEXT: 'TEXT'>, token_count=7)], thoughts_token_count=None, tool_use_prompt_token_count=None, tool_use_prompt_tokens_details=None, total_token_count=182) automatic_function_calling_history=[] parsed=None
        """

        if response.candidates:
            print("Gemini transcript: response.candidates: ", response.candidates)
            # transcription = response.candidates[0].contents[0].text

            google_transcription = response.text if response.text else ""

            return TranscriptionResult(
                service="google",
                transcription=google_transcription,
                confidence=None,  # Google non fornisce un punteggio di confidenza
                processing_time=processing_time,
                error=False,
                wer=wer(
                    google_transcription, transcript_obj.transcription_ground_truth
                ),
            )
        else:
            return TranscriptionResult(service="google", transcription="", error=True)
