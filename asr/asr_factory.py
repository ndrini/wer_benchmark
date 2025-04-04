# select the ASR service
#  WARNING: ALL the ASR service MUST start with "Asr" string

import inspect

# from .asr_azaispeechbatchtr import AzAISpeechBatchTr
# from .asr_azaispeechrealtimetr import AsrAzAISpeechRealTimeTr
from .asr_azaispeechfasttr import AsrAzAiSpeechFastTr
from .asr_azaispeechfasttrlang import AsrAzAiSpeechFastTrLang
from .asr_azure import AsrAzure
from .asr_azurelang import AsrAzureLang
from .asr_gemini import AsrGemini
from .asr_openai import AsrOpenaiGpt4oTr


def get_asr_engine(engine_name: str, **kwargs):
    """Devuelve una instancia del ASR seleccionado."""
    if engine_name == "azure":
        return AsrAzure()
    elif engine_name == "azurelang":
        return AsrAzureLang()
    elif engine_name == "gemini":
        return AsrGemini()
    # elif engine_name == "azaispeechbatchtr":
    #     return AsrAzAISpeechBatchTr()
    elif engine_name == "azaispeechfasttr":
        return AsrAzAiSpeechFastTr()
    elif engine_name == "azaispeechfasttrlang":
        return AsrAzAiSpeechFastTrLang()
    elif engine_name == "openaigpt4otr":
        return AsrOpenaiGpt4oTr()
    # elif engine_name == "whisper":
    #     return ASRWhisper(model_name=kwargs.get("model_name", "base"))
    else:
        raise ValueError(f"ASR {engine_name} no soportado")


def get_available_asr_engines() -> list[str]:
    """return the list of available ASR engines"""
    engines = []
    for name, obj in globals().items():
        if inspect.isclass(obj) and name.startswith("Asr"):
            engine_name: str = name.lower().replace("asr", "")
            engines.append(engine_name)
    return engines
