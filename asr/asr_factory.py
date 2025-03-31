# select the ASR service
#  WARNING: ALL the ASR service MUST start with "Asr" string

import inspect

from .asr_azure import AsrAzure

# from .asr_whisper import ASRWhisper


def get_asr_engine(engine_name: str, **kwargs):
    """Devuelve una instancia del ASR seleccionado."""
    if engine_name == "azure":
        return AsrAzure(
            subscription_key=kwargs["subscription_key"], region=kwargs["region"]
        )
    # elif engine_name == "whisper":
    #     return ASRWhisper(model_name=kwargs.get("model_name", "base"))
    else:
        raise ValueError(f"ASR {engine_name} no soportado")


def get_available_asr_engines():
    """return the list of available ASR engines"""
    engines = []
    for name, obj in globals().items():
        if inspect.isclass(obj) and name.startswith("Asr"):
            engines.append(name.lower().replace("asr", ""))
    return engines
