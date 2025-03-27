# select the ASR service
#
from asr_azure import AsrAzure
from asr_whisper import ASRWhisper


def get_asr_engine(engine_name: str, **kwargs):
    """Devuelve una instancia del ASR seleccionado."""
    if engine_name == "azure":
        return AsrAzure(
            subscription_key=kwargs["subscription_key"], region=kwargs["region"]
        )
    elif engine_name == "whisper":
        return ASRWhisper(model_name=kwargs.get("model_name", "base"))
    else:
        raise ValueError(f"ASR {engine_name} no soportado")
