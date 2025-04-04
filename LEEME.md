# Guía de instalación y desarrollo para el proyecto **web_benchmark**

Este documento proporciona instrucciones para la instalación y configuración del entorno de desarrollo para el proyecto **web_benchmark**. Asegúrate de seguir cada paso cuidadosamente.


## Ejecución

Ejecuta con venv activado: 

```bash
python -m main <sample_number>
```

El comando de ejecución tiene que seguir el patron:
`python -m main <info|N> [all|engine1,engine2,...]`
con:

- info para conocer los engines disponibles o bien N como integer del numero intero de exemplos a pillar
- all o bien el listado de engine srt a utilizar

Lee en el termina los resultados, ve el gráfico en la carpeta `results` (ordenados por fecha).

## TODO

- he definido el output esperado y las funciones que necesito para calcularlo (dibujarlo)
- ahora: escribir la carga de datos para el dataset
- escribir las interfaces hacia los modelos
- 



## Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes elementos en tu sistema:

- Python 3.12.8 o superior
- `pyenv` para la gestión de versiones de Python
- `uv` para la instalación de paquetes de Python
- Acceso a servicios de Git (como Azure DevOps, GitLab, etc.) configurado con claves SSH.

## Instalación


2. **Crea el entorno virtual**:
   Navega al directorio del proyecto y crea un entorno virtual, Activa el entorno virtual que acabas de crear, instala los paquetes:
```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install .
```

   Si encuentras errores relacionados con la construcción del paquete, verifica que tu archivo `pyproject.toml` esté correctamente configurado y que todos los archivos necesarios estén presentes en el directorio del proyecto.



## dataset

- ds_long
  - https://huggingface.co/datasets/portafolio/llamadas-celular-05/viewer/default/train?p=2&views%5B%5D=train
  - 30 seconds long traces, bad quality  
- ds_short
  - https://www.kaggle.com/datasets/bryanpark/spanish-single-speaker-speech-dataset
  - 6 seconds long traces, good quality








## Los Models (Engines en este proyecto)

Lo modelos de ML que producen la trascripciones se llaman engines in este proyectos. 

Aquí una reseña.


### ??? 
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech?pivots=programming-language-python


### batch transcription 
https://ai.azure.com/explore/models/aiservices/Azure-AI-Speech/version/1/registry/azureml-cogsvc/tryout/all?tid=bf618c51-6c72-4dc7-b52c-2f41381c1899#batch



### Azure cognitive service

AKA AZURE_AI_SPEECH_SERVICE


AZURE_AI_SPEECH_SERVICE_URL = https://westeurope.api.cognitive.microsoft.com/
; AZURE_AI_SPEECH_SERVICE_KEY=9O
AZURE_AI_SPEECH_SERVICE_REGION = westeurope




### Azure-AI-Speech

En la pagina [aiservices/Azure-AI-Speech](https://ai.azure.com/explore/models/aiservices/Azure-AI-Speech/version/1/registry/azureml-cogsvc/tryout?tid=bf618c51-6c72-4dc7-b52c-2f41381c1899#realtime) se encuentra los 3 modelos de nuestro interés:

- Real-time transcription
 - Live transcription capabilities on your own audio without writing any code.
- Fast transcription
 - Leveraging advanced speech recognition technology for rapid analysis and recognition.
 - [official docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create?tabs=locale-specified) 
- Batch transcription
 - Transcribe a large amount of audio in storage and get results asynchronously.

Para credenciales y ejemplos seguir el link "Go to speech playground": allí ver "view code" para accedere a credenciales y ejemplos.


### Gemini Google





## Solución de Problemas

- Si ves el mensaje `__vsc_prompt_cmd_original: command not found`, puede ser un problema con la configuración de tu terminal. Asegúrate de que tu archivo de configuración de la shell (como `~/.bashrc` o `~/.zshrc`) esté correctamente configurado.
- Si encuentras errores al intentar instalar paquetes, revisa el archivo `pyproject.toml` y asegúrate de que todos los archivos necesarios estén en su lugar.

## Estructura del Proyecto

El proyecto tiene la siguiente estructura de directorios:

```
WER_ASR_TTS/
├── asr/
├── tts/
├── scripts/
├── main.py
├── pyproject.toml
└── README.md
```

## Conclusión

Siguiendo estos pasos, deberías poder configurar correctamente el entorno de desarrollo para el proyecto WER_ASR_TTS. Si tienes alguna pregunta o encuentras problemas, no dudes en buscar ayuda en la documentación de Python o en foros de desarrollo.




## desarrollo

Ejecuta los ficheros con  `python -m` ' como módulos.

```python
(.venv) lillo@HP:~/workspace/b_ins/wer_benchmark$ python -m asr.compute_benchmark
```


Ejecuta: 

```bash
(.venv) lillo@HP:~/workspace/b_ins/wer_benchmark$ python -m pytest
=============================================================================== test session starts ===============================================================================
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/lillo/workspace/b_ins/wer_benchmark
configfile: pyproject.toml
collected 2 items                                                                                                                                                                 

tests/test_mean_estraction.py ..                                                                                                                                            [100%]

================================================================================ 2 passed in 0.86s ================================================================================
(.venv) lillo@HP:~/workspace/b_ins/wer_benchmark$ python -m asr.asr_utils
```