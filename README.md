# PROJECT09_TDS (Transcribe-Diarize-Summarize)

> docs-ver: 0.1


## Stack
- Ollama
- faster-whisper
- pyannote.audio
- torch 2.11
- FastAPI

## Scheme
```
                Стандартный Pipeline (StdPl)
+-------------------------------------------------------+                
|Транскрибирование                                      |
|        |                                              |
|        V                                              |       Полный Pipeline (FPl)
| Сплит результатов -> Форматирование полного текста -->|     --> Ollama-server -> LLM
|        ^                                              |
|        |                                              |
|    Диаризация                                         |
+-------------------------------------------------------+
```

## Deploy
Для деплоя можно использовать исходные файлы в этом репозитории.
> Необходимо получить Huggingface-token с принятой лицензией в gated модели. https://huggingface.co/pyannote/speaker-diarization-community-1

```bash

docker build --secret id=hf_token,src=hf_token.txt -t tds-api:conda-env .
```

Далее запуск контейнера.

```bash

docker run --gpus all -p 9012:9012 --name tds-api tds-api:conda-env
```


## Technical requirements

- Версия CUDA - 13.0+