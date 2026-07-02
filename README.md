# PROJECT09_TDS (Transcribe-Diarize-Summarize)

> docs-ver: 1.0

Система работает изолированно от сети Интернет и не обращается в сеть, например, на huggingface за токеном. 

## Окружение
- Ollama
- faster-whisper
- pyannote.audio
- torch 2.11
- FastAPI

## Схема
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

## Разворачивание
Для деплоя можно использовать исходные файлы в этом репозитории.
> [!IMPORTANT]
> Необходимо получить Huggingface-token с принятой лицензией в gated модели. https://huggingface.co/pyannote/speaker-diarization-community-1

```bash

docker build --secret id=hf_token,src=hf_token.txt -t tds-api:conda-env .
```

Далее запуск контейнера.
Важно передать OLLAMA_HOST для работы полного пайплайна.
Также нужно указать наименование модели на сервере Ollama, которую вы хотите использовать - OLLAMA_MODEL

> [!NOTE]
> По умолчанию в Docker версии Linux может быть отключено специальное DNS-имя.
> 
> Тогда нужно добавит флаг `--add-host host.docker.internal:host-gateway` или указать gateway docker-сети `172.17.0.1` вместо DNS

```bash

docker run --gpus all -p 9012:9012 -e OLLAMA_HOST=http://host.docker.internal:11434 -e OLLAMA_HOST="qwen3.5:4b" --name tds-api tds-api:conda-env
```


## Технические зависимости

- Версия CUDA - 13.0+

Для проверки версии:
```bash

nvidia-smi
```

> [!WARNING]
> Настоятельно рекомендуется использовать систему с поддержкой CUDA


## Полезные ссылки

1. [Документация по API](docs/api.md)
2. [Модуль транскрибирования](docs/transcriber.md)
3. [Модуль диаризации](docs/diarizer.md)
4. [Модуль суммаризации](docs/summarizer.md)