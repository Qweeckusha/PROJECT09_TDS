FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04
LABEL authors="qweeck"

RUN apt-get update && apt-get install -y wget libgomp1 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

COPY environment.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f environment.yml && \
    conda clean -afy

ENV LD_LIBRARY_PATH=/opt/conda/envs/tds-env/lib:$LD_LIBRARY_PATH

SHELL ["conda", "run", "-n", "tds-env", "/bin/bash", "-c"]

RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchaudio==2.11.0 \
    torchvision==0.26.0 \
    nvidia-cublas-cu12 \
    --extra-index-url https://download.pytorch.org/whl/cu130

COPY . .

RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token) && \
    python -c "import os; from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=os.environ['HF_TOKEN'])"

ARG WHISPER_MODEL_REPO=Systran/faster-whisper-large-v3
RUN hf download ${WHISPER_MODEL_REPO}

ENV HF_HUB_OFFLINE=1

ENV PYTHONUNBUFFERED=1

EXPOSE 9012

CMD ["conda", "run", "-n", "tds-env", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9012"]