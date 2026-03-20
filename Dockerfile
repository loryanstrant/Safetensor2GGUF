# ── Stage: runtime ──────────────────────────────────────────────────────────
# CUDA 12.8 supports NVIDIA Blackwell (B100/B200/GB200) and all older GPUs.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

LABEL maintainer="loryanstrant"
LABEL description="Web UI for converting LoRA adapters to GGUF format"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System packages ─────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# ── llama.cpp conversion scripts ────────────────────────────────────────────
# We only need the converter scripts and the gguf-py helper package.
RUN git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    git sparse-checkout set --skip-checks convert_lora_to_gguf.py convert_hf_to_gguf.py gguf-py && \
    git checkout && \
    pip3 install --no-cache-dir --break-system-packages /opt/llama.cpp/gguf-py

# ── Application code ────────────────────────────────────────────────────────
COPY app/ /app/

# ── Default volume mount point for models ───────────────────────────────────
VOLUME ["/models"]

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
