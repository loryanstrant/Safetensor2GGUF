# Safetensor2GGUF

A Dockerized web application that converts Hugging Face PEFT **LoRA adapters**
(`.safetensors`) to **GGUF** format for use with
[LocalAI](https://localai.io/) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

Built with **FastAPI**, powered by the upstream
[`convert_lora_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
script from **llama.cpp**, and packaged in an **NVIDIA CUDA 12.8** container
with first-class support for **Blackwell GPUs** (B100 / B200 / GB200) as well
as all older NVIDIA architectures.

---

## Features

| Feature | Details |
|---|---|
| **Web UI** | Point-and-click interface — browse for files, pick output type, hit Convert. |
| **File browser** | Built-in modal file picker navigates the mounted `/models` volume. |
| **Output types** | `f32`, `f16`, `bf16`, `q8_0`, `auto` — same options as the upstream script. |
| **Async conversion** | Runs in the background; logs stream to the browser in real time. |
| **Blackwell GPU support** | CUDA 12.8 base image; works on B100, B200, GB200 and older GPUs. |
| **Model support** | Any architecture supported by llama.cpp, including **Qwen 3.5-4B**. |

---

## Quick Start

### Prerequisites

* **Docker** ≥ 24 and **Docker Compose** ≥ 2
* **NVIDIA Container Toolkit** (for GPU passthrough)  
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

### 1. Clone the repository

```bash
git clone https://github.com/loryanstrant/Safetensor2GGUF.git
cd Safetensor2GGUF
```

### 2. Place your models

Create a `models/` directory (or point the volume elsewhere in
`docker-compose.yml`):

```
models/
├── Qwen3.5-4B-Q4_K_M.gguf      # base model (GGUF or HF directory)
└── personas/
    └── Gary/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

### 3. Build and run

```bash
docker compose up --build -d
```

The web UI is available at **<http://localhost:7860>**.

### 4. Convert

1. Open the browser at `http://localhost:7860`.
2. Set **LoRA Adapter Directory** to `/models/personas/Gary/`.
3. Set **Base Model Path** to `/models/Qwen3.5-4B-Q4_K_M.gguf` (or leave
   empty to auto-detect from the adapter config).
4. Choose **Output Type** — e.g. `q8_0`.
5. Click **Convert** and watch the logs stream in.

The resulting `.gguf` file will be written into the LoRA adapter directory
(or to the path you specified in **Output File**).

---

## Docker Compose Reference

```yaml
services:
  lora-converter:
    build: .
    container_name: lora-to-gguf
    ports:
      - "7860:7860"
    volumes:
      - ./models:/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

> **Tip:** If you do not have an NVIDIA GPU you can remove the `deploy` →
> `resources` block and conversions will run on the CPU (slower but functional).

---

## CLI Equivalent

The web UI wraps the same command you would run manually:

```bash
python3 /opt/llama.cpp/convert_lora_to_gguf.py \
  --base /models/Qwen3.5-4B-Q4_K_M.gguf \
  --outtype q8_0 \
  /models/personas/Gary/
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONVERT_SCRIPT_PATH` | `/opt/llama.cpp/convert_lora_to_gguf.py` | Override if the script lives elsewhere. |

---

## Building Without GPU

If you only need CPU-based conversion, swap the base image in the
`Dockerfile`:

```dockerfile
FROM python:3.12-slim
```

and remove the GPU reservations from `docker-compose.yml`.

---

## License

[MIT](LICENSE)
