# Models Directory

This directory contains various models used in the test-lightweight-models repository.

## Faster Whisper Models

The repository uses [faster-whisper](https://github.com/guillaumekln/faster-whisper), which is a faster implementation of OpenAI's Whisper model using CTranslate2. The models are optimized for efficient inference.

### Available Models

- **faster-whisper-tiny.en**: A small English-only model, good for quick tests and demos
- **faster-whisper-base.en**: A larger English-only model with better accuracy

### How to Clone Faster Whisper Models

To use the speech-to-text functionality, you need to clone the model files from Hugging Face. Follow these steps:

#### Prerequisites

1. Install Git LFS (Large File Storage) if you haven't already:
```bash
pip install git-lfs
```
   or follow the instructions on the [Git LFS website](https://git-lfs.github.com/).

```

2. Initialize Git LFS:
   ```bash
   git lfs install
   ```

#### Cloning the Models

1. Enter the models directory:
   ```bash
   cd models
   ```

2. Clone the tiny.en model (faster but less accurate):
   ```bash
   git clone https://huggingface.co/guillaumekln/faster-whisper-tiny.en
   cd faster-whisper-tiny.en
   git lfs pull
   cd ..
   ```

3. Another version of faster-whisper (slower but more accurate):
   ```bash
   git clone https://huggingface.co/guillaumekln/faster-whisper-base.en
   cd faster-whisper-base.en
   git lfs pull
   cd ..
   ```

### Using the Models

The models can be used with the faster-whisper library as shown in the example below:

```python
from faster_whisper import WhisperModel

# Load the model
model = WhisperModel("./models/faster-whisper-tiny.en", device="cpu", compute_type="float32")

# Transcribe audio
segments, info = model.transcribe("path/to/audio.mp3")

# Print transcription
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

## Other Models

This directory also contains other models used for different tasks:

- **YOLO Models**: Used for object detection (yolo11n.onnx)
- **OCR Models**: Used for text recognition (PP-OCRv3 models)

If you want to use the OCR Models, ***change the path in configs/config_rapid_ocr.yml*** to the path of the PP-OCRv3 models (both detection and recognition models).
For more information about these models, refer to their respective documentation.