import time
import re
from faster_whisper import WhisperModel
from pathlib import Path
from difflib import SequenceMatcher


def transcribe_audio(audio_file: str, model_path: str, device: str = "cpu",
                     compute_type: str = "float32") -> str:
    start_time = time.time()

    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_file)

    text = '\n'.join(segment.text for segment in segments)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return text


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    audio_path = base_dir / 'media' / 'eng_convo_30s.mp3'

    # Cần cd models -> git clone https://huggingface.co/guillaumekln/faster-whisper-tiny.en
    # sau đó cd thư mục tiny.en, git lfs pull để kéo model.bin về
    model_path = base_dir / 'models' / 'faster-whisper-base.en'

    transcription = transcribe_audio(
        str(audio_path),
        model_path=str(model_path),
        compute_type="float32"
    )

    print("Predict: ")
    print(transcription)


