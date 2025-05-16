import time
import os
from faster_whisper import WhisperModel
from pathlib import Path
import gc


def transcribe_audio_optimized(audio_file, model_path, compute_type="int8"):

    start_total = time.time()
    load_start = time.time()

    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=os.cpu_count(),  # Sử dụng tất cả thread có sẵn
        download_root=os.path.join(os.path.dirname(model_path), "cache")  # Cache model
    )

    load_time = time.time() - load_start
    print(f"Đã tải model trong {load_time:.2f} giây")

    transcribe_start = time.time()
    segments, info = model.transcribe(
        audio_file,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=False,
        suppress_blank=True,
        max_initial_timestamp=0.0
    )

    result = ""
    for segment in segments:
        result += segment.text.strip() + " "

    transcribe_time = time.time() - transcribe_start
    total_time = time.time() - start_total

    del model
    gc.collect()

    print(f"Thời gian xử lý transcribe: {transcribe_time:.2f} giây")
    print(f"Độ dài audio: {info.duration:.2f} giây")
    print(f"Thời gian tổng (tải model + xử lý): {total_time:.2f} giây")
    print(f"Overhead tải model: {load_time:.2f}s ({load_time / total_time * 100:.1f}% tổng thời gian)")

    return result.strip()


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent
    audio_path = base_dir / 'media' / 'eng_convo_30s.mp3'
    model_path = base_dir / 'models' / 'faster-whisper-tiny.en'

    print("\n--- Bắt đầu xử lý file ---")
    transcription = transcribe_audio_optimized(
        str(audio_path),
        str(model_path),
        compute_type="int8"
    )

    print("\nKết quả transcribe: ")
    print(transcription)
