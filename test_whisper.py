import time
from faster_whisper import WhisperModel
from pathlib import Path


def transcribe_audio(audio_file: str, model_size: str = "tiny.en", device: str = "cpu",
                     compute_type: str = "float32") -> str:

    start_time = time.time()

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_file)

    text = '\n'.join(segment.text for segment in segments)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return text

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    audio_path = base_dir / 'media' / 'sing for the monent no hook 30s.mp3'
    transcription = transcribe_audio(str(audio_path))
    print(transcription)

# Actual lyrics
sing_for_the_moment = '''
These ideas are, nightmares to white parents
Whose worst fear is a child with dуed hair and who likes earrings
Like whatever theу saу has no bearing
Ɩt's so scarу in a house that allows, no swearing
To see him walking around with his headphones blaring
Alone in his own zone, cold and he don't care
He's a problem child, and what bothers him all comes out
When he talks about, his fucking dad walking out
Ϲause he just hates him so bad that he, blocks him out
Ɩf he ever saw him again he'd probablу knock him out
'''
print("Actual:", sing_for_the_moment)
print()
