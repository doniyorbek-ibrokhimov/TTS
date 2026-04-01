import os
import torch
from TTS.api import TTS

REFERENCE_WAV = "/Users/d.ibrokhimov/Downloads/voices/vocals.wav"
OUTPUT_DIR = "/Users/d.ibrokhimov/Development/TTS/outputs"
TEXT = "Hello! It's me Mushtariy. This is my voice speaking in English. How does it sound to you?"


def next_output_path(reference_wav: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(reference_wav))[0]
    n = 1
    while True:
        path = os.path.join(output_dir, f"{prefix}_{n:03d}.wav")
        if not os.path.exists(path):
            return path
        n += 1


OUTPUT_PATH = next_output_path(REFERENCE_WAV, OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")
print(f"Output: {OUTPUT_PATH}")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text=TEXT,
    speaker_wav=REFERENCE_WAV,
    language="en",
    file_path=OUTPUT_PATH,
)

print(f"Saved to: {OUTPUT_PATH}")
