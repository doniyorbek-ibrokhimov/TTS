import json
import os
import torch
from TTS.api import TTS

REFERENCE_WAV = "/Users/d.ibrokhimov/Downloads/voices/vocals.wav"
OUTPUT_DIR = "/Users/d.ibrokhimov/Development/TTS/outputs"
TEXT = "Hello! It's me Mushtariy. This is my voice speaking in English. I feel like I will be great at it. How does it sound to you?"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"


def next_output_path(reference_wav: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(reference_wav))[0]
    n = 1
    while True:
        path = os.path.join(output_dir, f"{prefix}_{n:03d}.wav")
        if not os.path.exists(path):
            return path
        n += 1


def save_args(output_wav: str, args: dict) -> None:
    args_path = os.path.splitext(output_wav)[0] + ".json"
    with open(args_path, "w") as f:
        json.dump(args, f, indent=2)
    print(f"Args saved to: {args_path}")


OUTPUT_PATH = next_output_path(REFERENCE_WAV, OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")
print(f"Output: {OUTPUT_PATH}")

tts = TTS(MODEL_NAME).to(device)

tts.tts_to_file(
    text=TEXT,
    speaker_wav=REFERENCE_WAV,
    language=LANGUAGE,
    file_path=OUTPUT_PATH,
)

save_args(OUTPUT_PATH, {
    "model": MODEL_NAME,
    "text": TEXT,
    "speaker_wav": REFERENCE_WAV,
    "language": LANGUAGE,
    "device": device,
})

print(f"Saved to: {OUTPUT_PATH}")
