import torch
from TTS.api import TTS

REFERENCE_WAV = "/Users/d.ibrokhimov/Downloads/voices/vocals.wav"
OUTPUT_PATH = "/Users/d.ibrokhimov/Development/TTS/cloned_output_mushtariy.wav"
TEXT = "Hello! It's me Mushtariy. This is my voice speaking in English. How does it sound to you?"

device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text=TEXT,
    speaker_wav=REFERENCE_WAV,
    language="en",
    file_path=OUTPUT_PATH,
)

print(f"Saved to: {OUTPUT_PATH}")
