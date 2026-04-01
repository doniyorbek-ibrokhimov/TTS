import json
import os
import soundfile as sf
import torch
from TTS.api import TTS

REFERENCE_WAV = "/Users/d.ibrokhimov/Downloads/voices/vocals.wav"
OUTPUT_DIR = "/Users/d.ibrokhimov/Development/TTS/outputs"
TEXT = "Hello! I'm a voice clone of Doniyor's friend who's currently learning English. I can not speak English very well yet, but I feel like I will be great at it. How does it sound to you?"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"

# --- Tunable inference params ---
# temperature: lower = more faithful to reference voice, higher = more expressive/varied
TEMPERATURE = 0.3
# length_penalty: higher = shorter outputs
LENGTH_PENALTY = 1.0
# repetition_penalty: higher = reduces stuttering/loops
REPETITION_PENALTY = 5.0
# top_k / top_p: nucleus sampling controls
TOP_K = 50
TOP_P = 0.85
# gpt_cond_len: seconds of reference audio to condition GPT on (max 30)
GPT_COND_LEN = 30
# gpt_cond_chunk_len: chunk size for conditioning (must be <= gpt_cond_len)
GPT_COND_CHUNK_LEN = 6
# max_ref_len: seconds of reference fed to the decoder
MAX_REF_LEN = 30
# sound_norm_refs: normalize reference audio volume before conditioning
SOUND_NORM_REFS = True


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
model = tts.synthesizer.tts_model

out = model.full_inference(
    text=TEXT,
    ref_audio_path=REFERENCE_WAV,
    language=LANGUAGE,
    temperature=TEMPERATURE,
    length_penalty=LENGTH_PENALTY,
    repetition_penalty=REPETITION_PENALTY,
    top_k=TOP_K,
    top_p=TOP_P,
    gpt_cond_len=GPT_COND_LEN,
    gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
    max_ref_len=MAX_REF_LEN,
    sound_norm_refs=SOUND_NORM_REFS,
)

sf.write(OUTPUT_PATH, out["wav"], 24000)

save_args(OUTPUT_PATH, {
    "model": MODEL_NAME,
    "text": TEXT,
    "speaker_wav": REFERENCE_WAV,
    "language": LANGUAGE,
    "device": device,
    "temperature": TEMPERATURE,
    "length_penalty": LENGTH_PENALTY,
    "repetition_penalty": REPETITION_PENALTY,
    "top_k": TOP_K,
    "top_p": TOP_P,
    "gpt_cond_len": GPT_COND_LEN,
    "gpt_cond_chunk_len": GPT_COND_CHUNK_LEN,
    "max_ref_len": MAX_REF_LEN,
    "sound_norm_refs": SOUND_NORM_REFS,
})

print(f"Saved to: {OUTPUT_PATH}")
