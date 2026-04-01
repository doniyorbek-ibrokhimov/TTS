"""
XTTS v2 Fine-Tuning Script
==========================
3-phase pipeline:
  Phase 1 — Prepare dataset: transcribe audio with Whisper, split into sentences, write CSVs
  Phase 2 — Fine-tune: train the GPT encoder of XTTS v2 on your speaker data
  Phase 3 — Inference: generate a test clip using the fine-tuned model

Usage:
  # Run all phases end-to-end:
  python finetune.py

  # Run only a specific phase (useful for resuming):
  python finetune.py --phase 1   # dataset prep only
  python finetune.py --phase 2   # training only (requires Phase 1 output)
  python finetune.py --phase 3   # inference only (requires Phase 2 output)

Audio requirements for best results:
  - Minimum 2 minutes total, ideally 5–30 minutes
  - Clean speech, no background music or heavy reverb
  - wav / mp3 / flac accepted — will be resampled automatically
"""

import argparse
import gc
import glob
import json
import os

import soundfile as sf
import torch

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Directory containing your raw speaker audio files (wav/mp3/flac)
AUDIO_INPUT_DIR = "/Users/d.ibrokhimov/Downloads/voices"

# Where all outputs (dataset, checkpoints, test audio) will be written
OUTPUT_BASE_DIR = "/Users/d.ibrokhimov/Development/TTS/finetune_output"

# Speaker language code
LANGUAGE = "en"

# Speaker name — used as label in metadata CSVs
SPEAKER_NAME = "mushtariy"

# Text for the Phase 3 inference test
TEST_TEXT = "Hello! This is my fine-tuned voice speaking in English. How does it sound now?"

# ─── TRAINING HYPERPARAMS ─────────────────────────────────────────────────────
# On CPU/MPS (Apple Silicon) keep BATCH_SIZE=2 and raise GRAD_ACCUMULATION so
# effective batch = BATCH_SIZE * GRAD_ACCUMULATION >= 252 (recommended by Coqui).
# GPU with ≥16 GB VRAM can use BATCH_SIZE=4, GRAD_ACCUMULATION=63.
EPOCHS = 6
BATCH_SIZE = 2
GRAD_ACCUMULATION = 126   # 2 * 126 = 252 effective batch

# ─── INFERENCE PARAMS (Phase 3) ───────────────────────────────────────────────
TEMPERATURE = 0.3
REPETITION_PENALTY = 5.0
TOP_K = 50
TOP_P = 0.85
GPT_COND_LEN = 30
MAX_REF_LEN = 30
SOUND_NORM_REFS = True

# ─── DERIVED PATHS ────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(OUTPUT_BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(OUTPUT_BASE_DIR, "run", "training")
TRAIN_CSV = os.path.join(DATASET_DIR, "metadata_train.csv")
EVAL_CSV = os.path.join(DATASET_DIR, "metadata_eval.csv")
TEST_OUTPUT = os.path.join(OUTPUT_BASE_DIR, "test_output.wav")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Dataset preparation
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_prepare_dataset():
    print("\n" + "═" * 60)
    print("PHASE 1 — Preparing dataset")
    print("═" * 60)

    audio_types = (".wav", ".mp3", ".flac")
    audio_files = [
        p for p in glob.glob(os.path.join(AUDIO_INPUT_DIR, "**", "*"), recursive=True)
        if p.lower().endswith(audio_types)
    ]

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in {AUDIO_INPUT_DIR}\n"
            "Place your speaker WAV/MP3/FLAC files there and re-run."
        )

    print(f"Found {len(audio_files)} audio file(s): {[os.path.basename(f) for f in audio_files]}")

    # faster-whisper uses int8 on CPU (float16 is CUDA-only)
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from faster_whisper import WhisperModel
    import pandas
    import torchaudio
    from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

    print(f"Loading Whisper large-v2 on {device} ({compute_type}) …")
    asr_model = WhisperModel("large-v2", device=device, compute_type=compute_type)

    os.makedirs(DATASET_DIR, exist_ok=True)
    wavs_dir = os.path.join(DATASET_DIR, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    audio_total_seconds = 0
    BUFFER = 0.2  # seconds of padding around each sentence boundary

    for audio_path in audio_files:
        print(f"\nTranscribing: {os.path.basename(audio_path)}")
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_seconds += wav.size(-1) / sr

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=LANGUAGE)
        words_list = [w for seg in segments for w in list(seg.words)]

        sentence = ""
        sentence_start = None
        first_word = True
        clip_idx = 0
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        for word_idx, word in enumerate(words_list):
            if first_word:
                if word_idx == 0:
                    sentence_start = max(word.start - BUFFER, 0)
                else:
                    prev_end = words_list[word_idx - 1].end
                    sentence_start = max(word.start - BUFFER, (prev_end + word.start) / 2)
                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence.strip()
                sentence = multilingual_cleaners(sentence, LANGUAGE)

                if word_idx + 1 < len(words_list):
                    next_start = words_list[word_idx + 1].start
                else:
                    next_start = (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_start) / 2, word.end + BUFFER)

                clip_name = f"wavs/{base_name}_{str(clip_idx).zfill(8)}.wav"
                clip_path = os.path.join(DATASET_DIR, clip_name)
                clip_audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)

                # skip clips shorter than 0.33s
                if clip_audio.size(-1) >= sr / 3:
                    torchaudio.save(clip_path, clip_audio, sr)
                    metadata["audio_file"].append(clip_name)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(SPEAKER_NAME)
                    clip_idx += 1

                first_word = True

    print(f"\nTotal audio processed: {audio_total_seconds:.1f}s ({audio_total_seconds/60:.1f} min)")
    print(f"Sentence clips extracted: {len(metadata['audio_file'])}")

    if audio_total_seconds < 120:
        print("⚠️  WARNING: Less than 2 minutes of audio — quality may be poor. "
              "Aim for 5–30 minutes for best results.")

    df = pandas.DataFrame(metadata).sample(frac=1)
    n_eval = max(1, int(len(df) * 0.15))
    df[:n_eval].sort_values("audio_file").to_csv(EVAL_CSV, sep="|", index=False)
    df[n_eval:].sort_values("audio_file").to_csv(TRAIN_CSV, sep="|", index=False)

    print(f"Train samples: {len(df) - n_eval}  |  Eval samples: {n_eval}")
    print(f"Dataset written to: {DATASET_DIR}")

    del asr_model
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_finetune():
    print("\n" + "═" * 60)
    print("PHASE 2 — Fine-tuning XTTS v2 GPT encoder")
    print("═" * 60)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(EVAL_CSV):
        raise FileNotFoundError(
            "Dataset CSVs not found. Run Phase 1 first:\n"
            f"  {TRAIN_CSV}\n  {EVAL_CSV}"
        )

    import gc
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
    from TTS.utils.manage import ModelManager

    CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "XTTS_v2.0_original_model_files")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Download base XTTS v2 weights if not already present
    files_to_download = {
        "dvae.pth":       "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
        "mel_stats.pth":  "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
        "vocab.json":     "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
        "model.pth":      "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
        "config.json":    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json",
    }
    missing = [url for fname, url in files_to_download.items()
               if not os.path.isfile(os.path.join(CHECKPOINTS_DIR, fname))]
    if missing:
        print(f" > Downloading {len(missing)} base model file(s) …")
        ModelManager._download_model_files(missing, CHECKPOINTS_DIR, progress_bar=True)

    DVAE_CHECKPOINT  = os.path.join(CHECKPOINTS_DIR, "dvae.pth")
    MEL_NORM_FILE    = os.path.join(CHECKPOINTS_DIR, "mel_stats.pth")
    TOKENIZER_FILE   = os.path.join(CHECKPOINTS_DIR, "vocab.json")
    XTTS_CHECKPOINT  = os.path.join(CHECKPOINTS_DIR, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_DIR, "config.json")

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=DATASET_DIR,
        meta_file_train=TRAIN_CSV,
        meta_file_val=EVAL_CSV,
        language=LANGUAGE,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 s
        min_conditioning_length=66150,   # 3 s
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6 s
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        epochs=EPOCHS,
        output_path=TRAINING_DIR,
        model_args=model_args,
        run_name="GPT_XTTS_FT",
        project_name="XTTS_trainer",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Training on {len(train_samples)} samples, evaluating on {len(eval_samples)} samples")
    print(f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Grad accum: {GRAD_ACCUMULATION}  |  Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACCUMULATION,
        ),
        config,
        output_path=TRAINING_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    ft_checkpoint_dir = trainer.output_path
    print(f"\nFine-tuned checkpoint saved to: {ft_checkpoint_dir}")

    # Write a pointer file so Phase 3 can find the checkpoint automatically
    pointer = {"ft_checkpoint_dir": ft_checkpoint_dir, "xtts_config": XTTS_CONFIG_FILE, "tokenizer": TOKENIZER_FILE}
    pointer_path = os.path.join(OUTPUT_BASE_DIR, "checkpoint_pointer.json")
    with open(pointer_path, "w") as f:
        json.dump(pointer, f, indent=2)
    print(f"Checkpoint pointer written to: {pointer_path}")

    del model, trainer, train_samples, eval_samples
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Inference with fine-tuned model
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_inference(reference_wav: str = None):
    print("\n" + "═" * 60)
    print("PHASE 3 — Inference with fine-tuned model")
    print("═" * 60)

    pointer_path = os.path.join(OUTPUT_BASE_DIR, "checkpoint_pointer.json")
    if not os.path.exists(pointer_path):
        raise FileNotFoundError(
            "checkpoint_pointer.json not found. Run Phase 2 first."
        )

    with open(pointer_path) as f:
        pointer = json.load(f)

    ft_checkpoint_dir = pointer["ft_checkpoint_dir"]
    tokenizer_file    = pointer["tokenizer"]

    # Find the best (or latest) model checkpoint
    best_model = os.path.join(ft_checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model):
        candidates = sorted(glob.glob(os.path.join(ft_checkpoint_dir, "checkpoint_*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {ft_checkpoint_dir}")
        best_model = candidates[-1]
        print(f"best_model.pth not found, using latest: {os.path.basename(best_model)}")

    ft_config_file = os.path.join(ft_checkpoint_dir, "config.json")
    if not os.path.exists(ft_config_file):
        ft_config_file = pointer["xtts_config"]

    print(f"Checkpoint : {best_model}")
    print(f"Config     : {ft_config_file}")
    print(f"Tokenizer  : {tokenizer_file}")

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(ft_config_file)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=best_model,
        vocab_path=tokenizer_file,
        use_deepspeed=False,
    )

    device = "cuda" if torch.cuda.is_available() else "mps"
    model = model.to(device)
    model.eval()

    # Use provided reference, or fall back to any wav in the dataset
    if reference_wav is None:
        wavs = glob.glob(os.path.join(DATASET_DIR, "wavs", "*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No reference wav found in {DATASET_DIR}/wavs/")
        # pick the longest one for best conditioning
        reference_wav = max(wavs, key=os.path.getsize)

    print(f"Reference  : {reference_wav}")
    print(f"Text       : {TEST_TEXT}")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_wav],
        gpt_cond_len=GPT_COND_LEN,
        max_ref_length=MAX_REF_LEN,
        sound_norm_refs=SOUND_NORM_REFS,
    )

    out = model.inference(
        text=TEST_TEXT,
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        enable_text_splitting=True,
    )

    os.makedirs(os.path.dirname(TEST_OUTPUT), exist_ok=True)
    sf.write(TEST_OUTPUT, out["wav"], 24000)
    print(f"\nTest audio saved to: {TEST_OUTPUT}")
    print("Play it with:  afplay", TEST_OUTPUT)


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS v2 fine-tuning pipeline")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3],
        help="Run only a specific phase (1=dataset, 2=train, 3=inference). "
             "Omit to run all three phases end-to-end."
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        help="(Phase 3 only) Path to reference WAV for inference test. "
             "Defaults to the longest clip in the prepared dataset."
    )
    args = parser.parse_args()

    os.environ["COQUI_TOS_AGREED"] = "1"

    if args.phase is None or args.phase == 1:
        phase1_prepare_dataset()
    if args.phase is None or args.phase == 2:
        phase2_finetune()
    if args.phase is None or args.phase == 3:
        phase3_inference(reference_wav=args.ref)
