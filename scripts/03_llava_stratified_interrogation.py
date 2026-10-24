# ==============================================================================
# SCRIPT: scripts/03_llava_stratified_interrogation.py
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Selects the stratified high-signal subsample (top decile tau >= 0.90
#          across Surveillance, Bureaucracy/Biometric, Digital, and Bodily
#          Autonomy CLIP axes + random baseline control group; N = 1,200).
#          Streams images into volatile RAM and executes 4-bit quantized LLaVA
#          (llava-hf/llava-1.5-7b-hf) deterministic iconographic interrogation.
# ETHICS : ZERO PERSISTENT IMAGE STORAGE. Concludes Python's role in the project.
# OUTPUT : data/pcap_llava_interrogations.jsonl
# ==============================================================================

import io
import gc
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & DETERMINISTIC ICONOGRAPHIC PROMPT
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path("/content/drive/MyDrive/colab_pcap_folder")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
CLIP_PARQUET_PATH = DATA_DIR / "pcap_clip_embeddings.parquet"
LLAVA_JSONL_PATH = DATA_DIR / "pcap_llava_interrogations.jsonl"

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TARGET_SAMPLE_SIZE = 1200
HIGH_SIGNAL_COUNT = 900
CONTROL_GROUP_COUNT = 300
QUANTILE_TAU = 0.90
RANDOM_SEED = 42

USER_AGENT = "PCAP_Virginia_Tech_Dissertation_Bot/1.0 (chayne@vt.edu)"
HEADERS = {"User-Agent": USER_AGENT}

HIGH_SIGNAL_AXES = [
    "score_surveillance_biometric",
    "score_state_bureaucracy",
    "score_digital_mediation",
    "score_bodily_autonomy",
]

# Structured 4-Part Iconographic Prompt for LLaVA-1.5
LLAVA_PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "You are an art historian analyzing an artwork created by an incarcerated artist. "
    "Provide a detailed, objective iconographic description structured into four parts:\n"
    "(a) Formal composition, surface markings, and material texture;\n"
    "(b) Any depictions of eyes, security cameras, watchtowers, prison bars, razor wire, "
    "identification numbers, barcodes, or optical grids;\n"
    "(c) Bodily framing, posture, facial expression, and whether the subject returns a "
    "direct counter-gaze toward the viewer;\n"
    "(d) Visual practices of spatial or sensory disalienation, such as open horizons, "
    "natural landscapes, wildlife, or tactile craft.\n"
    "ASSISTANT:"
)


# ------------------------------------------------------------------------------
# 2. STRATIFIED HIGH-SIGNAL + BASELINE CONTROL SUBSAMPLING
# ------------------------------------------------------------------------------
def select_stratified_subsample(
    df: pd.DataFrame,
    target_n: int = TARGET_SAMPLE_SIZE,
    high_n: int = HIGH_SIGNAL_COUNT,
    control_n: int = CONTROL_GROUP_COUNT,
    tau: float = QUANTILE_TAU,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Identify artworks scoring in the top decile (tau >= 0.90) across the four
    target counter-visuality CLIP axes, plus a random baseline control group.
    Scales proportionally if the input corpus has fewer than 1,200 artworks.
    """
    # Calculate 90th percentile cutoff on each target axis
    thresholds = {ax: float(df[ax].quantile(tau)) for ax in HIGH_SIGNAL_AXES}
    print("[INFO] High-Signal Decile Thresholds (tau = 0.90):")
    for ax, val in thresholds.items():
        print(f"  -> {ax:<30}: {val:.4f}")

    high_mask = pd.Series(False, index=df.index)
    for ax in HIGH_SIGNAL_AXES:
        high_mask |= df[ax] >= thresholds[ax]

    high_pool = df[high_mask].copy()
    control_pool = df[~high_mask].copy()

    # Assign each high-signal artwork to its dominant standardized theoretical axis
    z_scores = pd.DataFrame(index=high_pool.index)
    for ax in HIGH_SIGNAL_AXES:
        std_val = float(df[ax].std()) or 1.0
        z_scores[ax] = (high_pool[ax] - float(df[ax].mean())) / std_val
    high_pool["sampling_stratum"] = (
        "high_signal_" + z_scores.idxmax(axis=1).str.replace("score_", "", regex=False)
    )
    control_pool["sampling_stratum"] = "baseline_control"

    if len(df) <= target_n:
        print(
            f"[INFO] Corpus size ({len(df)}) <= target ({target_n}); "
            "retaining full corpus with stratum labels."
        )
        return pd.concat([high_pool, control_pool], ignore_index=True)

    actual_high_n = min(high_n, len(high_pool))
    actual_ctrl_n = min(control_n, len(control_pool), target_n - actual_high_n)

    # Pandas 2.2+ safe loop across strata (preserves sampling_stratum column)
    sampled_parts = []
    for _, grp in high_pool.groupby("sampling_stratum"):
        n_stratum = max(1, int(round(actual_high_n * len(grp) / len(high_pool))))
        n_stratum = min(n_stratum, len(grp))
        sampled_parts.append(grp.sample(n=n_stratum, random_state=seed))

    sampled_high = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_high) > actual_high_n:
        sampled_high = sampled_high.sample(n=actual_high_n, random_state=seed)
    elif len(sampled_high) < actual_high_n:
        rem_pool = high_pool[~high_pool["item_id"].isin(set(sampled_high["item_id"]))]
        extra_rows = rem_pool.sample(
            n=actual_high_n - len(sampled_high), random_state=seed
        )
        sampled_high = pd.concat([sampled_high, extra_rows], ignore_index=True)

    sampled_ctrl = control_pool.sample(n=actual_ctrl_n, random_state=seed)
    combined = pd.concat([sampled_high, sampled_ctrl], ignore_index=True)

    assert "sampling_stratum" in combined.columns, "[ERROR] sampling_stratum column missing!"
    print(
        f"[TDD PASS] Stratified Subsample Selected: Total N={len(combined)} "
        f"(High-Signal={len(sampled_high)}, Baseline Control={len(sampled_ctrl)})"
    )
    return combined


# ------------------------------------------------------------------------------
# 3. LOAD 4-BIT QUANTIZED LLAVA MODEL ON GPU
# ------------------------------------------------------------------------------
def load_quantized_llava(model_id: str):
    """Load LLaVA-1.5-7B with 4-bit NF4 quantization for Colab GPU efficiency."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[ERROR] Script 03 requires a CUDA GPU runtime for 4-bit LLaVA inference!"
        )

    print(f"[INFO] Loading 4-bit NF4 quantized {model_id} onto CUDA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[TDD PASS] 4-bit Quantized LLaVA model and processor initialized.")
    return model, processor


# ------------------------------------------------------------------------------
# 4. EPHEMERAL RAM STREAMING & LLAVA INTERROGATION
# ------------------------------------------------------------------------------
def interrogate_single_artwork_ram(
    image_url: str,
    session: requests.Session,
    model,
    processor,
) -> str:
    """
    Stream artwork bytes into io.BytesIO, run deterministic LLaVA generation,
    and immediately purge all image binaries from volatile RAM.
    """
    buffer = None
    pil_image = None
    try:
        resp = session.get(image_url, headers=HEADERS, timeout=35)
        resp.raise_for_status()

        buffer = io.BytesIO(resp.content)
        pil_image = Image.open(buffer).convert("RGB")

        inputs = processor(
            text=LLAVA_PROMPT_TEMPLATE,
            images=pil_image,
            return_tensors="pt",
        ).to(model.device, torch.float16)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False,
            )

        decoded = processor.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in decoded:
            transcript = decoded.split("ASSISTANT:", 1)[-1].strip()
        else:
            transcript = decoded.strip()

        return transcript

    finally:
        if pil_image is not None:
            pil_image.close()
            del pil_image
        if buffer is not None:
            buffer.close()
            del buffer


# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION LOOP WITH FAULT-TOLERANT JSONL CHECKPOINTING
# ------------------------------------------------------------------------------
def main() -> None:
    if not CLIP_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Missing {CLIP_PARQUET_PATH}. Run Script 02 first!"
        )

    meta_cols = (
        [
            "item_id",
            "title",
            "raw_date",
            "exhibition_source",
            "medium_raw",
            "image_url",
            "omeka_web_url",
        ]
        + HIGH_SIGNAL_AXES
        + [
            "score_tactile_disalienation",
            "score_carceral_containment",
        ]
    )

    df = pd.read_parquet(CLIP_PARQUET_PATH, columns=meta_cols)
    sample_df = select_stratified_subsample(df)

    # Check existing JSONL output to resume cleanly if interrupted
    completed_ids: Set[str] = set()
    if LLAVA_JSONL_PATH.exists():
        with open(LLAVA_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "item_id" in rec and rec.get("llava_description"):
                        completed_ids.add(str(rec["item_id"]))
                except json.JSONDecodeError:
                    continue
        print(
            f"[INFO] Found {len(completed_ids)} completed LLaVA transcripts in "
            f"{LLAVA_JSONL_PATH}."
        )

    pending_df = sample_df[
        ~sample_df["item_id"].astype(str).isin(completed_ids)
    ].copy()

    if len(pending_df) == 0:
        print("[TDD PASS] All stratified sample artworks have been interrogated!")
        return

    model, processor = load_quantized_llava(LLAVA_MODEL_ID)
    session = requests.Session()

    print(
        f"[INFO] Executing LLaVA iconographic interrogation on {len(pending_df)} artworks..."
    )
    with open(LLAVA_JSONL_PATH, "a", encoding="utf-8") as jsonl_file:
        pbar = tqdm(
            pending_df.to_dict(orient="records"),
            desc="LLaVA Visual Interrogation",
        )
        for idx, row in enumerate(pbar, start=1):
            item_id = str(row["item_id"])
            img_url = str(row["image_url"]).strip()
            if not img_url:
                continue

            try:
                transcript = interrogate_single_artwork_ram(
                    image_url=img_url,
                    session=session,
                    model=model,
                    processor=processor,
                )

                record = {
                    "item_id": item_id,
                    "title": str(row.get("title", "")),
                    "sampling_stratum": str(row.get("sampling_stratum", "")),
                    "score_surveillance_biometric": float(row["score_surveillance_biometric"]),
                    "score_state_bureaucracy": float(row["score_state_bureaucracy"]),
                    "score_digital_mediation": float(row["score_digital_mediation"]),
                    "score_bodily_autonomy": float(row["score_bodily_autonomy"]),
                    "score_tactile_disalienation": float(row["score_tactile_disalienation"]),
                    "score_carceral_containment": float(row["score_carceral_containment"]),
                    "llava_model": LLAVA_MODEL_ID,
                    "llava_description": transcript,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            except Exception as err:
                print(f"\n[WARN] LLaVA failed on item_id={item_id}: {err}")
                continue

            if idx % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            time.sleep(random.uniform(0.5, 1.2))

    print("=" * 78)
    print(f"[TDD PASS] Script 03 Complete! (Python GPU Phase Concluded)")
    print(f"  -> Output JSONL : {LLAVA_JSONL_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
