# ==============================================================================
# scripts/03_llava_stratified_interrogation.py
# Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Selects a balanced 6-axis stratified high-signal subsample
#          and runs deterministic (`do_sample = False`) iconographic interrogation
#          using 4-bit NormalFloat (`NF4`) quantized llava-hf/llava-1.5-7b-hf
# OUTPUT : data/pcap_llava_interrogations.jsonl
# ==============================================================================

import io
import gc
import json
import time
import shutil
import random
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFile
import torch
import open_clip
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# ------------------------------------------------------------------------------
# 1. CONFIGURATION, CONTRASTIVE SURVEILLANCE & OBJECTS+STUFF LLAVA PROMPT
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path("/content/drive/MyDrive/colab_pcap_folder")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
CLIP_PARQUET_PATH = DATA_DIR / "pcap_clip_embeddings.parquet"
BACKUP_PARQUET_PATH = DATA_DIR / "pcap_clip_embeddings_v1_backup.parquet"
LLAVA_JSONL_PATH = DATA_DIR / "pcap_llava_interrogations.jsonl"

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Set PILOT_MODE = True to test 6 top-ranked artworks; set False for full N=1,200 run
PILOT_MODE = False

TARGET_SAMPLE_SIZE = 1200
HIGH_SIGNAL_COUNT = 900    # Top 150 highest-scoring exemplars across all 6 axes
CONTROL_GROUP_COUNT = 300  # 300 random baseline control artworks (< 90th percentile)
QUANTILE_TAU = 0.90
RANDOM_SEED = 42

USER_AGENT = "PCAP_Virginia_Tech_Research_Bot/1.0 (chayne@vt.edu)"
HEADERS = {"User-Agent": USER_AGENT}

HIGH_SIGNAL_AXES = [
    "score_surveillance_biometric",
    "score_carceral_containment",
    "score_bodily_autonomy",
    "score_tactile_disalienation",
    "score_digital_mediation",
    "score_state_bureaucracy",
]

# Contrastive Panoptic Surveillance Definition (Candidate B)
SURVEILLANCE_TARGET_PROMPTS = [
    "an artwork depicting a prison watchtower, security camera, searchlight beam, or panoptic surveillance",
    "being watched by correctional guards, disembodied watching eyes on a fence or sky, or cell observation window",
    "cctv camera lens, optical surveillance grid, retinal scan, or panopticon tower overlooking a prison",
]
SURVEILLANCE_CONFOUNDER_PROMPTS = [
    "a close-up pencil or charcoal drawing of a single sad human eye crying a tear",
    "close-up facial portrait focusing on eyelashes, eyebrows, and a reflective pupil",
]
ALPHA_CONFOUNDER_SUBTRACT = 0.25

# Prompt: Prison archive context, asks for held objects & wall markings,
# and separates foreground "objects" from background "stuff" (Arnold & Tilton, 2023, pp. 103-105)
LLAVA_PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "You are an art historian analyzing an artwork created by an incarcerated artist. "
    "Describe only what is visibly present in the work in three detailed paragraphs:\n"
    "1. Medium, colors, and formal texture: Describe the artistic medium, color palette, "
    "lighting, brushwork, line quality, and physical surface texture.\n"
    "2. Foreground figures and held objects: Describe any people or figures, their posture, "
    "facial expressions, the direction of their gaze, and anything worn on their bodies or "
    "held in their hands (such as chains, ropes, tools, cameras, phones, or weapons).\n"
    "3. Background stuff, architecture, and environment: Describe the surrounding setting, "
    "including any walls, fences, cell doors, windows, clocks, written text, numbers, symbols, "
    "or natural landscapes, water, trees, and open sky.\n"
    "ASSISTANT:"
)


# ------------------------------------------------------------------------------
# 2. SAFE IN-PLACE UPGRADE OF `score_surveillance_biometric` (CANDIDATE B)
# ------------------------------------------------------------------------------
def ensure_contrastive_surveillance_score() -> None:
    if BACKUP_PARQUET_PATH.exists():
        print(
            f"[INFO] Backup exists at {BACKUP_PARQUET_PATH.name}; "
            "Candidate B contrastive surveillance score is active."
        )
        return

    print("[INFO] Creating safe backup: data/pcap_clip_embeddings_v1_backup.parquet ...")
    shutil.copy2(CLIP_PARQUET_PATH, BACKUP_PARQUET_PATH)
    assert BACKUP_PARQUET_PATH.exists(), "[ERROR] Backup creation failed!"

    full_df = pd.read_parquet(CLIP_PARQUET_PATH)
    orig_rows, orig_cols = full_df.shape
    emb_cols = [f"clip_{i:03d}" for i in range(1, 769)]
    img_matrix = full_df[emb_cols].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def _encode_ensemble(prompts: List[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = tokenizer(prompts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean_feat = feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            return mean_feat.squeeze(0).cpu().numpy().astype(np.float32)

    vec_target = _encode_ensemble(SURVEILLANCE_TARGET_PROMPTS)
    vec_confounder = _encode_ensemble(SURVEILLANCE_CONFOUNDER_PROMPTS)
    vec_cand_b = vec_target - (ALPHA_CONFOUNDER_SUBTRACT * vec_confounder)
    vec_cand_b = (vec_cand_b / np.linalg.norm(vec_cand_b)).astype(np.float32)

    full_df["score_surveillance_biometric"] = (img_matrix @ vec_cand_b).astype(np.float32)
    assert full_df.shape == (orig_rows, orig_cols), "[ERROR] DataFrame shape mismatch!"
    full_df.to_parquet(CLIP_PARQUET_PATH, index=False, engine="pyarrow")
    print(f"[TDD PASS] Updated `score_surveillance_biometric` to Candidate B ({orig_rows} rows).")

    del clip_model, tokenizer, full_df, img_matrix
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------
# 3. TOP-150 RANKED QUOTA SUBSAMPLING (`nlargest` PER AXIS + CONTROL)
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
    Selects the Top 150 highest Z-score artworks (`nlargest`) for each of the 6
    theoretical axes (900 high-signal total) + 300 random baseline controls.
    """
    thresholds = {ax: float(df[ax].quantile(tau)) for ax in HIGH_SIGNAL_AXES}
    print(f"[INFO] High-Signal Decile Thresholds (tau = {tau:.2f}):")
    for ax, val in thresholds.items():
        print(f"  -> {ax:<30}: {val:.4f}")

    z_scores = pd.DataFrame(index=df.index)
    for ax in HIGH_SIGNAL_AXES:
        std_val = float(df[ax].std()) or 1.0
        z_scores[ax] = (df[ax] - float(df[ax].mean())) / std_val

    high_mask = pd.Series(False, index=df.index)
    for ax in HIGH_SIGNAL_AXES:
        high_mask |= df[ax] >= thresholds[ax]

    high_pool = df[high_mask].copy()
    control_pool = df[~high_mask].copy()

    high_pool["sampling_stratum"] = (
        "high_signal_"
        + z_scores.loc[high_pool.index].idxmax(axis=1).str.replace("score_", "", regex=False)
    )
    high_pool["peak_z"] = z_scores.loc[high_pool.index].max(axis=1)
    control_pool["sampling_stratum"] = "baseline_control"
    control_pool["peak_z"] = 0.0

    if len(df) <= target_n:
        return pd.concat([high_pool, control_pool], ignore_index=True).drop(columns=["peak_z"])

    # Select the TOP 150 highest-scoring exemplars (`nlargest`) per stratum
    per_axis_quota = high_n // len(HIGH_SIGNAL_AXES)
    sampled_parts = []
    used_ids = set()

    for ax in HIGH_SIGNAL_AXES:
        stratum_name = "high_signal_" + ax.replace("score_", "")
        grp = high_pool[high_pool["sampling_stratum"] == stratum_name]
        n_take = min(per_axis_quota, len(grp))
        if n_take > 0:
            picked = grp.nlargest(n_take, "peak_z")
            sampled_parts.append(picked)
            used_ids.update(picked["item_id"].tolist())

    sampled_high = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled_high) < high_n:
        shortfall = min(high_n - len(sampled_high), len(high_pool) - len(sampled_high))
        rem_pool = high_pool[~high_pool["item_id"].isin(used_ids)]
        extra_rows = rem_pool.nlargest(shortfall, "peak_z")
        sampled_high = pd.concat([sampled_high, extra_rows], ignore_index=True)

    actual_ctrl_n = min(control_n, len(control_pool), target_n - len(sampled_high))
    sampled_ctrl = control_pool.sample(n=actual_ctrl_n, random_state=seed)

    combined = (
        pd.concat([sampled_high, sampled_ctrl], ignore_index=True)
        .drop(columns=["peak_z"])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    print("\n[INFO] Final Stratified Sample Breakdown (Top-Ranked Exemplars + Control):")
    for st, cnt in combined["sampling_stratum"].value_counts().items():
        print(f"  -> {st:<40}: {cnt:4d} artworks")
    return combined


# ------------------------------------------------------------------------------
# 4. LOW-LEVEL HCI FORMAL METRICS & EPHEMERAL RAM LLAVA INTERROGATION
# ------------------------------------------------------------------------------
def compute_hci_formal_metrics(pil_image: Image.Image) -> Tuple[float, float, float]:
    """
    Computes 3 low-level formal color annotations in RAM following Arnold & Tilton
    (2023, Chapter 3, pp. 64-65, 93): mean intensity (brightness), mean chroma
    (saturation proxy), and warm-color pixel ratio.
    """
    arr = np.asarray(pil_image.resize((128, 128)), dtype=np.float32) / 255.0
    intensity = float(arr.mean())
    chroma = float((arr.max(axis=-1) - arr.min(axis=-1)).mean())
    # Warm pixels: Red channel exceeds Blue channel on non-grayscale pixels
    colorful_mask = (arr.max(axis=-1) - arr.min(axis=-1)) > 0.10
    if colorful_mask.sum() > 0:
        warm_ratio = float((arr[..., 0][colorful_mask] >= arr[..., 2][colorful_mask]).mean())
    else:
        warm_ratio = 0.5
    return round(intensity, 4), round(chroma, 4), round(warm_ratio, 4)


def load_quantized_llava(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] Script 03 requires a CUDA GPU runtime!")

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
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def interrogate_single_artwork_ram(
    image_url: str,
    session: requests.Session,
    model,
    processor,
) -> Tuple[str, float, float, float]:
    buffer = None
    pil_image = None
    inputs = None
    output_ids = None
    try:
        resp = session.get(image_url, headers=HEADERS, timeout=35)
        resp.raise_for_status()

        buffer = io.BytesIO(resp.content)
        pil_image = Image.open(buffer).convert("RGB")
        brightness, saturation, warm_ratio = compute_hci_formal_metrics(pil_image)

        inputs = processor(
            text=LLAVA_PROMPT_TEMPLATE,
            images=pil_image,
            return_tensors="pt",
        ).to(model.device, torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=325,
                do_sample=False,
                repetition_penalty=1.12,
            )

        decoded = processor.decode(output_ids[0], skip_special_tokens=True)
        transcript = (
            decoded.split("ASSISTANT:", 1)[-1].strip()
            if "ASSISTANT:" in decoded
            else decoded.strip()
        )
        return transcript, brightness, saturation, warm_ratio

    finally:
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        if pil_image is not None:
            pil_image.close()
            del pil_image
        if buffer is not None:
            buffer.close()
            del buffer


# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION LOOP
# ------------------------------------------------------------------------------
def main() -> None:
    ensure_contrastive_surveillance_score()

    meta_cols = [
        "item_id", "title", "raw_date", "exhibition_source",
        "medium_raw", "image_url", "omeka_web_url",
    ] + HIGH_SIGNAL_AXES

    df = pd.read_parquet(CLIP_PARQUET_PATH, columns=meta_cols)
    sample_df = select_stratified_subsample(df)

    model, processor = load_quantized_llava(LLAVA_MODEL_ID)
    session = requests.Session()

    if PILOT_MODE:
        print("\n" + "=" * 78)
        print("[PILOT MODE ACTIVE] Testing 6 top-ranked artworks (1 per axis)...")
        print("=" * 78)
        pilot_rows = (
            sample_df[sample_df["sampling_stratum"] != "baseline_control"]
            .groupby("sampling_stratum")
            .first()
            .reset_index()
        )
        for _, row in pilot_rows.iterrows():
            transcript, b_val, s_val, w_val = interrogate_single_artwork_ram(
                image_url=str(row["image_url"]).strip(),
                session=session,
                model=model,
                processor=processor,
            )
            print(f"\n>>> STRATUM : {row['sampling_stratum']}")
            print(f">>> TITLE   : {row['title']} (ID: {row['item_id']}) | Medium: {row['medium_raw']}")
            print(f">>> FORMAL  : Brightness={b_val:.3f}, Chroma={s_val:.3f}, WarmRatio={w_val:.3f}")
            print(f">>> URL     : {row['omeka_web_url']}")
            print(f">>> LLAVA   :\n{transcript}\n" + "-" * 78)
        print("[PILOT PASS] Flip PILOT_MODE = False (Line 41) for the full N=1,200 run.")
        return

    completed_ids: Set[str] = set()
    if LLAVA_JSONL_PATH.exists():
        with open(LLAVA_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("item_id") and rec.get("llava_description"):
                            completed_ids.add(str(rec["item_id"]))
                    except json.JSONDecodeError:
                        continue

    pending_df = sample_df[~sample_df["item_id"].astype(str).isin(completed_ids)].copy()
    if len(pending_df) == 0:
        print("[TDD PASS] All 1,200 stratified artworks have been interrogated!")
        return

    with open(LLAVA_JSONL_PATH, "a", encoding="utf-8") as jsonl_file:
        pbar = tqdm(pending_df.to_dict(orient="records"), desc="LLaVA Visual Interrogation")
        for idx, row in enumerate(pbar, start=1):
            item_id = str(row["item_id"])
            img_url = str(row["image_url"]).strip()
            if not img_url:
                continue

            try:
                transcript, b_val, s_val, w_val = interrogate_single_artwork_ram(
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
                    "formal_brightness": b_val,
                    "formal_saturation": s_val,
                    "formal_warm_ratio": w_val,
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
            time.sleep(random.uniform(0.3, 0.7))

    print(f"[TDD PASS] Script 03 Complete! Saved to {LLAVA_JSONL_PATH}")


if __name__ == "__main__":
    main()
