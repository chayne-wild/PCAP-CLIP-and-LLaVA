# ==============================================================================
# SCRIPT: scripts/02_stream_clip_distant_viewing.py
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Streams each PCAP artwork's image bytes into volatile RAM (io.BytesIO),
#          computes L2-normalized 768-D CLIP (ViT-L/14) visual embeddings, and
#          calculates zero-shot cosine similarity scores against a 6-axis
#          Theoretical Prompt Matrix grounded in Fanon/Tosquelles (5.1),
#          Arnold & Tilton (5.2), and Mirzoeff/Browne (5.4).
# ETHICS : ZERO PERSISTENT IMAGE STORAGE. Every image buffer is immediately
#          closed and garbage-collected after the GPU forward pass.
# OUTPUT : data/pcap_clip_embeddings.parquet
# ==============================================================================

import io
import gc
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageFile
import torch
import open_clip
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------------------
# 1. PATHS, MODEL PARAMETERS & 6-AXIS THEORETICAL PROMPT MATRIX
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path("/content/drive/MyDrive/colab_pcap_folder")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
MASTER_CSV_PATH = DATA_DIR / "pcap_master_metadata.csv"
CLIP_PARQUET_PATH = DATA_DIR / "pcap_clip_embeddings.parquet"

CLIP_MODEL_ARCH = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 768
FLUSH_BATCH_SIZE = 100

USER_AGENT = "PCAP_Virginia_Tech_Dissertation_Bot/1.0 (chayne@vt.edu)"
HEADERS = {"User-Agent": USER_AGENT}

# 6-Axis Theoretical Codebook (Ensemble prompts averaged in L2-normalized space)
THEORETICAL_PROMPT_MATRIX: Dict[str, List[str]] = {
    "score_surveillance_biometric": [
        "an artwork depicting a giant watching eye, panoptic gaze, security camera, or prison watchtower",
        "surveillance spotlight, optical grid, retinal scan, or being watched by correctional guards",
        "eyes staring through prison bars, cctv camera lens, or panopticon tower overlooking a yard",
    ],
    "score_carceral_containment": [
        "an artwork depicting steel prison bars, razor wire fence, locked cell door, and concrete walls",
        "claustrophobic prison cell interior, handcuffs, chains, shackles, and architectural enclosure",
        "barbed wire coils, solitary confinement box, and cold institutional prison corridor",
    ],
    "score_bodily_autonomy": [
        "a defiant self-portrait of a person returning the viewer's gaze with dignity and bodily sovereignty",
        "an expressive portrait of an assertive human figure reclaiming bodily autonomy and strength",
        "close-up human face looking directly forward with resilience, pride, and emotional depth",
    ],
    "score_tactile_disalienation": [
        "a richly textured handmade artwork with thick impasto, vibrant organic nature, trees, and open sky",
        "serene natural landscape, wildlife, flowing water, flowers, and warm tactile sensory vitality",
        "intricately crafted tactile surface celebrating nature, spiritual freedom, and open horizons",
    ],
    "score_digital_mediation": [
        "an artwork depicting a prison telephone behind plexiglass, video visitation monitor, or digital screen",
        "electronic cables, computer terminal, pixelated television screen, or telecommunications barrier",
        "talking through an institutional telephone handset or staring at a glowing digital kiosk interface",
    ],
    "score_state_bureaucracy": [
        "an artwork incorporating prisoner identification numbers, barcodes, mugshot placards, or state forms",
        "stamped bureaucratic numbers, inmate ID badge, legal documents, tally marks, and serial digits",
        "a human figure labeled with a barcode, prison identification number, or institutional paperwork",
    ],
}


# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION & ZERO-SHOT TEXT ENCODING
# ------------------------------------------------------------------------------
def load_clip_and_encode_prompts(device: torch.device):
    """
    Load OpenCLIP ViT-L/14 and precompute the L2-normalized 768-D text prototype
    vectors for each of the 6 theoretical axes.
    """
    print(f"[INFO] Loading OpenCLIP {CLIP_MODEL_ARCH} ({CLIP_PRETRAINED}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_ARCH, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_ARCH)

    axis_names = list(THEORETICAL_PROMPT_MATRIX.keys())
    axis_embeddings = []

    with torch.no_grad():
        for axis in axis_names:
            prompts = THEORETICAL_PROMPT_MATRIX[axis]
            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens).float()
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Average ensemble prompts and re-normalize to unit hypersphere
            mean_feat = text_feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            axis_embeddings.append(mean_feat)

    # Shape: (6, 768)
    prompt_matrix_tensor = torch.cat(axis_embeddings, dim=0)
    assert prompt_matrix_tensor.shape == (6, EMBEDDING_DIM), (
        f"[ERROR] Expected (6, {EMBEDDING_DIM}) prompt tensor, got {prompt_matrix_tensor.shape}"
    )
    print("[TDD PASS] 6-Axis Theoretical Prompt Matrix encoded and L2-normalized.")
    return model, preprocess, axis_names, prompt_matrix_tensor


# ------------------------------------------------------------------------------
# 3. EPHEMERAL RAM IMAGE STREAMING & FORWARD PASS
# ------------------------------------------------------------------------------
def stream_and_extract_single_artwork(
    image_url: str,
    session: requests.Session,
    model,
    preprocess,
    prompt_matrix_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream image bytes into an ephemeral io.BytesIO RAM buffer, compute the 768-D
    CLIP embedding and 6 cosine similarity scores, and immediately purge the
    image binary from memory.
    """
    buffer = None
    pil_image = None
    try:
        resp = session.get(image_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # Load strictly into volatile RAM buffer
        buffer = io.BytesIO(resp.content)
        pil_image = Image.open(buffer).convert("RGB")
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    img_feat = model.encode_image(image_tensor)
            else:
                img_feat = model.encode_image(image_tensor)

            img_feat = img_feat.float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Cosine similarity in unit-normalized space is the dot product
            sim_scores = (img_feat @ prompt_matrix_tensor.T).squeeze(0)

            emb_np = img_feat.squeeze(0).cpu().numpy().astype(np.float32)
            scores_np = sim_scores.cpu().numpy().astype(np.float32)

        return emb_np, scores_np

    finally:
        # Strict Ethical Compliance: Always close and delete image buffers
        if pil_image is not None:
            pil_image.close()
            del pil_image
        if buffer is not None:
            buffer.close()
            del buffer


# ------------------------------------------------------------------------------
# 4. TDD SELF-TEST ON SYNTHETIC IN-MEMORY TENSOR
# ------------------------------------------------------------------------------
def run_tdd_clip_verification(model, preprocess, prompt_matrix_tensor, device) -> None:
    """Verify forward pass dimensions and L2 normalization using an in-memory image."""
    test_buf = io.BytesIO()
    test_img = Image.new("RGB", (224, 224), color=(90, 120, 160))
    test_img.save(test_buf, format="JPEG")
    test_buf.seek(0)

    loaded_img = Image.open(test_buf).convert("RGB")
    img_tensor = preprocess(loaded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img_tensor).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        scores = (feat @ prompt_matrix_tensor.T).squeeze(0)

    loaded_img.close()
    test_buf.close()
    del loaded_img, test_buf, test_img

    l2_norm = float(torch.norm(feat).item())
    assert abs(l2_norm - 1.0) < 1e-4, f"[ERROR] Embedding L2 norm={l2_norm} != 1.0"
    assert feat.shape == (1, EMBEDDING_DIM), f"[ERROR] Invalid shape {feat.shape}"
    assert scores.shape == (6,), f"[ERROR] Invalid score vector shape {scores.shape}"
    print(f"[TDD PASS] Ephemeral CLIP forward pass verified (L2 norm={l2_norm:.5f}).")


# ------------------------------------------------------------------------------
# 5. MAIN STREAMING ORCHESTRATOR WITH PARQUET CHECKPOINTING
# ------------------------------------------------------------------------------
def main() -> None:
    if not MASTER_CSV_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Missing {MASTER_CSV_PATH}. Run Script 01 first!"
        )

    master_df = pd.read_csv(MASTER_CSV_PATH, dtype=str).fillna("")
    print(f"[INFO] Loaded {len(master_df)} artwork records from {MASTER_CSV_PATH}.")

    # Check for existing Parquet checkpoint to support fault-tolerant resume
    processed_ids = set()
    existing_df = None
    if CLIP_PARQUET_PATH.exists():
        try:
            existing_df = pd.read_parquet(CLIP_PARQUET_PATH)
            processed_ids = set(existing_df["item_id"].astype(str).tolist())
            print(
                f"[INFO] Resuming from existing Parquet checkpoint: "
                f"{len(processed_ids)} artworks already embedded."
            )
        except Exception as exc:
            print(f"[WARN] Could not read existing Parquet file ({exc}); starting fresh.")
            existing_df = None

    remaining_df = master_df[~master_df["item_id"].astype(str).isin(processed_ids)].copy()
    if len(remaining_df) == 0:
        print("[TDD PASS] All artworks in master CSV are already embedded in Parquet!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, axis_names, prompt_matrix_tensor = load_clip_and_encode_prompts(device)
    run_tdd_clip_verification(model, preprocess, prompt_matrix_tensor, device)

    emb_col_names = [f"clip_{i:03d}" for i in range(1, EMBEDDING_DIM + 1)]
    session = requests.Session()
    batch_records = []

    print(
        f"[INFO] Streaming {len(remaining_df)} remaining artworks into volatile RAM..."
    )
    pbar = tqdm(remaining_df.to_dict(orient="records"), desc="CLIP Distant Viewing")

    for row in pbar:
        item_id = str(row["item_id"])
        img_url = str(row["image_url"]).strip()
        if not img_url:
            continue

        try:
            emb_vec, score_vec = stream_and_extract_single_artwork(
                image_url=img_url,
                session=session,
                model=model,
                preprocess=preprocess,
                prompt_matrix_tensor=prompt_matrix_tensor,
                device=device,
            )

            out_record = dict(row)
            for ax_name, sc_val in zip(axis_names, score_vec):
                out_record[ax_name] = float(sc_val)
            for col_name, e_val in zip(emb_col_names, emb_vec):
                out_record[col_name] = float(e_val)

            batch_records.append(out_record)

        except Exception as err:
            print(f"\n[WARN] Skipping item_id={item_id} due to stream error: {err}")
            continue

        # Polite rate-limiting
        time.sleep(random.uniform(0.3, 0.8))

        # Periodically flush batch to Apache Parquet
        if len(batch_records) >= FLUSH_BATCH_SIZE:
            batch_df = pd.DataFrame(batch_records)
            if existing_df is not None:
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            else:
                combined_df = batch_df
            combined_df.to_parquet(CLIP_PARQUET_PATH, index=False, engine="pyarrow")
            existing_df = combined_df
            batch_records.clear()
            gc.collect()

    # Final flush of remaining records
    if batch_records:
        batch_df = pd.DataFrame(batch_records)
        if existing_df is not None:
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
        else:
            combined_df = batch_df
        combined_df.to_parquet(CLIP_PARQUET_PATH, index=False, engine="pyarrow")
        existing_df = combined_df
        batch_records.clear()
        gc.collect()

    # Final TDD Verification of Parquet Output
    assert CLIP_PARQUET_PATH.exists(), "[ERROR] Parquet output file was not created!"
    final_meta = pq.read_metadata(CLIP_PARQUET_PATH)
    print("=" * 78)
    print(f"[TDD PASS] Script 02 Complete!")
    print(f"  -> Output Parquet : {CLIP_PARQUET_PATH}")
    print(f"  -> Total Rows     : {final_meta.num_rows}")
    print(f"  -> Total Columns  : {final_meta.num_columns} (Metadata + 6 Scores + 768 Dims)")
    print("=" * 78)


if __name__ == "__main__":
    main()
