# ==============================================================================
# MASTER PIPELINE ORCHESTRATOR: PCAP COUNTER-VISUALITY DISTANT VIEWING
# Purpose: Executes the modular Python/R pipeline sequentially, strips Colab
#          '%%R' cell magics on the fly for CLI Rscript compatibility, verifies
#          output artifacts in data/ and results/, and logs sessionInfo().
# ==============================================================================

import os
import subprocess
import tempfile
from pathlib import Path

# --- 1. EXECUTION CONFIGURATION FLAGS ---
COLAB_ROOT        = Path("/content/drive/MyDrive/colab_pcap_folder")
RUN_API_HARVEST   = False  # Set True ONLY to re-query live UMich Omeka S API (01)
RUN_CLIP_STREAM   = False  # Set True on GPU runtime to stream & embed images via CLIP (02)
RUN_LLAVA_STREAM  = False  # Set True on GPU runtime to run 4-bit LLaVA interrogation (03)
RUN_R_PIPELINE    = True   # Run Tidyverse UMAP, STM, regressions, igraph & dossier (04-09)

# --- 2. DIRECTORY INITIALIZATION ---
WORKSPACE = COLAB_ROOT if COLAB_ROOT.exists() else Path.cwd()
os.chdir(WORKSPACE)

for folder in ["data", "scripts", "results/charts", "results/figures"]:
    (WORKSPACE / folder).mkdir(parents=True, exist_ok=True)


def run_python_script(script_path: str) -> None:
    print(f"\n>>> EXECUTING PYTHON SCRIPT: {script_path}")
    subprocess.run(["python3", script_path], check=True)


def run_r_script(script_path: str) -> None:
    """Strips Colab '%%R' cell magic from Line 1 before executing via Rscript."""
    print(f"\n>>> EXECUTING R SCRIPT: {script_path}")
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = [line for line in lines if not line.strip().startswith("%%R")]

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as tmp:
        tmp.writelines(clean_lines)
        tmp_path = tmp.name

    try:
        subprocess.run(["Rscript", tmp_path], check=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def export_session_info() -> None:
    """Writes R sessionInfo() to results/session_info.txt for computational reproducibility."""
    r_cmd = 'writeLines(capture.output(sessionInfo()), "results/session_info.txt")'
    subprocess.run(["Rscript", "-e", r_cmd], check=True)
    print("[INFO] Exported R environment specifications to 'results/session_info.txt'.")


if __name__ == "__main__":
    if RUN_API_HARVEST:
        run_python_script("scripts/01_harvest_pcap_omeka_api.py")

    if RUN_CLIP_STREAM:
        run_python_script("scripts/02_stream_clip_distant_viewing.py")

    if RUN_LLAVA_STREAM:
        run_python_script("scripts/03_llava_stratified_interrogation.py")

    if RUN_R_PIPELINE:
        run_r_script("scripts/04_tidy_metadata_and_umap.R")
        run_r_script("scripts/05_llava_tidy_nlp_and_searchK.R")
        run_r_script("scripts/06_stm_visual_motifs_k25.R")
        run_r_script("scripts/07_countervisuality_regressions.R")
        run_r_script("scripts/08_crossmodal_network_igraph.R")
        run_r_script("scripts/09_qualitative_artwork_dossier.R")

    export_session_info()
    print("\n[MASTER PIPELINE COMPLETE] All PCAP artifacts verified in data/ and results/.")
