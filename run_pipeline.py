# ==============================================================================
# MASTER PIPELINE ORCHESTRATOR: PCAP COUNTER-VISUALITY DISTANT VIEWING
# Cultural Resistance: Counter-Visuality in the PCAP Archive
# Purpose: Executes the modular Python/R pipeline sequentially (00 -> 09),
#          strips Colab '%%R' cell magics on the fly for CLI Rscript
#          compatibility, verifies all output artifacts in data/ and results/,
#          and logs R sessionInfo().
# ==============================================================================

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List

# --- 1. EXECUTION CONFIGURATION FLAGS ---
CHROMEBOOK_ROOT   = Path("/mnt/chromeos/GoogleDrive/MyDrive/colab_pcap_folder")
COLAB_ROOT        = Path("/content/drive/MyDrive/colab_pcap_folder")

RUN_ENV_SETUP     = False  # Set True in Colab to provision GPU wheels & R binary cache (00)
RUN_API_HARVEST   = False  # Set True ONLY to re-query live UMich Omeka S API (01)
RUN_CLIP_STREAM   = False  # Set True on GPU runtime to stream & embed images via CLIP (02)
RUN_LLAVA_STREAM  = False  # Set True on GPU runtime to run 4-bit NF4 LLaVA interrogation (03)
RUN_R_PIPELINE    = True   # Run Tidyverse UMAP, STM, regressions, igraph & dossier (04-09)

# --- 2. DIRECTORY INITIALIZATION ---
if CHROMEBOOK_ROOT.exists():
    WORKSPACE = CHROMEBOOK_ROOT
elif COLAB_ROOT.exists():
    WORKSPACE = COLAB_ROOT
else:
    WORKSPACE = Path.cwd()

os.chdir(WORKSPACE)

for folder in ["data", "scripts", "results/charts", "results/figures", "r_libs"]:
    (WORKSPACE / folder).mkdir(parents=True, exist_ok=True)

EXPECTED_ARTIFACTS: List[str] = [
    "data/pcap_master_metadata.csv",
    "data/pcap_excluded_items.csv",
    "data/pcap_llava_interrogations.jsonl",
    "data/pcap_tidy_corpus.rds",
    "data/pcap_llava_stm_out.rds",
    "data/pcap_llava_searchK.rds",
    "data/pcap_stm_visual_k25.rds",
    "data/pcap_visual_topic_corr.rds",
    "results/charts/table01_pcap_medium_taxonomy.csv",
    "results/charts/table02_cluster_prompt_profiles.csv",
    "results/charts/table03_llava_visual_topics.csv",
    "results/charts/table04_temporal_regressions.csv",
    "results/charts/table04_temporal_regressions.html",
    "results/charts/table04b_half_decade_trends.csv",
    "results/charts/table05_material_motif_correlations.csv",
    "results/charts/qualitative_artwork_dossier.txt",
    "results/figures/fig01_disalienation_by_medium.png",
    "results/figures/fig02_umap_visual_clusters.png",
    "results/figures/fig02b_umap_faceted_by_medium.png",
    "results/figures/fig03_llava_stm_searchK.png",
    "results/figures/fig04_countervisuality_trajectories.png",
    "results/figures/fig05_crossmodal_network_igraph.png",
    "results/session_info.txt",
]


def run_python_script(script_path: str) -> None:
    print(f"\n>>> EXECUTING PYTHON SCRIPT: {script_path}")
    subprocess.run([sys.executable, script_path], check=True)


def run_r_script(script_path: str) -> None:
    """Strips Colab '%%R' cell magic from Line 1 before executing via Rscript."""
    print(f"\n>>> EXECUTING R SCRIPT: {script_path}")
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = [line for line in lines if not line.strip().startswith("%%R")]

    r_env = os.environ.copy()
    r_libs_dir = WORKSPACE / "r_libs"
    if r_libs_dir.exists():
        existing_r_libs = r_env.get("R_LIBS_USER", "")
        r_env["R_LIBS_USER"] = (
            f"{r_libs_dir}:{existing_r_libs}" if existing_r_libs else str(r_libs_dir)
        )

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as tmp:
        tmp.writelines(clean_lines)
        tmp_path = tmp.name

    try:
        subprocess.run(["Rscript", tmp_path], env=r_env, check=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def export_session_info() -> None:
    """Writes R sessionInfo() to results/session_info.txt for computational reproducibility."""
    r_cmd = 'writeLines(capture.output(sessionInfo()), "results/session_info.txt")'
    subprocess.run(["Rscript", "-e", r_cmd], check=True)
    print("[INFO] Exported R environment specifications to 'results/session_info.txt'.")


def verify_pipeline_artifacts(artifacts: List[str]) -> None:
    """Verifies that all expected output artifacts in data/ and results/ exist on disk."""
    missing = [p for p in artifacts if not (WORKSPACE / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"[ERROR] Missing expected pipeline artifacts:\n  - " + "\n  - ".join(missing)
        )
    print(f"[TDD PASS] Verified all {len(artifacts)} expected artifacts in data/ and results/.")


if __name__ == "__main__":
    if RUN_ENV_SETUP:
        run_python_script("scripts/00_colab_environment_setup.py")

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
    verify_pipeline_artifacts(EXPECTED_ARTIFACTS)
    print("\n[MASTER PIPELINE COMPLETE] All PCAP artifacts verified in data/ and results/.")
