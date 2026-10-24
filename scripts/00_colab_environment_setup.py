# ==============================================================================
# SCRIPT: scripts/00_colab_environment_setup.py
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Phase 0 Environment Initialization for Google Colab (Python + %%R).
#          Mounts Google Drive, locks working directory, builds the standardized
#          repository hierarchy, installs GPU computer vision libraries, loads
#          rpy2 RMagic, and installs R Tidyverse/STM/UMAP binary packages.
# ==============================================================================

import os
import sys
import subprocess
from pathlib import Path
from typing import List

# ------------------------------------------------------------------------------
# 1. GOOGLE DRIVE MOUNT & WORKING DIRECTORY LOCK
# ------------------------------------------------------------------------------
COLAB_ROOT = Path("/content/drive/MyDrive/colab_pcap_folder")
REQUIRED_DIRS = [
    COLAB_ROOT / "data",
    COLAB_ROOT / "scripts",
    COLAB_ROOT / "results" / "charts",
    COLAB_ROOT / "results" / "figures",
]


def initialize_workspace(root_path: Path, subdirs: List[Path]) -> None:
    """Mount Google Drive (if in Colab) and create the required directory tree."""
    print("=" * 78)
    print("[INFO] PHASE 0: Initializing PCAP Counter-Visuality Colab Environment")
    print("=" * 78)

    try:
        from google.colab import drive  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            print("[INFO] Mounting Google Drive at /content/drive ...")
            drive.mount("/content/drive", force_remount=False)
        else:
            print("[INFO] Google Drive is already mounted.")
    except ImportError:
        print("[WARN] Not running inside Google Colab. Using local fallback path.")
        root_path = Path.cwd() / "colab_pcap_folder"
        subdirs = [
            root_path / "data",
            root_path / "scripts",
            root_path / "results" / "charts",
            root_path / "results" / "figures",
        ]

    # Create directory hierarchy
    for directory in [root_path] + subdirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Verified directory: {directory}")

    # Lock working directory
    os.chdir(root_path)
    assert Path.cwd().resolve() == root_path.resolve(), (
        f"[ERROR] Failed to lock working directory to {root_path}"
    )
    print(f"[TDD PASS] Working directory locked to: {Path.cwd()}")


initialize_workspace(COLAB_ROOT, REQUIRED_DIRS)

# ------------------------------------------------------------------------------
# 2. INSTALL PYTHON GPU & DATA SERIALIZATION LIBRARIES
# ------------------------------------------------------------------------------
PYTHON_PACKAGES = [
    "open_clip_torch>=2.24.0",
    "transformers>=4.40.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.30.0",
    "pyarrow>=15.0.0",
    "requests>=2.31.0",
    "Pillow>=10.0.0",
    "tqdm>=4.66.0",
    "rpy2>=3.5.11",
]


def install_python_packages(packages: List[str]) -> None:
    """Install required Python packages quietly via pip with error reporting."""
    print("\n[INFO] Installing Python GPU & serialization libraries...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] pip installation failed:\n{result.stderr}")
        raise RuntimeError("Python package installation failed.")
    print("[TDD PASS] Python libraries installed successfully.")


install_python_packages(PYTHON_PACKAGES)

# ------------------------------------------------------------------------------
# 3. TDD VERIFICATION OF PYTHON LIBRARIES & GPU RUNTIME
# ------------------------------------------------------------------------------
def verify_python_environment() -> None:
    """Run TDD assertions to confirm Python dependencies and check GPU state."""
    import torch
    import open_clip
    import transformers
    import pyarrow
    from PIL import Image

    print(f"[INFO] PyTorch Version      : {torch.__version__}")
    print(f"[INFO] OpenCLIP Version     : {open_clip.__version__}")
    print(f"[INFO] Transformers Version : {transformers.__version__}")
    print(f"[INFO] PyArrow Version      : {pyarrow.__version__}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[TDD PASS] GPU Runtime Active: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print(
            "[WARN] No CUDA GPU detected! CPU runtime is active. "
            "CPU is ideal for R scripts (04-09), but switch to a GPU runtime "
            "(T4/L4/A100) before running Python Scripts 02 and 03."
        )


verify_python_environment()

# ------------------------------------------------------------------------------
# 4. ACTIVATE RPY2 RMAGIC FOR GOOGLE COLAB
# ------------------------------------------------------------------------------
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "rpy2.ipython")
        print("[TDD PASS] rpy2.ipython extension (%R / %%R) activated.")
except Exception as exc:
    print(f"[WARN] Could not auto-activate rpy2 magic via IPython: {exc}")

# ==============================================================================
# 5. R PACKAGE INSTALLATION BLOCK (Executed automatically via rpy2 below, or
#    paste into a Colab cell with %%R)
# ==============================================================================
R_SETUP_SCRIPT = """
options(warn = 1)
cat("==============================================================================\\n")
cat("[INFO] Configuring Posit Public Package Manager (Ubuntu Binary Repo) for R...\\n")
cat("==============================================================================\\n")

# Use pre-compiled Ubuntu Jammy Linux binaries to cut install time from ~35m to ~1m
options(
  repos = c(
    RSPM = "[https://packagemanager.posit.co/cran/__linux__/jammy/latest](https://packagemanager.posit.co/cran/__linux__/jammy/latest)",
    CRAN = "[https://cloud.r-project.org](https://cloud.r-project.org)"
  ),
  Ncpus = parallel::detectCores()
)

required_r_pkgs <- c(
  "tidyverse",
  "arrow",
  "jsonlite",
  "uwot",
  "cluster",
  "tidytext",
  "textstem",
  "stm",
  "igraph",
  "broom",
  "modelsummary",
  "gt",
  "patchwork",
  "scales",
  "Matrix"
)

missing_pkgs <- required_r_pkgs[!vapply(required_r_pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_pkgs) > 0) {
  cat(sprintf("[INFO] Installing %d missing R packages: %s\\n",
              length(missing_pkgs), paste(missing_pkgs, collapse = ", ")))
  install.packages(missing_pkgs, dependencies = TRUE, quiet = TRUE)
} else {
  cat("[INFO] All required R packages are already installed.\\n")
}

# TDD Assertion Gate: Verify every package loads cleanly
failed_pkgs <- character(0)
for (pkg in required_r_pkgs) {
  ok <- suppressPackageStartupMessages(
    require(pkg, character.only = TRUE, quietly = TRUE)
  )
  if (!ok) {
    failed_pkgs <- c(failed_pkgs, pkg)
  }
}

if (length(failed_pkgs) > 0) {
  stop(sprintf("[ERROR] The following R packages failed to load: %s",
               paste(failed_pkgs, collapse = ", ")))
}

cat("[TDD PASS] All 15 R packages installed and verified for Phase 2 (Scripts 04-09).\\n")
"""

if __name__ == "__main__":
    print("\n[INFO] Executing R environment provisioning via rpy2...")
    try:
        import rpy2.robjects as robjects
        robjects.r(R_SETUP_SCRIPT)
    except Exception as r_err:
        print(f"[ERROR] R environment setup encountered an error: {r_err}")
        raise
