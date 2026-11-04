# ==============================================================================
# scripts/00_colab_environment_setup.py (FAST BINARY & DRIVE-CACHED)
# Cultural Resistance: Counter-Visuality in the PCAP Archive
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
    COLAB_ROOT / "r_libs",  # Persistent R binary cache across Colab disconnects
]


def initialize_workspace(root_path: Path, subdirs: List[Path]) -> Path:
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
            root_path / "r_libs",
        ]

    for directory in [root_path] + subdirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Verified directory: {directory}")

    os.chdir(root_path)
    assert Path.cwd().resolve() == root_path.resolve(), (
        f"[ERROR] Failed to lock working directory to {root_path}"
    )
    print(f"[TDD PASS] Working directory locked to: {Path.cwd()}")
    return root_path


COLAB_ROOT = initialize_workspace(COLAB_ROOT, REQUIRED_DIRS)

# ------------------------------------------------------------------------------
# 2. CONDITIONAL PYTHON GPU INSTALL (SKIPS HEAVY GPU WHEELS ON CPU RUNTIMES)
# ------------------------------------------------------------------------------
import torch

HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n[TDD PASS] GPU Runtime Active: {device_name} ({vram_gb:.2f} GB VRAM)")
    print("[INFO] Installing Python GPU & vision libraries for Steps 02-03...")
    GPU_PACKAGES = [
        "open_clip_torch>=2.24.0",
        "transformers>=4.40.0,<5.0.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + GPU_PACKAGES
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"[ERROR] pip installation failed:\n{res.stderr}")
    print("[TDD PASS] Python GPU libraries installed successfully.")
else:
    print(
        "\n[INFO] CPU Runtime Detected! Skipping GPU-only Python wheels "
        "(open_clip, bitsandbytes) to save startup time.\n"
        "       -> CPU is ready for Step 01 (API Harvest) and Steps 04-09 (R/STM)."
    )

# ------------------------------------------------------------------------------
# 3. FAST UBUNTU APT BINARY PRE-INSTALL FOR HEAVY C++ R PACKAGES
# ------------------------------------------------------------------------------
APT_R_BINARIES = [
    "r-cran-tidyverse",
    "r-cran-arrow",
    "r-cran-jsonlite",
    "r-cran-uwot",
    "r-cran-cluster",
    "r-cran-tidytext",
    "r-cran-tm",
    "r-cran-snowballc",
    "r-cran-stm",
    "r-cran-igraph",
    "r-cran-broom",
    "r-cran-patchwork",
    "r-cran-scales",
    "r-cran-matrix",
]

if Path("/usr/bin/apt-get").exists():
    print("\n[INFO] Installing pre-compiled Ubuntu .deb binaries for heavy C++ R packages...")
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    # Install each available apt binary quietly (ignores any not in current PPA)
    for apt_pkg in APT_R_BINARIES:
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", apt_pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    print("[TDD PASS] Ubuntu system R binary check complete.")

# ------------------------------------------------------------------------------
# 4. ACTIVATE RPY2 RMAGIC FOR GOOGLE COLAB
# ------------------------------------------------------------------------------
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("reload_ext", "rpy2.ipython")
        print("[TDD PASS] rpy2.ipython extension (%R / %%R) activated.")
except Exception as exc:
    print(f"[WARN] Could not auto-activate rpy2 magic via IPython: {exc}")

# ------------------------------------------------------------------------------
# 5. R BINARY SETUP (POSIT HTTPUserAgent + DRIVE CACHE + dependencies = NA)
# ------------------------------------------------------------------------------
R_SETUP_SCRIPT = """
options(warn = 1)
cat("==============================================================================\\n")
cat("[INFO] Configuring Persistent Drive R Cache & Posit Linux Binary Repo...\\n")
cat("==============================================================================\\n")

# 1. Mount persistent R library cache on Google Drive so packages survive disconnects
drive_r_lib <- "/content/drive/MyDrive/colab_pcap_folder/r_libs"
if (dir.exists(drive_r_lib)) {
  .libPaths(unique(c(drive_r_lib, .libPaths())))
  cat(sprintf("[INFO] Persistent Drive R library active: %s\\n", drive_r_lib))
}

# 2. CRITICAL: Set HTTPUserAgent so Posit Package Manager serves pre-compiled
#    Linux binaries instead of C++ source tarballs!
options(
  HTTPUserAgent = sprintf(
    "R/%s R (%s)",
    getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
  )
)

os_codename <- "jammy"
if (file.exists("/etc/os-release")) {
  os_lines <- readLines("/etc/os-release", warn = FALSE)
  code_line <- grep("^VERSION_CODENAME=", os_lines, value = TRUE)
  if (length(code_line) > 0) {
    os_codename <- sub("^VERSION_CODENAME=", "", gsub('"', '', code_line[1]))
  }
}
rspm_url <- sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", os_codename)
cat(sprintf("[INFO] Using Posit Binary Repo (%s) with binary User-Agent.\\n", os_codename))

options(
  repos = c(
    RSPM = rspm_url,
    CRAN = "https://cloud.r-project.org"
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
  "tm",
  "SnowballC",
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
  cat(sprintf("[INFO] Installing %d remaining R packages from binary repo: %s\\n",
              length(missing_pkgs), paste(missing_pkgs, collapse = ", ")))
  # CRITICAL: Use dependencies = NA (Depends/Imports/LinkingTo only), NOT TRUE!
  install.packages(missing_pkgs, lib = .libPaths()[1], dependencies = NA, quiet = TRUE)
} else {
  cat("[INFO] All 15 required R packages are already cached and ready!\\n")
}

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
    import rpy2.robjects as robjects
    robjects.r(R_SETUP_SCRIPT)

    if not HAS_GPU:
        print("\n" + "!" * 78)
        print("[REMINDER FOR STEPS 02 & 03 — GPU REQUIRED]")
        print("  You are currently on a CPU runtime (Python GPU wheels were skipped).")
        print("  After Step 01 finishes, switch your Colab runtime to a GPU (T4/L4/A100)")
        print("  and RE-RUN THIS STEP 00 CELL before running Steps 02 and 03!")
        print("  (Your R packages are cached in Drive r_libs/, so re-running will be fast.)")
        print("!" * 78)
