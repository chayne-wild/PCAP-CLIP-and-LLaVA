%%R
# ==============================================================================
# SCRIPT: scripts/06_stm_visual_motifs_k25.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Trains the deterministic K = 25 Structural Topic Model
#          (init.type = "Spectral") on the cleaned LLaVA iconographic
#          descriptions with prevalence = ~ s(clean_year) + medium_category.
#          Extracts FREX, Highest Prob, Lift, and Score word matrices
#          (labelTopics) and maps the core counter-visual motifs (Section 5.3).
# OUTPUTS: data/pcap_stm_visual_k25.rds
#          results/charts/table03_llava_visual_topics.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stm)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 06: Deterministic K = 25 Spectral STM on LLaVA Visual Motifs\n")
cat("==============================================================================\n")

if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

stm_out_path <- "data/pcap_llava_stm_out.rds"
if (!file.exists(stm_out_path)) {
  stop("[ERROR] Missing data/pcap_llava_stm_out.rds. Run Script 05 first!")
}

out_visual <- readRDS(stm_out_path)
K_TARGET <- 25L

# ------------------------------------------------------------------------------
# 1. FIT DETERMINISTIC SPECTRAL STM (K = 25)
# ------------------------------------------------------------------------------
cat(sprintf("[INFO] Fitting Spectral STM with K = %d on %d artworks...\n",
            K_TARGET, length(out_visual$documents)))

n_unique_yrs <- length(unique(out_visual$meta$clean_year))
prevalence_formula <- if (n_unique_yrs >= 10) {
  ~ s(clean_year) + medium_category
} else {
  ~ clean_year + medium_category
}

set.seed(42)
stm_k25 <- stm::stm(
  documents  = out_visual$documents,
  vocab      = out_visual$vocab,
  K          = K_TARGET,
  prevalence = prevalence_formula,
  data       = out_visual$meta,
  init.type  = "Spectral",
  max.em.its = 150,
  verbose    = FALSE
)

# ------------------------------------------------------------------------------
# 2. EXTRACT FREX, HIGHEST PROB, LIFT, AND SCORE MATRICES
# ------------------------------------------------------------------------------
topic_labels <- stm::labelTopics(stm_k25, n = 10)
theta_means <- colMeans(stm_k25$theta)

collapse_words <- function(mat, row_idx) {
  paste(mat[row_idx, ], collapse = ", ")
}

# Theoretical Anchor Lexicons to map top counter-visual motif categories
motif_anchors <- list(
  "Panoptic Eyes & Watchtowers" = c(
    "eye", "watchtower", "tower", "camera", "gaze", "stare", "spotlight",
    "surveillance", "lens", "watch", "guard", "pupil", "iris"
  ),
  "Forensic Numbers & Barcodes" = c(
    "number", "barcode", "digit", "stamp", "code", "label", "identification",
    "serial", "form", "text", "letter", "grid", "placard", "tally"
  ),
  "Fragmented/Shackled Bodies & Counter-Gaze" = c(
    "chain", "shackle", "handcuff", "bar", "wire", "razor", "cage", "cell",
    "face", "hand", "body", "expression", "defiant", "direct", "tear", "scream"
  ),
  "Pastoral Escape & Tactile Disalienation" = c(
    "tree", "sky", "horizon", "mountain", "water", "bird", "flower", "nature",
    "forest", "sun", "landscape", "cloud", "river", "texture", "impasto", "carve"
  ),
  "Telephonic/Screen Separation & Digital Mediation" = c(
    "phone", "telephone", "screen", "monitor", "glass", "window", "booth",
    "cord", "cable", "digital", "pixel", "electronic", "computer", "visitation"
  )
)

score_motif_match <- function(words_vec, anchor_vec) {
  sum(words_vec %in% anchor_vec)
}

topic_rows <- vector("list", K_TARGET)
for (k in seq_len(K_TARGET)) {
  frex_vec  <- topic_labels$frex[k, ]
  prob_vec  <- topic_labels$prob[k, ]
  combined_top <- unique(c(frex_vec, prob_vec))

  overlap_scores <- vapply(
    motif_anchors,
    function(anchors) score_motif_match(combined_top, anchors),
    numeric(1)
  )
  best_family <- names(which.max(overlap_scores))
  if (max(overlap_scores) == 0) {
    best_family <- "General Carceral Iconography"
  }

  topic_rows[[k]] <- tibble(
    topic_id           = k,
    topic_label        = sprintf("Topic %02d: %s (%s)", k, best_family, paste(frex_vec[1:3], collapse = "/")),
    theoretical_family = best_family,
    anchor_overlap     = max(overlap_scores),
    expected_prop      = round(theta_means[k], 5),
    expected_pct       = round(100 * theta_means[k], 2),
    frex_top10         = collapse_words(topic_labels$frex, k),
    prob_top10         = collapse_words(topic_labels$prob, k),
    lift_top10         = collapse_words(topic_labels$lift, k),
    score_top10        = collapse_words(topic_labels$score, k)
  )
}

table03_topics <- bind_rows(topic_rows) %>%
  arrange(desc(expected_prop))

# Attach topic table metadata inside the saved RDS list for downstream scripts (07-09)
attr(stm_k25, "topic_summary_table") <- table03_topics
saveRDS(stm_k25, "data/pcap_stm_visual_k25.rds")
readr::write_csv(table03_topics, "results/charts/table03_llava_visual_topics.csv")

stopifnot(
  file.exists("data/pcap_stm_visual_k25.rds"),
  file.exists("results/charts/table03_llava_visual_topics.csv"),
  nrow(table03_topics) == 25
)
cat("[TDD PASS] Script 06 Complete! Saved data/pcap_stm_visual_k25.rds & table03_llava_visual_topics.csv.\n")
