%%R
# ==============================================================================
# SCRIPT: scripts/09_qualitative_artwork_dossier.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Executes stm::findThoughts() combined with joint CLIP-LLaVA composite
#          Z-score ranking to extract the top 5 exemplar artworks for each of
#          the 5 target counter-visual motifs. Generates a formatted qualitative
#          dossier containing PCAP Item ID, Title, Exhibition Year, Medium,
#          Security Level, Artist Statement, LLaVA Iconographic Transcript, and
#          Omeka Web URL for close art-historical reading in Section 5.4.
# OUTPUT : results/charts/qualitative_artwork_dossier.txt
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stm)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 09: Joint CLIP-LLaVA Exemplar Extraction & Qualitative Dossier\n")
cat("==============================================================================\n")

if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

stm_k25 <- readRDS("data/pcap_stm_visual_k25.rds")
out_visual <- readRDS("data/pcap_llava_stm_out.rds")
table03_topics <- attr(stm_k25, "topic_summary_table")

meta_df <- out_visual$meta %>% as_tibble()
theta_mat <- stm_k25$theta

# ------------------------------------------------------------------------------
# 1. DEFINE TARGET COUNTER-VISUAL MOTIF FAMILIES & JOINT CLIP-LLAVA SCORING
# ------------------------------------------------------------------------------
target_families <- list(
  list(
    family_name = "Panoptic Eyes & Watchtowers",
    clip_col    = "score_surveillance_biometric",
    fallback_k  = 1L
  ),
  list(
    family_name = "Forensic Numbers & Barcodes",
    clip_col    = "score_state_bureaucracy",
    fallback_k  = 2L
  ),
  list(
    family_name = "Fragmented/Shackled Bodies & Counter-Gaze",
    clip_col    = "score_bodily_autonomy",
    fallback_k  = 3L
  ),
  list(
    family_name = "Pastoral Escape & Tactile Disalienation",
    clip_col    = "score_tactile_disalienation",
    fallback_k  = 4L
  ),
  list(
    family_name = "Telephonic/Screen Separation & Digital Mediation",
    clip_col    = "score_digital_mediation",
    fallback_k  = 5L
  )
)

dossier_path <- "results/charts/qualitative_artwork_dossier.txt"
con <- file(dossier_path, open = "wt", encoding = "UTF-8")

writeLines("================================================================================", con)
writeLines("QUALITATIVE ARTWORK DOSSIER — CHAPTER 5: COUNTER-VISUALITY IN THE PCAP ARCHIVE", con)
writeLines(sprintf("Generated: %s | Joint CLIP (ViT-L/14) + LLaVA (STM K=25) Ranking", Sys.time()), con)
writeLines("================================================================================\n", con)

for (fam in target_families) {
  fam_name <- fam$family_name
  clip_col <- fam$clip_col

  matched_topics <- table03_topics %>%
    filter(theoretical_family == fam_name) %>%
    arrange(desc(anchor_overlap), desc(expected_prop))

  if (nrow(matched_topics) > 0) {
    primary_k <- matched_topics$topic_id[1]
    frex_words <- matched_topics$frex_top10[1]
  } else {
    primary_k <- min(fam$fallback_k, ncol(theta_mat))
    frex_words <- table03_topics %>% filter(topic_id == primary_k) %>% pull(frex_top10)
  }

  # Execute stm::findThoughts() on the primary motif topic to verify STM alignment
  n_thoughts <- min(15L, nrow(meta_df))
  ft_obj <- stm::findThoughts(
    model  = stm_k25,
    texts  = meta_df$llava_description,
    topics = primary_k,
    n      = n_thoughts
  )
  ft_indices <- unlist(ft_obj$index)

  # Compute Joint Composite Score = 0.5 * Z(STM Theta_k) + 0.5 * Z(CLIP Cosine Similarity)
  stm_theta_vec <- theta_mat[, primary_k]
  clip_score_vec <- meta_df[[clip_col]]

  z_stm  <- as.numeric(scale(stm_theta_vec))
  z_clip <- as.numeric(scale(clip_score_vec))
  joint_score <- 0.5 * coalesce(z_stm, 0) + 0.5 * coalesce(z_clip, 0)

  exemplars_df <- meta_df %>%
    mutate(
      doc_idx         = row_number(),
      stm_theta       = round(stm_theta_vec, 4),
      clip_sim        = round(clip_score_vec, 4),
      joint_z         = round(joint_score, 4),
      in_findthoughts = doc_idx %in% ft_indices
    ) %>%
    arrange(desc(in_findthoughts), desc(joint_z)) %>%
    slice_head(n = 5)

  writeLines("--------------------------------------------------------------------------------", con)
  writeLines(sprintf("MOTIF FAMILY : %s", toupper(fam_name)), con)
  writeLines(sprintf("PRIMARY STM  : Topic %02d | Top FREX Terms: %s", primary_k, frex_words), con)
  writeLines(sprintf("CLIP AXIS    : %s", clip_col), con)
  writeLines("--------------------------------------------------------------------------------\n", con)

  for (r in seq_len(nrow(exemplars_df))) {
    ex <- exemplars_df[r, ]
    stmt_str <- ifelse(
      is.na(ex$artist_statement) || nchar(str_trim(ex$artist_statement)) == 0,
      "[No artist statement archived on Omeka S item record]",
      ex$artist_statement
    )

    writeLines(sprintf("  [EXEMPLAR #%d | Joint Z-Score: %+.3f | STM Theta: %.3f | CLIP Sim: %.3f]",
                       r, ex$joint_z, ex$stm_theta, ex$clip_sim), con)
    writeLines(sprintf("  PCAP Item ID     : %s (Log/ID: %s)", ex$item_id, coalesce(ex$identifier, "N/A")), con)
    writeLines(sprintf("  Artwork Title    : %s", coalesce(ex$title, "Untitled")), con)
    writeLines(sprintf("  Artist           : %s", coalesce(ex$artist_name, "Anonymous")), con)
    writeLines(sprintf("  Exhibition Year  : %s (%s)", as.character(ex$clean_year), coalesce(ex$exhibition_source, "PCAP Annual Exhibition")), con)
    writeLines(sprintf("  Material Medium  : %s [Taxonomy: %s]", coalesce(ex$medium_raw, "Unspecified"), as.character(ex$medium_category)), con)
    writeLines(sprintf("  Facility / Level : %s [%s]", coalesce(ex$facility_raw, "Unspecified"), coalesce(ex$security_level, "Unspecified")), con)
    writeLines(sprintf("  Omeka S Web URL  : %s", coalesce(ex$omeka_web_url, "N/A")), con)
    writeLines(sprintf("  Artist Statement : %s", stmt_str), con)
    writeLines(sprintf("  LLaVA Transcript : %s\n", ex$llava_description), con)
  }
}

close(con)

stopifnot(
  file.exists(dossier_path),
  file.info(dossier_path)$size > 200
)
cat(sprintf("[TDD PASS] Script 09 Complete! Qualitative dossier written to %s (%d bytes).\n",
            dossier_path, file.info(dossier_path)$size))
