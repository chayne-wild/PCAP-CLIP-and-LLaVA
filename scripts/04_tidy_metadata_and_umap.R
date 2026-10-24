%%R
# ==============================================================================
# SCRIPT: scripts/04_tidy_metadata_and_umap.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Phase 2 R/Tidyverse Entrypoint. Ingests data/pcap_clip_embeddings.parquet
#          natively via arrow::read_parquet(). Cleans exhibition years, parses
#          Michigan DOC security levels, classifies raw medium strings into the
#          4-part carceral materiality taxonomy (Section 5.1), runs UMAP manifold
#          projection (uwot::umap) and K-means clustering on the 768-D CLIP space
#          (Section 5.2), and generates Figures 01 & 02 and Tables 01 & 02.
# OUTPUTS: data/pcap_tidy_corpus.rds
#          results/charts/table01_pcap_medium_taxonomy.csv
#          results/charts/table02_cluster_prompt_profiles.csv
#          results/figures/fig01_disalienation_by_medium.png
#          results/figures/fig02_umap_visual_clusters.png
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(uwot)
  library(cluster)
  library(patchwork)
  library(scales)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 04: Tidy Metadata Normalization, Disalienation Test & CLIP UMAP\n")
cat("==============================================================================\n")

# ------------------------------------------------------------------------------
# 1. DIRECTORY & INPUT VERIFICATION
# ------------------------------------------------------------------------------
if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

parquet_path <- "data/pcap_clip_embeddings.parquet"
if (!file.exists(parquet_path)) {
  stop(sprintf("[ERROR] Missing input file: %s. Run Python Scripts 01-02 first!", parquet_path))
}

dir.create("data", showWarnings = FALSE, recursive = TRUE)
dir.create("results/charts", showWarnings = FALSE, recursive = TRUE)
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)

raw_clip_df <- arrow::read_parquet(parquet_path) %>% as_tibble()
cat(sprintf("[INFO] Ingested %d artworks and %d columns from %s.\n",
            nrow(raw_clip_df), ncol(raw_clip_df), parquet_path))

# ------------------------------------------------------------------------------
# 2. METADATA NORMALIZATION FUNCTIONS (YEAR, MEDIUM, SECURITY LEVEL)
# ------------------------------------------------------------------------------
# 2A. Exhibition Year Parser: Extracts 4-digit year (1996-2026) from raw_date,
#     or computes (1995 + N) from "Nth Annual Exhibition" in exhibition_source,
#     or parses the "-YY" suffix on PCAP Exhibition Log Numbers.
parse_exhibition_year <- function(raw_date, exhibition_source, identifier) {
  yr_direct <- str_extract(coalesce(raw_date, ""), "\\b(199[6-9]|20[0-2][0-9])\\b") %>%
    as.integer()

  ex_num <- str_match(
    tolower(coalesce(exhibition_source, "")),
    "\\b([0-9]{1,2})(?:st|nd|rd|th)\\s+annual\\s+exhibition"
  )[, 2] %>% as.integer()
  yr_from_ex <- ifelse(!is.na(ex_num) & ex_num >= 1 & ex_num <= 31, 1995L + ex_num, NA_integer_)

  log_yy <- str_match(coalesce(identifier, ""), "-[0-9]{2}$")[, 1] %>%
    str_remove("-") %>%
    as.integer()
  yr_from_log <- case_when(
    !is.na(log_yy) & log_yy >= 96 & log_yy <= 99 ~ 1900L + log_yy,
    !is.na(log_yy) & log_yy >= 0  & log_yy <= 26 ~ 2000L + log_yy,
    TRUE ~ NA_integer_
  )

  clean_yr <- coalesce(yr_direct, yr_from_ex, yr_from_log)
  clean_yr <- ifelse(clean_yr < 1996L | clean_yr > 2026L, NA_integer_, clean_yr)
  return(clean_yr)
}

# 2B. Material Medium Taxonomy (Section 5.1: Fanon/Tosquelles & Art Brut)
classify_carceral_medium <- function(medium_raw) {
  m_low <- tolower(coalesce(medium_raw, ""))

  case_when(
    # 1. Contraband / Found / Salvaged Carceral Substrates
    str_detect(
      m_low,
      paste(
        c("soap", "toilet paper", "tissue", "state form", "institutional form",
          "m&m", "candy wrapper", "wrapper", "cardboard", "corrugated",
          "matchstick", "match", "toothpaste", "coffee", "tea", "kool-aid",
          "bedsheet", "sheet", "pillowcase", "handkerchief", "sock", "jacket",
          "uniform", "deconstructed", "envelope", "newspaper", "book page",
          "floor wax", "deodorant", "salvaged", "found", "contraband",
          "bread", "foil", "playing card", "dried flower", "ash", "styrofoam"),
        collapse = "|"
      )
    ) ~ "Contraband/Found Carceral Material",

    # 2. Mixed Media & Sculptural / 3D Assemblage
    str_detect(
      m_low,
      paste(
        c("mixed media", "mixed", "collage", "sculpture", "ceramic", "clay",
          "wood", "carving", "assemblage", "papier", "paper mache", "origami",
          "3d", "diorama", "textile", "yarn", "thread", "bead", "mod podge"),
        collapse = "|"
      )
    ) ~ "Mixed/Sculptural",

    # 3. Graphite / Ink / Dry Drawing Media on Paper
    str_detect(
      m_low,
      paste(
        c("ballpoint", "pen", "ink", "graphite", "pencil", "charcoal",
          "colored pencil", "marker", "sharpie", "crayon", "chalk"),
        collapse = "|"
      )
    ) & !str_detect(m_low, "acrylic|oil|watercolor|gouache|tempera|canvas") ~ "Graphite/Ink on Paper",

    # 4. Standard Studio Paint & Canvas / Pastel
    str_detect(
      m_low,
      paste(
        c("acrylic", "oil", "watercolor", "gouache", "pastel", "tempera",
          "paint", "canvas"),
        collapse = "|"
      )
    ) ~ "Standard Paint/Canvas",

    # Fallback default for unspecified drawing/painting records
    TRUE ~ "Standard Paint/Canvas"
  )
}

# 2C. Michigan DOC Security Level Parser
parse_security_level <- function(facility_raw) {
  f_low <- tolower(coalesce(facility_raw, ""))
  case_when(
    f_low == "" ~ "Unspecified MDOC Facility",
    str_detect(f_low, "max|marquette|ionia|baraga|st\\. louis|algonac|standish|huron valley|reformatory") ~
      "Level IV-V (High / Maximum)",
    str_detect(f_low, "macomb|chippewa|kinross|muskegon|saginaw|carson city|bellamy|gus harrison|oaks|thumb|handlon|lakeland|g\\. robert cotton") ~
      "Level II-IV (Multi-Level Security)",
    str_detect(f_low, "parnall|cooper street|central michigan|earnest c\\. brooks|newberry|ojibway|camp") ~
      "Level I-II (Minimum / Low-Medium)",
    TRUE ~ "Level II-IV (Multi-Level Security)"
  )
}

# TDD Gate: Verify medium & year classifiers on edge cases before corpus application
stopifnot(
  classify_carceral_medium("Carved Ivory soap and toilet paper with ballpoint pen") ==
    "Contraband/Found Carceral Material",
  classify_carceral_medium("deconstructed and reconstructed jackets: fabric, acrylic paint and ink") ==
    "Contraband/Found Carceral Material",
  classify_carceral_medium("Ballpoint pen and graphite pencil on paper") ==
    "Graphite/Ink on Paper",
  classify_carceral_medium("Acrylic and oil on canvas") ==
    "Standard Paint/Canvas",
  parse_exhibition_year("", "26th Annual Exhibition of Art by Michigan Prisoners", "") == 2021L
)
cat("[TDD PASS] Metadata normalization helper functions verified.\n")

# ------------------------------------------------------------------------------
# 3. APPLY TIDY NORMALIZATION & COMPUTE DISALIENATION METRICS (SECTION 5.1)
# ------------------------------------------------------------------------------
metadata_df <- raw_clip_df %>%
  select(-starts_with("clip_")) %>%
  mutate(
    item_id = as.character(item_id),
    clean_year = parse_exhibition_year(raw_date, exhibition_source, identifier),
    year_imputed = is.na(clean_year),
    clean_year = coalesce(clean_year, median(clean_year, na.rm = TRUE) %>% as.integer(), 2012L),
    era_window = ifelse(clean_year >= 2010L, "Carceral-Digital Turn (2010-2024)", "Pre-Digital Era (1996-2009)"),
    medium_category = factor(
      classify_carceral_medium(medium_raw),
      levels = c(
        "Contraband/Found Carceral Material",
        "Mixed/Sculptural",
        "Graphite/Ink on Paper",
        "Standard Paint/Canvas"
      )
    ),
    security_level = parse_security_level(facility_raw),
    # Section 5.1 Theoretical Metric: Net Tactile Disalienation vs. Institutional Enclosure
    disalienation_index = as.numeric(
      scale(score_tactile_disalienation + 0.5 * score_bodily_autonomy) -
      scale(score_carceral_containment + 0.5 * score_state_bureaucracy)
    )
  )

# ------------------------------------------------------------------------------
# 4. UMAP MANIFOLD PROJECTION & K-MEANS VISUAL CLUSTERING (SECTION 5.2)
# ------------------------------------------------------------------------------
cat("[INFO] Extracting 768-D CLIP matrix for UMAP manifold projection...\n")
clip_mat <- raw_clip_df %>%
  select(starts_with("clip_")) %>%
  as.matrix()

stopifnot(ncol(clip_mat) == 768)

set.seed(42)
n_rows <- nrow(clip_mat)
n_neighbors_val <- min(30L, max(5L, floor(n_rows / 4)))

cat(sprintf("[INFO] Running uwot::umap on %d x 768 matrix (n_neighbors=%d, metric='cosine')...\n",
            n_rows, n_neighbors_val))
umap_coords <- uwot::umap(
  clip_mat,
  n_neighbors = n_neighbors_val,
  n_components = 2,
  min_dist = 0.12,
  metric = "cosine",
  n_threads = parallel::detectCores(),
  verbose = FALSE
)

# Assign K = 6 Macro-Visual Clusters on the CLIP representation
k_clusters <- min(6L, max(2L, floor(n_rows / 10)))
set.seed(42)
km_fit <- kmeans(clip_mat, centers = k_clusters, nstart = 20, iter.max = 100)

tidy_corpus <- metadata_df %>%
  mutate(
    umap_1 = umap_coords[, 1],
    umap_2 = umap_coords[, 2],
    visual_cluster_id = km_fit$cluster
  )

cluster_profiles <- tidy_corpus %>%
  group_by(visual_cluster_id) %>%
  summarise(
    n_artworks = n(),
    share_pct = round(100 * n() / nrow(tidy_corpus), 2),
    mean_surveillance_biometric = mean(score_surveillance_biometric, na.rm = TRUE),
    mean_carceral_containment   = mean(score_carceral_containment, na.rm = TRUE),
    mean_bodily_autonomy        = mean(score_bodily_autonomy, na.rm = TRUE),
    mean_tactile_disalienation  = mean(score_tactile_disalienation, na.rm = TRUE),
    mean_digital_mediation      = mean(score_digital_mediation, na.rm = TRUE),
    mean_state_bureaucracy      = mean(score_state_bureaucracy, na.rm = TRUE),
    mean_disalienation_index    = mean(disalienation_index, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    visual_cluster = sprintf("Cluster %d", visual_cluster_id)
  )

tidy_corpus <- tidy_corpus %>%
  left_join(
    cluster_profiles %>% select(visual_cluster_id, visual_cluster),
    by = "visual_cluster_id"
  )

saveRDS(tidy_corpus, "data/pcap_tidy_corpus.rds")
readr::write_csv(cluster_profiles, "results/charts/table02_cluster_prompt_profiles.csv")
cat("[TDD PASS] Saved data/pcap_tidy_corpus.rds and results/charts/table02_cluster_prompt_profiles.csv\n")

# ------------------------------------------------------------------------------
# 5. SECTION 5.1 TAXONOMY TABLE & HYPOTHESIS TEST (TABLE 01 & FIGURE 01)
# ------------------------------------------------------------------------------
kw_test <- kruskal.test(score_tactile_disalienation ~ medium_category, data = tidy_corpus)
cat(sprintf("[INFO] Section 5.1 Kruskal-Wallis Test across Mediums: H = %.3f, p = %.4e\n",
            kw_test$statistic, kw_test$p.value))

table01_medium_taxonomy <- tidy_corpus %>%
  group_by(medium_category) %>%
  summarise(
    n_artworks = n(),
    corpus_percentage = round(100 * n() / nrow(tidy_corpus), 2),
    mean_tactile_disalienation = round(mean(score_tactile_disalienation, na.rm = TRUE), 4),
    sd_tactile_disalienation   = round(sd(score_tactile_disalienation, na.rm = TRUE), 4),
    mean_bodily_autonomy       = round(mean(score_bodily_autonomy, na.rm = TRUE), 4),
    mean_surveillance_score    = round(mean(score_surveillance_biometric, na.rm = TRUE), 4),
    mean_bureaucracy_score     = round(mean(score_state_bureaucracy, na.rm = TRUE), 4),
    mean_disalienation_index   = round(mean(disalienation_index, na.rm = TRUE), 4),
    kruskal_wallis_p           = signif(kw_test$p.value, 4),
    .groups = "drop"
  )

readr::write_csv(table01_medium_taxonomy, "results/charts/table01_pcap_medium_taxonomy.csv")
cat("[TDD PASS] Saved results/charts/table01_pcap_medium_taxonomy.csv\n")

# Figure 01: Disalienation & Tactile Vitality by Carceral Material Medium
fig01 <- ggplot(
  tidy_corpus,
  aes(x = fct_rev(medium_category), y = score_tactile_disalienation, fill = medium_category)
) +
  geom_violin(alpha = 0.45, color = NA, trim = FALSE) +
  geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.85, color = "grey15") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Contraband/Found Carceral Material" = "#B22222",
      "Mixed/Sculptural"                   = "#D95F02",
      "Graphite/Ink on Paper"              = "#7570B3",
      "Standard Paint/Canvas"              = "#1B9E77"
    ),
    guide = "none"
  ) +
  labs(
    title = "Aesthetics of Disalienation Across Carceral Material Substrates",
    subtitle = sprintf(
      "CLIP Zero-Shot Tactile Vitality / Disalienation Cosine Similarity (N = %s; Kruskal-Wallis p = %.2e)",
      scales::comma(nrow(tidy_corpus)), kw_test$p.value
    ),
    x = "Material Medium Taxonomy (Section 5.1)",
    y = "CLIP Cosine Similarity: Tactile Disalienation & Sensory Vitality",
    caption = "Source: University of Michigan PCAP Archive (ViT-L/14 Distant Viewing). White diamonds indicate group means."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey30", size = 10.5),
    axis.text.y = element_text(face = "bold", size = 10.5),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = "results/figures/fig01_disalienation_by_medium.png",
  plot = fig01,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)

# Figure 02: UMAP Manifold Projection of the PCAP Visual Corpus
fig02 <- ggplot(
  tidy_corpus,
  aes(x = umap_1, y = umap_2, color = visual_cluster)
) +
  geom_point(alpha = 0.55, size = 1.1) +
  scale_color_brewer(palette = "Dark2", name = "Visual Cluster") +
  labs(
    title = "Distant Viewing the PCAP Corpus: 768-D CLIP UMAP Manifold",
    subtitle = sprintf(
      "2D UMAP Projection of ViT-L/14 Embeddings Across %s Incarcerated Artworks",
      scales::comma(nrow(tidy_corpus))
    ),
    x = "UMAP Dimension 1 (Cosine Metric)",
    y = "UMAP Dimension 2 (Cosine Metric)",
    caption = "Source: University of Michigan PCAP Visual Art Database (Arnold & Tilton Distant Viewing Framework)."
  ) +
  guides(color = guide_legend(override.aes = list(size = 3.5, alpha = 0.95))) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey30", size = 10.5),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = "results/figures/fig02_umap_visual_clusters.png",
  plot = fig02,
  width = 10,
  height = 7,
  dpi = 300,
  bg = "white"
)

stopifnot(
  file.exists("data/pcap_tidy_corpus.rds"),
  file.exists("results/charts/table01_pcap_medium_taxonomy.csv"),
  file.exists("results/charts/table02_cluster_prompt_profiles.csv"),
  file.exists("results/figures/fig01_disalienation_by_medium.png"),
  file.exists("results/figures/fig02_umap_visual_clusters.png")
)
cat("[TDD PASS] Script 04 Complete! All 5 target artifacts verified.\n")
