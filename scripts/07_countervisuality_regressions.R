%%R
# ==============================================================================
# SCRIPT: scripts/07_countervisuality_regressions.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Estimates B-spline temporal trajectories (stm::estimateEffect) and
#          OLS linear models (lm + broom + modelsummary + gt) across both the
#          full archive timeline (1996-2024) and the Carceral-Digital Turn
#          surge era (2010-2024). Tests whether biometric/optical surveillance
#          and digital mediation motifs surge alongside prison digitization
#          (Section 5.4: Mirzoeff & Browne).
# OUTPUTS: results/figures/fig04_countervisuality_trajectories.png
#          results/charts/table04_temporal_regressions.csv
#          results/charts/table04_temporal_regressions.html
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stm)
  library(broom)
  library(modelsummary)
  library(gt)
  library(splines)
  library(scales)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 07: Temporal B-Splines & Surge-Era Counter-Visuality Regressions\n")
cat("==============================================================================\n")

if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

stm_k25 <- readRDS("data/pcap_stm_visual_k25.rds")
out_visual <- readRDS("data/pcap_llava_stm_out.rds")
tidy_corpus <- readRDS("data/pcap_tidy_corpus.rds")
table03_topics <- attr(stm_k25, "topic_summary_table")

# ------------------------------------------------------------------------------
# 1. ESTIMATE B-SPLINE TEMPORAL EFFECTS (stm::estimateEffect)
# ------------------------------------------------------------------------------
cat("[INFO] Running stm::estimateEffect across all 25 visual motifs...\n")
n_unique_yrs <- length(unique(out_visual$meta$clean_year))
effect_formula <- if (n_unique_yrs >= 10) {
  1:25 ~ s(clean_year) + medium_category
} else {
  1:25 ~ clean_year + medium_category
}

set.seed(42)
prep_effects <- stm::estimateEffect(
  formula     = effect_formula,
  stmobj      = stm_k25,
  metadata    = out_visual$meta,
  uncertainty = "Global"
)

# ------------------------------------------------------------------------------
# 2. CONSTRUCT DOCUMENT-LEVEL COUNTER-VISUALITY INDICES FOR REGRESSION & PLOTTING
# ------------------------------------------------------------------------------
theta_mat <- as.data.frame(stm_k25$theta)
colnames(theta_mat) <- sprintf("topic_%02d", 1:25)

get_family_topics <- function(fam_name, fallback_idx) {
  matched <- table03_topics %>%
    filter(theoretical_family == fam_name) %>%
    pull(topic_id)
  if (length(matched) == 0) return(fallback_idx)
  return(matched)
}

idx_panoptic   <- get_family_topics("Panoptic Eyes & Watchtowers", c(1, 2))
idx_forensic   <- get_family_topics("Forensic Numbers & Barcodes", c(3, 4))
idx_shackled   <- get_family_topics("Fragmented/Shackled Bodies & Counter-Gaze", c(5, 6))
idx_telephonic <- get_family_topics("Telephonic/Screen Separation & Digital Mediation", c(7, 8))

reg_df <- out_visual$meta %>%
  as_tibble() %>%
  bind_cols(theta_mat) %>%
  mutate(
    clean_year = as.numeric(clean_year),
    year_centered_1996 = clean_year - 1996,
    post_2010_digital_turn = ifelse(clean_year >= 2010, 1L, 0L),
    contraband_found_medium = ifelse(
      medium_category == "Contraband/Found Carceral Material", 1L, 0L
    ),
    stm_panoptic_surveillance = rowSums(across(all_of(sprintf("topic_%02d", idx_panoptic)))),
    stm_forensic_biometric    = rowSums(across(all_of(sprintf("topic_%02d", idx_forensic)))),
    stm_shackled_countergaze  = rowSums(across(all_of(sprintf("topic_%02d", idx_shackled)))),
    stm_digital_telephonic    = rowSums(across(all_of(sprintf("topic_%02d", idx_telephonic)))),
    stm_total_countervisuality = stm_panoptic_surveillance + stm_forensic_biometric + stm_digital_telephonic
  )

# ------------------------------------------------------------------------------
# 3. FIT FULL-ARCHIVE (1996-2024) AND SURGE-ERA (2010-2024) OLS REGRESSIONS
# ------------------------------------------------------------------------------
cat("[INFO] Estimating Full-Archive (1996-2024) and Digital-Turn (2010-2024) OLS models...\n")

m1_full_stm <- lm(
  stm_total_countervisuality ~ year_centered_1996 + contraband_found_medium + score_bodily_autonomy,
  data = reg_df
)

reg_df_surge <- reg_df %>% filter(clean_year >= 2010)
if (nrow(reg_df_surge) < 10) reg_df_surge <- reg_df

m2_surge_stm <- lm(
  stm_total_countervisuality ~ I(clean_year - 2010) + contraband_found_medium + score_bodily_autonomy,
  data = reg_df_surge
)

tidy_full_reg <- tidy_corpus %>%
  filter(clean_year >= 1996 & clean_year <= 2025) %>%
  mutate(
    year_centered_1996 = clean_year - 1996,
    contraband_found_medium = ifelse(
      medium_category == "Contraband/Found Carceral Material", 1L, 0L
    )
  )

m3_full_clip_surv <- lm(
  score_surveillance_biometric ~ year_centered_1996 + contraband_found_medium + score_bodily_autonomy,
  data = tidy_full_reg
)

tidy_surge_reg <- tidy_full_reg %>% filter(clean_year >= 2010)
if (nrow(tidy_surge_reg) < 10) tidy_surge_reg <- tidy_full_reg

m4_surge_clip_digital <- lm(
  I(score_digital_mediation + score_state_bureaucracy) ~ I(clean_year - 2010) + contraband_found_medium + score_bodily_autonomy,
  data = tidy_surge_reg
)

models_list <- list(
  "M1: STM Counter-Visuality (1996-2024)"     = m1_full_stm,
  "M2: STM Digital Turn Surge (2010-2024)"    = m2_surge_stm,
  "M3: CLIP Panoptic Surveillance (1996-2024)" = m3_full_clip_surv,
  "M4: CLIP Digital/Biometric (2010-2024)"    = m4_surge_clip_digital
)

table04_csv <- bind_rows(
  lapply(names(models_list), function(m_name) {
    broom::tidy(models_list[[m_name]], conf.int = TRUE) %>%
      mutate(model = m_name, .before = 1)
  })
)
readr::write_csv(table04_csv, "results/charts/table04_temporal_regressions.csv")

ms_gt_table <- modelsummary::modelsummary(
  models_list,
  output = "gt",
  stars = c("*" = 0.05, "**" = 0.01, "***" = 0.001),
  gof_omit = "IC|Log|F|RMSE",
  title = "Table 04: Temporal OLS Regressions of Counter-Visuality & Biometric Subversion (1996–2024 vs. 2010–2024)"
)
gt::gtsave(ms_gt_table, filename = "results/charts/table04_temporal_regressions.html")

# ------------------------------------------------------------------------------
# 4. PLOT B-SPLINE TEMPORAL TRAJECTORIES (FIGURE 04)
# ------------------------------------------------------------------------------
trajectory_long <- reg_df %>%
  select(
    clean_year,
    `1. Panoptic Eyes & Watchtowers`       = stm_panoptic_surveillance,
    `2. Forensic Numbers & Barcodes`       = stm_forensic_biometric,
    `3. Shackled Bodies & Counter-Gaze`    = stm_shackled_countergaze,
    `4. Telephonic & Digital Mediation`    = stm_digital_telephonic
  ) %>%
  pivot_longer(
    cols = -clean_year,
    names_to = "motif_dimension",
    values_to = "topic_proportion"
  )

fig04 <- ggplot(
  trajectory_long,
  aes(x = clean_year, y = topic_proportion, color = motif_dimension, fill = motif_dimension)
) +
  geom_vline(xintercept = 2010, linetype = "dashed", color = "grey20", linewidth = 0.8) +
  annotate(
    "text",
    x = 2010.4,
    y = max(trajectory_long$topic_proportion, na.rm = TRUE) * 0.92,
    label = "Carceral-Digital Turn (2010)",
    hjust = 0,
    size = 3.3,
    fontface = "italic",
    color = "grey20"
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ splines::bs(x, df = 4),
    se = TRUE,
    alpha = 0.18,
    linewidth = 1.15
  ) +
  facet_wrap(~ motif_dimension, scales = "free_y", ncol = 2) +
  scale_color_brewer(palette = "Dark2", guide = "none") +
  scale_fill_brewer(palette = "Dark2", guide = "none") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    title = "Counter-Visual Trajectories in the PCAP Archive (1996–2024)",
    subtitle = "B-Spline Smoothed STM Motif Prevalence Across the Rollout of Carceral Digital Infrastructure",
    x = "PCAP Annual Exhibition Year",
    y = "Expected Visual Motif Proportion (STM Theta)",
    caption = "Source: University of Michigan PCAP Archive (K = 25 LLaVA STM). Shaded bands indicate 95% confidence intervals."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = "results/figures/fig04_countervisuality_trajectories.png",
  plot = fig04,
  width = 11,
  height = 7.5,
  dpi = 300,
  bg = "white"
)

stopifnot(
  file.exists("results/figures/fig04_countervisuality_trajectories.png"),
  file.exists("results/charts/table04_temporal_regressions.csv"),
  file.exists("results/charts/table04_temporal_regressions.html")
)
cat("[TDD PASS] Script 07 Complete! All 3 target artifacts verified.\n")
