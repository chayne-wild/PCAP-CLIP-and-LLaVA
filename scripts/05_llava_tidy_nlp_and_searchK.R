%%R
# ==============================================================================
# SCRIPT: scripts/05_llava_tidy_nlp_and_searchK.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Fault-tolerantly ingests data/pcap_llava_interrogations.jsonl, joins
#          with data/pcap_tidy_corpus.rds, executes tidytext tokenization and
#          textstem lemmatization, and applies the "Visual Plural Bypass"
#          stop-word filter both BEFORE and AFTER lemmatization. Builds the stm
#          document-term object (out_visual) and runs searchK across
#          K in {10, 15, 20, 25, 30, 40}.
# OUTPUTS: data/pcap_llava_stm_out.rds
#          data/pcap_llava_searchK.rds
#          results/figures/fig03_llava_stm_searchK.png
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(tidytext)
  library(textstem)
  library(stm)
  library(patchwork)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 05: LLaVA Tidy NLP, Visual Plural Bypass & STM searchK\n")
cat("==============================================================================\n")

if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

jsonl_path <- "data/pcap_llava_interrogations.jsonl"
corpus_rds_path <- "data/pcap_tidy_corpus.rds"

if (!file.exists(jsonl_path) || !file.exists(corpus_rds_path)) {
  stop("[ERROR] Missing data/pcap_llava_interrogations.jsonl or data/pcap_tidy_corpus.rds!")
}

# ------------------------------------------------------------------------------
# 1. FAULT-TOLERANT LINE-BY-LINE JSONL INGESTION
# ------------------------------------------------------------------------------
raw_lines <- readLines(jsonl_path, warn = FALSE, encoding = "UTF-8")
parsed_records <- list()

for (i in seq_along(raw_lines)) {
  ln <- str_trim(raw_lines[[i]])
  if (nchar(ln) == 0) next
  rec <- tryCatch(jsonlite::fromJSON(ln), error = function(e) NULL)
  if (!is.null(rec) && !is.null(rec$item_id) && !is.null(rec$llava_description)) {
    parsed_records[[length(parsed_records) + 1]] <- as_tibble(rec)
  }
}

llava_df <- bind_rows(parsed_records) %>%
  mutate(item_id = as.character(item_id)) %>%
  distinct(item_id, .keep_all = TRUE) %>%
  filter(nchar(str_trim(llava_description)) > 20)

cat(sprintf("[INFO] Fault-tolerantly ingested %d unique LLaVA transcripts.\n", nrow(llava_df)))

tidy_corpus <- readRDS(corpus_rds_path) %>%
  mutate(item_id = as.character(item_id))

joined_df <- llava_df %>%
  select(item_id, sampling_stratum, llava_description) %>%
  inner_join(tidy_corpus, by = "item_id")

stopifnot(nrow(joined_df) > 0)
cat(sprintf("[INFO] Joined LLaVA transcripts with tidy metadata: N = %d artworks.\n", nrow(joined_df)))

# ------------------------------------------------------------------------------
# 2. THE "VISUAL PLURAL BYPASS" STOP-WORD FILTER (PRE- AND POST-LEMMATIZATION)
# ------------------------------------------------------------------------------
visual_vlm_boilerplate <- unique(c(
  "image", "images", "painting", "paintings", "drawing", "drawings",
  "artwork", "artworks", "depict", "depicts", "depicted", "depicting",
  "depiction", "depictions", "show", "shows", "shown", "showing",
  "feature", "features", "featured", "featuring", "background", "backgrounds",
  "foreground", "foregrounds", "color", "colors", "colored", "colour", "colours",
  "style", "styles", "piece", "pieces", "artist", "artists", "viewer", "viewers",
  "composition", "compositions", "overall", "appear", "appears", "appeared",
  "appearing", "visible", "create", "creates", "created", "creating",
  "sense", "senses", "convey", "conveys", "conveyed", "conveying",
  "suggest", "suggests", "suggested", "suggesting", "scene", "scenes",
  "element", "elements", "detail", "details", "detailed", "include", "includes",
  "included", "including", "seen", "see", "visual", "visually", "form", "forms",
  "formal", "part", "parts", "section", "sections", "aspect", "aspects",
  "subject", "subjects", "medium", "media", "work", "works", "art", "picture",
  "pictures", "illustration", "illustrations", "portrait", "portraits",
  "first", "second", "third", "fourth", "one", "two", "three", "four",
  "left", "right", "center", "central", "top", "bottom", "side", "sides",
  "middle", "area", "areas", "space", "spaces", "use", "uses", "used", "using"
))

data("stop_words", package = "tidytext")
master_stop_vocab <- unique(c(stop_words$word, visual_vlm_boilerplate))

# TDD Self-Test: Verify Visual Plural Bypass eliminates both "paintings" and "depicting"
test_tokens <- tibble(word = c("paintings", "depicting", "watchtowers", "barcodes")) %>%
  filter(!word %in% master_stop_vocab) %>%
  mutate(lemma = textstem::lemmatize_words(word)) %>%
  filter(!lemma %in% master_stop_vocab)

stopifnot(
  !("painting" %in% test_tokens$lemma),
  !("depict" %in% test_tokens$lemma),
  all(c("watchtower", "barcode") %in% test_tokens$lemma)
)
cat("[TDD PASS] 'Visual Plural Bypass' pre- and post-lemmatization stop-word gate verified.\n")

# Apply tokenization -> Pre-Filter -> Lemmatization -> Post-Filter
cleaned_tokens <- joined_df %>%
  select(item_id, llava_description) %>%
  unnest_tokens(output = word, input = llava_description, token = "words") %>%
  filter(str_detect(word, "^[a-z]{3,}$")) %>%
  # Gate 1: Pre-lemmatization filter (catches plural/inflected VLM boilerplate)
  filter(!word %in% master_stop_vocab) %>%
  mutate(lemma = textstem::lemmatize_words(word)) %>%
  # Gate 2: Post-lemmatization filter (catches normalized stems)
  filter(!lemma %in% master_stop_vocab) %>%
  filter(nchar(lemma) >= 3)

reconstructed_docs <- cleaned_tokens %>%
  group_by(item_id) %>%
  summarise(cleaned_text = paste(lemma, collapse = " "), .groups = "drop")

stm_input_df <- joined_df %>%
  inner_join(reconstructed_docs, by = "item_id") %>%
  filter(nchar(cleaned_text) > 10)

# ------------------------------------------------------------------------------
# 3. BUILD STM CORPUS OBJECT (out_visual)
# ------------------------------------------------------------------------------
cat("[INFO] Building STM document-term matrix via textProcessor & prepDocuments...\n")
processed <- stm::textProcessor(
  documents = stm_input_df$cleaned_text,
  metadata = stm_input_df,
  lowercase = FALSE,
  removestopwords = FALSE,
  removenumbers = TRUE,
  removepunctuation = FALSE,
  stem = FALSE,
  verbose = FALSE
)

lower_thresh_val <- ifelse(length(processed$documents) >= 200, 3L, 1L)
out_visual <- stm::prepDocuments(
  documents = processed$documents,
  vocab = processed$vocab,
  meta = processed$meta,
  lower.thresh = lower_thresh_val,
  verbose = FALSE
)

saveRDS(out_visual, "data/pcap_llava_stm_out.rds")
cat(sprintf(
  "[TDD PASS] Saved data/pcap_llava_stm_out.rds (%d documents, %d vocabulary terms).\n",
  length(out_visual$documents), length(out_visual$vocab)
))

# ------------------------------------------------------------------------------
# 4. RUN STM searchK DIAGNOSTICS ACROSS K IN {10, 15, 20, 25, 30, 40}
# ------------------------------------------------------------------------------
candidate_K <- c(10, 15, 20, 25, 30, 40)
cat(sprintf("[INFO] Running stm::searchK across K = {%s}...\n",
            paste(candidate_K, collapse = ", ")))

# Adapt B-spline degrees of freedom safely if running on a small pilot subset
n_unique_yrs <- length(unique(out_visual$meta$clean_year))
prevalence_formula <- if (n_unique_yrs >= 10) {
  ~ s(clean_year) + medium_category
} else {
  ~ clean_year + medium_category
}

set.seed(42)
searchk_fit <- stm::searchK(
  documents = out_visual$documents,
  vocab = out_visual$vocab,
  K = candidate_K,
  prevalence = prevalence_formula,
  data = out_visual$meta,
  init.type = "Spectral",
  verbose = FALSE
)

saveRDS(searchk_fit, "data/pcap_llava_searchK.rds")

searchk_metrics <- tibble(
  K        = unlist(searchk_fit$results$K),
  heldout  = unlist(searchk_fit$results$heldout),
  residual = unlist(searchk_fit$results$residual),
  semcoh   = unlist(searchk_fit$results$semcoh),
  exclus   = unlist(searchk_fit$results$exclus)
)

# ------------------------------------------------------------------------------
# 5. PLOT 4-PANEL searchK DIAGNOSTIC FIGURE (FIGURE 03)
# ------------------------------------------------------------------------------
make_diag_panel <- function(df, y_col, title_str, y_lab, color_hex) {
  ggplot(df, aes(x = K, y = .data[[y_col]])) +
    geom_vline(xintercept = 25, linetype = "dashed", color = "#B22222", linewidth = 0.8) +
    geom_line(color = color_hex, linewidth = 1.1) +
    geom_point(size = 3, color = color_hex) +
    scale_x_continuous(breaks = candidate_K) +
    labs(title = title_str, x = "Number of Topics (K)", y = y_lab) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11.5))
}

p1 <- make_diag_panel(searchk_metrics, "heldout",  "A. Held-Out Likelihood", "Log-Likelihood", "#1B9E77")
p2 <- make_diag_panel(searchk_metrics, "exclus",   "B. Topic Exclusivity",   "FREX Exclusivity", "#D95F02")
p3 <- make_diag_panel(searchk_metrics, "semcoh",   "C. Semantic Coherence",  "Coherence Score", "#7570B3")
p4 <- make_diag_panel(searchk_metrics, "residual", "D. Dispersion Residuals", "Residual Variance", "#E7298A")

fig03 <- (p1 + p2) / (p3 + p4) +
  plot_annotation(
    title = "Structural Topic Model (STM) Hyperparameter Validation on LLaVA Visual Descriptions",
    subtitle = "Spectral Initialization across K in {10, 15, 20, 25, 30, 40}; Red dashed line indicates target K = 25",
    caption = "Source: PCAP Stratified High-Signal LLaVA Subsample (Visual Plural Bypass Filtered)."
  )

ggsave(
  filename = "results/figures/fig03_llava_stm_searchK.png",
  plot = fig03,
  width = 10,
  height = 7,
  dpi = 300,
  bg = "white"
)

stopifnot(
  file.exists("data/pcap_llava_stm_out.rds"),
  file.exists("data/pcap_llava_searchK.rds"),
  file.exists("results/figures/fig03_llava_stm_searchK.png")
)
cat("[TDD PASS] Script 05 Complete! All 3 target artifacts verified.\n")
