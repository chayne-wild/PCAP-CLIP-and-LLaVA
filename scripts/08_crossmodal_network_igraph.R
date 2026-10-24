%%R
# ==============================================================================
# SCRIPT: scripts/08_crossmodal_network_igraph.R
# CHAPTER: Chapter 5 — Cultural Resistance: Counter-Visuality in the PCAP Archive
# PURPOSE: Computes the topic correlation matrix (stm::topicCorr) across the
#          25 LLaVA visual motifs and integrates CLIP visual clusters and
#          carceral material medium covariates into a cross-modal igraph
#          network visualization, testing how material scarcity (found/contraband
#          substrates) structurally links to counter-visual themes.
# OUTPUTS: data/pcap_visual_topic_corr.rds
#          results/figures/fig05_crossmodal_network_igraph.png
#          results/charts/table05_material_motif_correlations.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stm)
  library(igraph)
})

cat("==============================================================================\n")
cat("[INFO] SCRIPT 08: Cross-Modal Network Topology (LLaVA Motifs x Mediums x CLIP)\n")
cat("==============================================================================\n")

if (dir.exists("/content/drive/MyDrive/colab_pcap_folder")) {
  setwd("/content/drive/MyDrive/colab_pcap_folder")
}

stm_k25 <- readRDS("data/pcap_stm_visual_k25.rds")
out_visual <- readRDS("data/pcap_llava_stm_out.rds")
table03_topics <- attr(stm_k25, "topic_summary_table")

# ------------------------------------------------------------------------------
# 1. COMPUTE STM TOPIC CORRELATION MATRIX (stm::topicCorr)
# ------------------------------------------------------------------------------
cat("[INFO] Computing stm::topicCorr across all 25 visual motifs...\n")
topic_corr_obj <- stm::topicCorr(stm_k25, method = "simple", cutoff = 0.05)
saveRDS(topic_corr_obj, "data/pcap_visual_topic_corr.rds")

# ------------------------------------------------------------------------------
# 2. COMPUTE CROSS-MODAL CORRELATIONS (MOTIFS x MEDIUMS x CLIP CLUSTERS)
# ------------------------------------------------------------------------------
theta_df <- as.data.frame(stm_k25$theta)
colnames(theta_df) <- sprintf("Topic_%02d", 1:25)

meta_df <- out_visual$meta %>% as_tibble()

medium_levels <- unique(as.character(meta_df$medium_category))
cluster_levels <- unique(as.character(meta_df$visual_cluster))

medium_mat <- sapply(medium_levels, function(m) as.numeric(meta_df$medium_category == m))
if (!is.matrix(medium_mat)) medium_mat <- matrix(medium_mat, ncol = length(medium_levels))
colnames(medium_mat) <- paste0("Medium: ", medium_levels)

cluster_mat <- sapply(cluster_levels, function(cl) as.numeric(meta_df$visual_cluster == cl))
if (!is.matrix(cluster_mat)) cluster_mat <- matrix(cluster_mat, ncol = length(cluster_levels))
colnames(cluster_mat) <- paste0("CLIP ", cluster_levels)

cov_mat <- cbind(medium_mat, cluster_mat)
cross_corr_edges <- list()

for (k in 1:25) {
  t_vec <- theta_df[[k]]
  t_info <- table03_topics %>% filter(topic_id == k)
  t_short <- sprintf("T%02d (%s)", k, str_extract(t_info$frex_top10, "^[^,]+"))

  for (c_idx in seq_len(ncol(cov_mat))) {
    cov_name <- colnames(cov_mat)[c_idx]
    c_vec <- cov_mat[, c_idx]
    if (sd(t_vec) > 0 && sd(c_vec) > 0) {
      ct <- suppressWarnings(cor.test(t_vec, c_vec))
      cross_corr_edges[[length(cross_corr_edges) + 1]] <- tibble(
        from_node          = t_short,
        to_node            = cov_name,
        topic_id           = k,
        theoretical_family = t_info$theoretical_family,
        covariate_type     = ifelse(str_starts(cov_name, "Medium:"), "Material Medium", "CLIP Cluster"),
        pearson_r          = round(as.numeric(ct$estimate), 4),
        p_value            = signif(ct$p.value, 4)
      )
    }
  }
}

table05_correlations <- bind_rows(cross_corr_edges) %>%
  arrange(desc(abs(pearson_r)))

readr::write_csv(table05_correlations, "results/charts/table05_material_motif_correlations.csv")
cat("[TDD PASS] Saved results/charts/table05_material_motif_correlations.csv\n")

# ------------------------------------------------------------------------------
# 3. BUILD & RENDER MULTIPARTITE IGRAPH NETWORK (FIGURE 05)
# ------------------------------------------------------------------------------
corr_mat <- topic_corr_obj$cor
colnames(corr_mat) <- rownames(corr_mat) <- vapply(1:25, function(k) {
  t_info <- table03_topics %>% filter(topic_id == k)
  sprintf("T%02d (%s)", k, str_extract(t_info$frex_top10, "^[^,]+"))
}, character(1))

topic_edges <- as_tibble(as.table(corr_mat)) %>%
  transmute(
    from_node = as.character(Var1),
    to_node   = as.character(Var2),
    weight    = as.numeric(n)
  ) %>%
  filter(from_node < to_node, weight > 0.06) %>%
  mutate(edge_type = "Topic-Topic Covariance")

cross_edges_filtered <- table05_correlations %>%
  filter(pearson_r > 0.03) %>%
  slice_head(n = 35) %>%
  transmute(
    from_node = as.character(from_node),
    to_node   = as.character(to_node),
    weight    = abs(pearson_r),
    edge_type = "Cross-Modal Link"
  )

all_edges <- bind_rows(topic_edges, cross_edges_filtered)
if (nrow(all_edges) == 0) {
  all_edges <- table05_correlations %>%
    slice_head(n = 20) %>%
    transmute(
      from_node = as.character(from_node),
      to_node   = as.character(to_node),
      weight    = pmax(abs(pearson_r), 0.05),
      edge_type = "Cross-Modal Link"
    )
}

g <- igraph::graph_from_data_frame(all_edges, directed = FALSE)

node_names <- V(g)$name
V(g)$modality <- case_when(
  str_starts(node_names, "Medium:") ~ "Carceral Material Medium",
  str_starts(node_names, "CLIP ")   ~ "CLIP Macro-Visual Cluster",
  TRUE                              ~ "LLaVA Iconographic Motif (STM)"
)

V(g)$color <- case_when(
  V(g)$modality == "Carceral Material Medium"       ~ "#B22222",
  V(g)$modality == "CLIP Macro-Visual Cluster"      ~ "#1B9E77",
  TRUE                                              ~ "#386CB0"
)

V(g)$size <- case_when(
  V(g)$modality == "Carceral Material Medium"       ~ 13,
  V(g)$modality == "CLIP Macro-Visual Cluster"      ~ 11,
  TRUE                                              ~ 8.5
)

set.seed(42)
layout_fr <- igraph::layout_with_fr(g, niter = 1000)

png(
  filename = "results/figures/fig05_crossmodal_network_igraph.png",
  width = 3200,
  height = 2400,
  res = 300,
  bg = "white"
)
par(mar = c(2, 2, 4, 2))
plot(
  g,
  layout = layout_fr,
  vertex.color = V(g)$color,
  vertex.frame.color = "white",
  vertex.size = V(g)$size,
  vertex.label.color = "grey10",
  vertex.label.cex = 0.72,
  vertex.label.dist = 0.65,
  vertex.label.family = "sans",
  edge.width = scales::rescale(E(g)$weight, to = c(1.0, 4.5)),
  edge.color = ifelse(E(g)$edge_type == "Cross-Modal Link", "#D95F0288", "#88888866"),
  main = "Cross-Modal Counter-Visuality Network: LLaVA Motifs, Carceral Mediums & CLIP Clusters"
)
legend(
  "bottomleft",
  legend = c(
    "Carceral Material Medium (Section 5.1)",
    "CLIP Visual Cluster (Section 5.2)",
    "LLaVA STM Visual Motif (Section 5.3)"
  ),
  pt.bg  = c("#B22222", "#1B9E77", "#386CB0"),
  pch    = 21,
  pt.cex = 2.0,
  cex    = 0.85,
  bty    = "n"
)
dev.off()

stopifnot(
  file.exists("data/pcap_visual_topic_corr.rds"),
  file.exists("results/figures/fig05_crossmodal_network_igraph.png"),
  file.exists("results/charts/table05_material_motif_correlations.csv")
)
cat("[TDD PASS] Script 08 Complete! All 3 target artifacts verified.\n")
