require(Seurat)
require(tidyverse)
require(patchwork)
require(cowplot)
require(RColorBrewer)
require(Biobase)
require(dbscan)
library(patchwork)
library(ggpubr)
library(gridExtra)
library(anndata)
library(MOFA2)

setwd("/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/Data_codes_cmp_norm/Cite_seq_detailed_ablation")

rna_data_ann_data = read_h5ad("data_mofa_rna.h5ad")
rna_data = t(as.matrix(rna_data_ann_data$X))

adt_data = read.csv("data_mofa_protein.csv")
adt_data = adt_data[,-1]
# Compute variance of each column
col_vars <- apply(adt_data, 2, var)
# Get the indices of the top 30 most variable columns
top30_idx <- order(col_vars, decreasing = TRUE)[1:30]
# Subset the data to keep only those columns
adt_data <- as.matrix(t(adt_data[, top30_idx]))

meta = read.csv("meta.csv")
meta = meta[,11]

data <- list(view_1 = rna_data, view_2 = adt_data)

data$view_1 <- rna_data
data$view_2 <-  adt_data

colnames(data$view_1) <- colnames(data$view_2)

MOFAobject <- create_mofa(data)

data_opts <- get_default_data_options(MOFAobject)
model_opts <- get_default_model_options(MOFAobject)
train_opts <- get_default_training_options(MOFAobject)

MOFAobject <- prepare_mofa(
  object = MOFAobject,
  data_options = data_opts,
  model_options = model_opts,
  training_options = train_opts
)

outfile = file.path(getwd(),"model_mofa_tea_seq.hdf5")
MOFAobject.trained <- run_mofa(MOFAobject, outfile, use_basilisk = TRUE)

model <- load_model("/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/Data_codes_cmp_norm/Cite_seq_detailed_ablation/model_mofa_tea_seq.hdf5")
Nsamples = sum(model@dimensions$N)

sample_metadata <- data.frame(
  sample = samples_names(model)[[1]],
  type = meta
)
samples_metadata(model) <- sample_metadata

Z <- get_factors(model)$group1
write.csv(Z,"mofa_embeddings_cite_seq.csv")

set.seed(42)
model <- run_umap(model)


