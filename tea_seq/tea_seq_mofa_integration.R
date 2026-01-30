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

setwd("/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/Data_codes_cmp_norm/Tea_seq_detailed_ablation_3pct/15_20_ranks")

rna_data_ann_data = read_h5ad("cleaned_rna_reads_tea_seq.h5ad")
rna_data = rna_data_ann_data$X

atac_data_ann_data = read_h5ad("cleaned_atac_reads_tea_seq.h5ad")
atac_data = atac_data_ann_data$X

adt_data = read.csv("adt_data_mofa_10_15.csv")
adt_data = adt_data[,-1]
adt_data = t(adt_data)

meta = read.csv("cleaned_cell_labels_meta_tea_seq.csv")
meta = meta[,2]

data <- list(view_1 = rna_data, view_2 = atac_data, view_3 = adt_data)

data$view_1 <- rna_data
data$view_2 <- atac_data
data$view_3 <- adt_data
colnames(data$view_3) <- colnames(data$view_2)

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

model <- load_model("/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/Data_codes_cmp_norm/Tea_seq_detailed_ablation_3pct/15_20_ranks/model_mofa_tea_seq.hdf5")
Nsamples = sum(model@dimensions$N)

sample_metadata <- data.frame(
  sample = samples_names(model)[[1]],
  type = meta
)
samples_metadata(model) <- sample_metadata

Z <- get_factors(model)$group1
write.csv(Z,"mofa_embeddings_tea_seq.csv")

set.seed(42)
model <- run_umap(model)



