library("readr")
library(rhdf5)
library(Matrix)
library(tidyverse)
library(glue)
library(dplyr)
library(Seurat)
require(RColorBrewer)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(cowplot)
require(patchwork)
require(RColorBrewer)
require(Biobase)
require(reticulate)
require(nng)
require(leiden)
require(combinat)
require(anndata)
library(SeuratData)
library(SeuratDisk)


#########################################

# Brew colors for the UMAP
n = 100
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

#########################################

setwd("/Users/sagnik/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/github_repo/Data/")


# RNA and ATAC Expression Matrix

cell_barcodes_gene <- h5read("feature_matrix.h5", "matrix/barcodes")
gene_expression <- h5read("feature_matrix.h5", "matrix")
counts_gene <- sparseMatrix(
  dims = gene_expression$shape,
  i = as.numeric(gene_expression$indices),
  p = as.numeric(gene_expression$indptr),
  x = as.numeric(gene_expression$data),
  index1 = FALSE
)
colnames(counts_gene) <- cell_barcodes_gene
rownames(counts_gene) <- gene_expression[["features"]]$name
transcript_features <- which(gene_expression[["features"]]$feature_type=="Gene Expression")
atac_features <- which(gene_expression[["features"]]$feature_type=="Peaks")


rna_expression_matrix <- counts_gene[transcript_features,]
atac_expression_matrix <- counts_gene[atac_features,]

#########################################

# Pre-Processing scRNA seq data

rna_expression_matrix <- as.matrix(rna_expression_matrix)
duplicated_rownames <- rownames(rna_expression_matrix)[duplicated(rownames(rna_expression_matrix))]
rna_expression_matrix <- rna_expression_matrix[!duplicated(rownames(rna_expression_matrix)), ]

rna_tea_seq <- CreateSeuratObject(counts = rna_expression_matrix, project = "tea_seq", min.cells = 50, min.features = 200)
rna_tea_seq[["percent.mt"]] <- PercentageFeatureSet(rna_tea_seq, pattern = "^MT-")
rna_tea_seq[["percent.rb"]] <- PercentageFeatureSet(rna_tea_seq, "^RP[SL]")

# Visualize QC metrics as a violin plot
VlnPlot(rna_tea_seq, features = c("nFeature_RNA", "nCount_RNA"), ncol = 2)
VlnPlot(rna_tea_seq, features = c("percent.mt","percent.rb"), ncol = 2)


# We filter cells that have unique feature counts over 3,500 or less than 200
# We filter cells that have >20% mitochondrial counts
rna_tea_seq <- subset(rna_tea_seq, subset = nFeature_RNA > 200 & nFeature_RNA < 3500 & percent.mt < 20 & percent.rb < 5)


rna_tea_seq <- NormalizeData(rna_tea_seq)
rna_tea_seq <- FindVariableFeatures(rna_tea_seq, selection.method = "vst", nfeatures = 2000)
rna_tea_seq <- ScaleData(rna_tea_seq)
rna_tea_seq <- RunPCA(rna_tea_seq, features = VariableFeatures(object = rna_tea_seq))

ElbowPlot(rna_tea_seq)

rna_tea_seq <- FindNeighbors(rna_tea_seq, dims = 1:25)
rna_tea_seq <- FindClusters(rna_tea_seq, resolution = 0.7)

rna_tea_seq <- RunUMAP(rna_tea_seq, dims = 1:25)
DimPlot(rna_tea_seq, reduction = "umap",cols = col_vector[10:29], group.by = "seurat_clusters")
filtered_cells_rna <- colnames(rna_tea_seq)

#########################################

# Label transfer from cite-seq

if(!file.exists("pbmc_10k_v3.rds")) {
  download.file("https://www.dropbox.com/s/3f3p5nxrn5b3y4y/pbmc_10k_v3.rds?dl=1",
                "pbmc_10k_v3.rds")
}
pbmc_so <- readRDS("pbmc_10k_v3.rds")
pbmc_so <- UpdateSeuratObject(pbmc_so)
pbmc_so <- RunUMAP(pbmc_so, dim=1:50, return.model=TRUE)
pbmc_mat <- pbmc_so@assays$RNA@counts

pbmc_genes <- rownames(pbmc_mat)
keep_genes <- pbmc_genes %in% rownames(filtered_cells_rna)
pbmc_mat <- pbmc_mat[keep_genes,]

pbmc_genes <- rownames(pbmc_mat)
pbmc_meta <- pbmc_so@meta.data
pbmc_meta$original_barcodes <- rownames(pbmc_meta)

pbmc_meta <- pbmc_meta[pbmc_meta$celltype != "Platelets",]
pbmc_mat <- pbmc_mat[,pbmc_meta$original_barcodes]

pbmc_meta <- lapply(pbmc_meta,
                    function(x) {
                      if(class(x) == "factor") {
                        as.character(x)
                      } else {
                        x
                      }
                    })


pbmc_cell_types <- matrix(pbmc_meta[[8]])
rownames(pbmc_cell_types) <- pbmc_meta[[9]]


pbmc_so <- AddMetaData(
  object = pbmc_so,
  metadata = pbmc_cell_types,
  col.name = 'celltype'
)

anchors <- FindTransferAnchors(
  reference = pbmc_so,
  query = rna_tea_seq,
  normalization.method = "LogNormalize",
  reference.reduction = "pca",
  dims = 1:50,
  k.anchor = 20
)

rna_tea_seq <- MapQuery(
  anchorset = anchors,
  query = rna_tea_seq,
  reference = pbmc_so,
  refdata = "celltype",
  reference.reduction = "pca", 
  reduction.model = "umap",
)

rna_tea_seq_meta <- rna_tea_seq$predicted.id
rownames(rna_tea_seq_meta) <- colnames(rna_tea_seq_meta)


rna_tea_seq <- AddMetaData(
  object = rna_tea_seq,
  metadata =rna_tea_seq_meta,
  col.name = 'celltype'
)

DimPlot(rna_tea_seq, reduction = "umap",cols = col_vector[10:29], group.by = "celltype")


#########################################

# Pre-Processing scATAC seq data

filtered_cells_rna <- colnames(rna_tea_seq)
filtered_cell_barcodes_seq <- intersect(colnames(atac_expression_matrix),filtered_cells_rna)

atac_tea_seq <- CreateAssayObject(atac_expression_matrix[,filtered_cell_barcodes_seq])
atac_tea_seq <- CreateSeuratObject(counts = atac_tea_seq, project = "tea_seq", min.cells = 50, min.features = 1000)
atac_cell_names <- colnames(atac_tea_seq)

meta_for_atac <- rna_tea_seq$celltype
meta_for_atac <- meta_for_atac[atac_cell_names]

atac_tea_seq <- AddMetaData(
  object = atac_tea_seq,
  metadata =meta_for_atac,
  col.name = 'celltype'
)

atac_tea_seq <- subset(atac_tea_seq, subset = nFeature_RNA < 10000)

atac_tea_seq <- NormalizeData(atac_tea_seq)
atac_tea_seq <- FindVariableFeatures(atac_tea_seq,nfeatures = 5000)
atac_tea_seq <- ScaleData(atac_tea_seq)
atac_tea_seq <- RunPCA(atac_tea_seq, features = VariableFeatures(object = atac_tea_seq))

ElbowPlot(atac_tea_seq)

atac_tea_seq <- FindNeighbors(atac_tea_seq, dims = 1:10)
atac_tea_seq <- FindClusters(atac_tea_seq, resolution = 1)

atac_tea_seq <- RunUMAP(atac_tea_seq, dims = 1:10)


filtered_cell_barcodes_seq <- colnames(atac_tea_seq)
DimPlot(atac_tea_seq, reduction = "umap",cols = col_vector[10:29], group.by = "celltype")


atac_tea_seq <- atac_tea_seq[,filtered_cell_barcodes_seq]
rna_tea_seq <- rna_tea_seq[,filtered_cell_barcodes_seq]

#########################################

# Pre-Processing ADT data

adt_data <- read.csv("adt_counts.csv")
rownames(adt_data) <- adt_data[,1]
cells_filtered <- adt_data[,1] %in% sub("-1","",filtered_cell_barcodes_seq)
adt_data <- adt_data[cells_filtered,-c(1,2)]

adt_total <- rowSums(adt_data)
hist(log10(adt_total),breaks=200)
sum(adt_total > 500)
# ~6.7k barcodes have > 500
sum(adt_total > 8000)
# only 46 barcodes are high outliers
adt_data <- adt_data[adt_total > 500,]
adt_data <- adt_data[adt_total < 5000,]

adt_data <- t(adt_data)

colnames(adt_data) <- paste0(colnames(adt_data), "-1")

filtered_cell_barcodes_trimodal <- intersect(filtered_cell_barcodes_seq,colnames(adt_data))

atac_tea_seq <- atac_tea_seq[,filtered_cell_barcodes_trimodal]
rna_tea_seq <- rna_tea_seq[,filtered_cell_barcodes_trimodal]
adt_data <- adt_data[,filtered_cell_barcodes_trimodal]
rna_tea_seq_meta_final <- rna_tea_seq$celltype[filtered_cell_barcodes_trimodal]

#########################################

# Storing the processed data

rna_data_final <- GetAssayData(object = rna_tea_seq, slot = "scale.data")[,filtered_cell_barcodes_trimodal]
atac_data_final <- GetAssayData(object = atac_tea_seq, slot = "scale.data")[,filtered_cell_barcodes_trimodal]

ad_rna <- AnnData(X = as.matrix(rna_data_final))
write_h5ad(ad_rna, "cleaned_rna_reads_tea_seq.h5ad")

ad_atac <- AnnData(X = as.matrix(atac_data_final))
write_h5ad(ad_atac, "cleaned_atac_reads_tea_seq.h5ad")

write.csv(t(adt_data),"cleaned_adt_tea_seq.csv")
write.csv(rna_tea_seq_meta_final,"cleaned_cell_labels_meta_tea_seq.csv")


