require(Seurat)
require(tidyverse)
require(patchwork)
require(cowplot)
require(RColorBrewer)
require(Biobase)
require(clusterSim)
require(fpc)
require(clusterSim)


lisi <- function(
    X, cell_labels, no_nn = 10 , nn_eps = 0
){
  lisi <- NULL
  N <- nrow(X)
  dknn <- RANN::nn2(X, k = no_nn, eps = nn_eps)
  for(cell in 1:N){
    nn.idx_cell <- dknn$nn.idx[cell, 2:ncol(dknn$nn.idx)]
    index_neighbor <- cell_labels[nn.idx_cell]
    prop_cell_labels_nn <- as.numeric(table(index_neighbor)/length(index_neighbor))
    inv_simpson_index <- 1/sum(prop_cell_labels_nn^2)
    lisi <- c(lisi, inv_simpson_index)
  }
  return(lisi)
}

mean_nn <- function(
    vector_nn_neighbor,
    M
){
  print(vector_nn_neighbor)
  temp_matrix = M[vector_nn_neighbor,]
  return(apply(temp_matrix,MARGIN=2,FUN=mean))
}

compute_final_embedding <- function(
    Matrix_nn_index,
    Matrix_query
){
  embedding_matrix = NULL
  for(i in 1:dim(Matrix_nn_index)[1]){
    embedding_matrix = rbind(embedding_matrix,mean_nn(vector_nn_neighbor=Matrix_nn_index[i,],M=Matrix_query))
  }
  return(embedding_matrix)
}

DimPlot <- function(
    data,
    dims = c(1, 2),
    cells = NULL,
    cols = NULL,
    pt.size = NULL,
    reduction = NULL,
    group.by = NULL,
    split.by = NULL,
    shape.by = NULL,
    order = NULL,
    shuffle = FALSE,
    seed = 1,
    label = FALSE,
    label.size = 4,
    label.color = 'black',
    label.box = FALSE,
    repel = FALSE,
    cells.highlight = NULL,
    cols.highlight = '#DE2D26',
    sizes.highlight = 1,
    na.value = 'grey50',
    ncol = NULL,
    combine = TRUE,
    raster = NULL,
    raster.dpi = c(512, 512)
) {
  if (length(x = dims) != 2) {
    stop("'dims' must be a two-length vector")
  }
  colnames(data) <- paste0("UMAP", dims)
  data <- as.data.frame(x = data)
  dims <- paste0("UMAP", dims)
  data <- cbind(data, group.by)
  orig.groups <- group.by
  group.by <- colnames(x = data)[3:ncol(x = data)]
  for (group in group.by) {
    if (!is.factor(x = data[, group])) {
      data[, group] <- factor(x = data[, group])
    }
  }
  if (!is.null(x = shape.by)) {
    data[, shape.by] <- object[[shape.by, drop = TRUE]]
  }
  if (!is.null(x = split.by)) {
    data[, split.by] <- object[[split.by, drop = TRUE]]
  }
  if (isTRUE(x = shuffle)) {
    set.seed(seed = seed)
    data <- data[sample(x = 1:nrow(x = data)), ]
  }
  plots <- lapply(
    X = group.by,
    FUN = function(x) {
      plot <- SingleDimPlot(
        data = data[, c(dims, x, split.by, shape.by)],
        dims = dims,
        col.by = x,
        cols = cols,
        pt.size = pt.size,
        shape.by = shape.by,
        order = order,
        label = FALSE,
        cells.highlight = cells.highlight,
        cols.highlight = cols.highlight,
        sizes.highlight = sizes.highlight,
        na.value = na.value,
        raster = raster,
        raster.dpi = raster.dpi
      )
      if (label) {
        plot <- LabelClusters(
          plot = plot,
          id = x,
          repel = repel,
          size = label.size,
          split.by = split.by,
          box = label.box,
          color = label.color
        )
      }
      if (!is.null(x = split.by)) {
        plot <- plot + FacetTheme() +
          facet_wrap(
            facets = vars(!!sym(x = split.by)),
            ncol = if (length(x = group.by) > 1 || is.null(x = ncol)) {
              length(x = unique(x = data[, split.by]))
            } else {
              ncol
            }
          )
      }
      plot <- if (is.null(x = orig.groups)) {
        plot + labs(title = NULL)
      } else {
        plot + labs(title = NULL)
      }
    }
  )
  if (!is.null(x = split.by)) {
    ncol <- 1
  }
  if (combine) {
    plots <- wrap_plots(plots, ncol = orig.groups %iff% ncol)
  }
  return(plots)
}

set.seed(10)

setwd("/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/Data_codes_cmp_norm/Cite_seq_detailed_ablation/")

# Brew colors for the UMAP
n = 100
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))


x_embedding_raw = read.csv("pca_embeddings_for_seurat_integration_adt_rpcs.csv")
x_embedding_raw = x_embedding_raw[,-1]

y_embedding_raw = read.csv("pca_embeddings_for_seurat_integration_rna_rpcs.csv")
y_embedding_raw = y_embedding_raw[,-1]

colnames(x_embedding_raw) <- paste("pro", 1:dim(x_embedding_raw)[2])

colnames(x_embedding_raw) <- paste("adt", 1:dim(x_embedding_raw)[2])

adt <- as.sparse(t(x_embedding_raw))
colnames(adt) <- paste("p", 1:1:dim(x_embedding_raw)[1])

colnames(y_embedding_raw) <- paste("atac", 1:dim(y_embedding_raw)[2])


rna <- as.sparse(t(y_embedding_raw))
colnames(rna) <- paste("p", 1:1:dim(y_embedding_raw)[1])

all.equal(colnames(rna), colnames(adt))

data <- CreateSeuratObject(counts = rna)


atac_assay <- CreateAssayObject(counts = adt)
data[["ADT"]] <- atac_assay


DefaultAssay(data) <- 'RNA'
VariableFeatures(data) <- rownames(data[["RNA"]])

rna_pca_to_put <- as.matrix(y_embedding_raw)
rownames(rna_pca_to_put) <- colnames(data)

data[["pca"]] <- CreateDimReducObject(embeddings = rna_pca_to_put, key = "PCA_", assay = DefaultAssay(data))


DefaultAssay(data) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(data) <- rownames(data[["ADT"]])
adt_pca_to_put <- as.matrix(x_embedding_raw)
rownames(adt_pca_to_put) <- colnames(data)
data[["apca"]] <- CreateDimReducObject(embeddings = adt_pca_to_put, key = "APCA_", assay = DefaultAssay(data))


DefaultAssay(data) <- 'RNA'

data <- FindMultiModalNeighbors(
  data, reduction.list = list("apca", "pca" ), 
  dims.list = list(1:29, 1:49), modality.weight.name = "RNA.weight"
)

data <- RunUMAP(data, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

DefaultAssay(data) <- 'RNA'

data_x = Embeddings(object = data, reduction = "wnn.umap")
dims <- 1:2
dims <- paste0(Key(object = "X_"), dims)
colnames(data_x) = dims


meta_data = read.csv("labels_modified_45_rpcs.csv")
cell_names = meta_data$X0
cell_names[(cell_names %in% c("NK Proliferating","NK_CD56bright"))] = "NK"
distinguished_cell_name = c("CD14 Mono","CD4 Naive","NK","CD4 TCM","CD8 TEM","CD8 TEM","CD8 Naive","B naive","CD16 Mono","CD4 TEM","gdT","B memory","MAIT","CD8 TCM ","cDC2","Treg","Platelet","B intermediate","CD4 CTL","Doublet","dnT")
others_name = !(cell_names %in% distinguished_cell_name)
cell_names_modified = cell_names
cell_names_modified[others_name] = "other"

p1 = DimPlot(data_x,group.by=cell_names_modified,pt.size=0.04,cols = col_vector[10:38])+ggtitle("WNN atlas")+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position = "none")


jt_embedding_gmm = read.csv("orchamp_embeddings_integrated_50_rpcs.csv")
jt_embedding_gmm = jt_embedding_gmm[,-1]
data_together_pca = jt_embedding_gmm
rownames(data_together_pca) = paste("p", 1:1:dim(data_together_pca)[1])

colnames(data_together_pca) <- paste("RNA", 1:dim(data_together_pca)[2])

rna <- as.sparse(t(data_together_pca))
colnames(rna) <- paste("p", 1:1:dim(data_together_pca)[1])

data_together_gmm <- CreateSeuratObject(counts = rna)

DefaultAssay(data_together_gmm) <- "RNA"

VariableFeatures(data_together_gmm) <- rownames(data_together_gmm[["RNA"]])
data_together_gmm[["pca"]] <- CreateDimReducObject(embeddings = as.matrix(data_together_pca), key = "PCA_", assay = DefaultAssay(data_together_gmm))
data_together_gmm <- RunUMAP(data_together_gmm, reduction = "pca", dims = 1:79, verbose = FALSE)

data_atlas_together_gmm = Embeddings(object = data_together_gmm, reduction = "umap")
p2 = DimPlot(data_atlas_together_gmm,group.by=cell_names_modified,pt.size=0.04,cols = col_vector[10:38])+ggtitle("orchAMP atlas")+theme(plot.title = element_text(hjust = 0.5))


p5 = p1+p2
ggsave("atlas_job_talk_r1.pdf", p5, width = 15, height = 8)


data_atlas_tvi = read.csv("umap_post_save_cite_seq_tvi_75rpcs.csv",row.names=c(1))
rownames(data_atlas_tvi) = paste("p", 1:1:dim(data_atlas_tvi)[1])
colnames(data_atlas_tvi) <- paste("RNA", 1:dim(data_atlas_tvi)[2])
rna <- as.sparse(t(data_atlas_tvi))
colnames(rna) <- paste("p", 1:1:dim(data_atlas_tvi)[1])
data_atlas_tvis <- CreateSeuratObject(counts = rna)
DefaultAssay(data_atlas_tvis) <- "RNA"
VariableFeatures(data_atlas_tvis) <- rownames(data_atlas_tvi[["RNA"]])
data_atlas_tvis[["umap"]] <- CreateDimReducObject(embeddings = as.matrix(data_atlas_tvi), key = "UMAP_", assay = DefaultAssay(data_atlas_tvis))


cell_names_train = read.csv("labels_citeseq_tvi_75rpcs.csv",row.names=c(1))
cell_names_train[(cell_names_train %in% c("NK Proliferating","NK_CD56bright"))] = "NK"
distinguished_cell_name = c("CD14 Mono","CD4 Naive","NK","CD4 TCM","CD8 TEM","CD8 TEM","CD8 Naive","B naive","CD16 Mono","CD4 TEM","gdT","B memory","MAIT","CD8 TCM ","cDC2","Treg","Platelet","B intermediate","CD4 CTL","Doublet","dnT")
others_name = !(cell_names_train[,1] %in% distinguished_cell_name)
cell_names_modified_tvi = cell_names_train
cell_names_modified_tvi[others_name,] = "other"

p3 = DimPlot(data_atlas_tvi,group.by=cell_names_modified_tvi,pt.size=0.04,cols = col_vector[10:38])+ggtitle("totalVI atlas")+theme(plot.title = element_text(hjust = 0.5))+theme(legend.position = "none")


jt_embedding_mofa = read.csv("mofa_embeddings_cite_seq.csv")
jt_embedding_mofa = jt_embedding_mofa[,-1]
data_together_pca = jt_embedding_mofa
rownames(data_together_pca) = paste("p", 1:1:dim(data_together_pca)[1])

colnames(data_together_pca) <- paste("RNA", 1:dim(data_together_pca)[2])

rna <- as.sparse(t(data_together_pca))
colnames(rna) <- paste("p", 1:1:dim(data_together_pca)[1])

data_together_mofa <- CreateSeuratObject(counts = rna)

DefaultAssay(data_together_mofa) <- "RNA"

VariableFeatures(data_together_mofa) <- rownames(data_together_mofa[["RNA"]])
data_together_mofa[["pca"]] <- CreateDimReducObject(embeddings = as.matrix(data_together_pca), key = "PCA_", assay = DefaultAssay(data_together_gmm))
data_together_mofa <- RunUMAP(data_together_mofa, reduction = "pca", dims = 1:19, verbose = FALSE)

data_atlas_together_mofa = Embeddings(object = data_together_mofa, reduction = "umap")
p4 = DimPlot(data_atlas_together_mofa,group.by=cell_names_modified,pt.size=0.04,cols = col_vector[10:38])+ggtitle("MOFA+ atlas")+theme(plot.title = element_text(hjust = 0.5))+theme(legend.position = "none") 



p6 = p1+p3+p2

p7 = (p1 + theme(legend.position = "none") | p2 + theme(legend.position = "none")) / (p3 + theme(legend.position = "none") | p4 + theme(legend.position = "right")) 

ggsave("atlas_job_talk_r1_meth_3.pdf", p7, width = 12, height = 10)

# Quality metrics

# Computing Metrics
# Silhouette Score

names_unique = unique(cell_names_modified)
cell_names_integer = rep(0,length(names_unique))
for(count in 1:length(names_unique))
{
  places_where_present = (cell_names_modified == names_unique[count])
  cell_names_integer[places_where_present] = count
}

names_unique_tvi = unique(cell_names_modified_tvi[,1])
cell_names_integer_tvi = rep(0,length(names_unique_tvi))
for(count in 1:length(names_unique_tvi))
{
  places_where_present_tvi = (cell_names_modified_tvi[,1] == names_unique_tvi[count])
  cell_names_integer_tvi[places_where_present_tvi] = count
}


dmat_wnn <- as.matrix(dist(data_x))
silhouette_wnn <- silhouette(cell_names_integer,dmatrix = dmat_wnn)
sum_sil_wnn <- summary(silhouette_wnn)
sil_score_wnn<- sum_sil_wnn$avg.width

# dmat_orchamp <- as.matrix(dist(data_atlas_together))
# silhouette_orchamp <- silhouette(cell_names_integer,dmatrix = dmat_orchamp)
# sum_sil_orchamp <- summary(silhouette_orchamp)
# sil_score_orchamp<- sum_sil_orchamp$avg.width

dmat_orchamp_gmm <- as.matrix(dist(data_atlas_together_gmm))
silhouette_orchamp_gmm <- silhouette(cell_names_integer,dmatrix = dmat_orchamp_gmm)
sum_sil_orchamp_gmm <- summary(silhouette_orchamp_gmm)
sil_score_orchamp_gmm<- sum_sil_orchamp_gmm$avg.width

dmat_tvi <- as.matrix(dist(data_atlas_tvi))
silhouette_tvi <- silhouette(cell_names_integer_tvi,dmatrix = dmat_tvi)
sum_sil_tvi <- summary(silhouette_tvi)
sil_score_tvi<- sum_sil_tvi$avg.width

dmat_mofa <- as.matrix(dist(data_atlas_together_mofa))
silhouette_mofa <- silhouette(cell_names_integer,dmatrix = dmat_mofa)
sum_sil_mofa <- summary(silhouette_mofa)
sil_score_mofa<- sum_sil_mofa$avg.width



# Average Rand Index

data <- FindNeighbors(data, reduction = 'wnn.umap', dims = 1:2)
data <- FindClusters(data, resolution = 0.2, method = 4)
seuratclusterwnn <- data$seurat_clusters
adj_r_index_wnn <- mclust::adjustedRandIndex(seuratclusterwnn, cell_names_integer)

# data_together<- FindNeighbors(data_together, reduction = 'umap', dims = 1:2)
# data_together <- FindClusters(data_together, resolution = 0.21, method = 4)
# seuratclusteramp <- data_together$seurat_clusters
# adj_r_index_orchamp <- mclust::adjustedRandIndex(seuratclusteramp, cell_names_integer)

data_together_gmm <- FindNeighbors(data_together_gmm, reduction = 'umap', dims = 1:2)
data_together_gmm <- FindClusters(data_together_gmm, resolution = 0.18, method = 4)
seuratclusterampgmm <- data_together_gmm$seurat_clusters
adj_r_index_orchamp_gmm <- mclust::adjustedRandIndex(seuratclusterampgmm, cell_names_integer)

data_atlas_tvis <- FindNeighbors(data_atlas_tvis, reduction = 'umap', dims = 1:2)
data_atlas_tvis <- FindClusters(data_atlas_tvis, resolution = 0.2, method = 4)
seuratclustertvi <- data_atlas_tvis$seurat_clusters
adj_r_index_tvi <- mclust::adjustedRandIndex(seuratclustertvi, cell_names_integer_tvi)

data_together_mofa <- FindNeighbors(data_together_mofa, reduction = 'umap', dims = 1:2)
data_together_mofa <- FindClusters(data_together_mofa, resolution = 0.2, method = 4)
seuratclustermofa <- data_together_mofa$seurat_clusters
adj_r_index_mofa <- mclust::adjustedRandIndex(seuratclustermofa, cell_names_integer)




# V Measure

v_measure_wnn <- clevr::v_measure(seuratclusterwnn, cell_names_integer)
#v_measure_orchamp <- clevr::v_measure(seuratclusteramp, cell_names_integer)
v_measure_orchamp_gmm <- clevr::v_measure(seuratclusterampgmm, cell_names_integer)
v_measure_tvi <- clevr::v_measure(seuratclustertvi, cell_names_integer_tvi)
v_measure_mofa <- clevr::v_measure(seuratclustermofa, cell_names_integer)

## LISI Score

colnames <- colnames(data_x)
lisi_wnn <- mean(lisi(data_x,cell_names_integer))

# colnames <- colnames(data_atlas_together)
# lisi_orchamp <- mean(lisi(data_atlas_together,cell_names_integer))

colnames <- colnames(data_atlas_together_gmm)
lisi_orchamp_gmm <- mean(lisi(data_atlas_together_gmm,cell_names_integer))

colnames <- colnames(data_atlas_tvi)
lisi_tvi <- mean(lisi(data_atlas_tvi,cell_names_integer_tvi))

colnames <- colnames(data_atlas_together_mofa)
lisi_mofa <- mean(lisi(data_atlas_together_mofa,cell_names_integer))

c(adj_r_index_wnn, adj_r_index_orchamp_gmm, adj_r_index_tvi, adj_r_index_mofa)
c(sil_score_wnn, sil_score_orchamp_gmm, sil_score_tvi, sil_score_mofa)
c(lisi_wnn, lisi_orchamp_gmm, lisi_tvi, lisi_mofa)
c(v_measure_wnn, v_measure_orchamp_gmm, v_measure_tvi, v_measure_mofa)




