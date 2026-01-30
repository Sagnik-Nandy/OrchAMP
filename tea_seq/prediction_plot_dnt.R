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

# Brew colors for the UMAP
n = 100
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

setwd('/Users/sagniknandy/Library/CloudStorage/Dropbox/PhD Projects/Current_Projects_with_Seniors/AMP+Multimodal_Data/AoS Revision R1/Data_codes/tea_seq_prediction/')

meta_data = read.csv("cleaned_cell_labels_meta_tea_seq.csv")
cell_names = meta_data$x

cell_names_train = read.csv("labels_modified_pred_full_set_rna.csv",row.names=c(1))
cell_names_modified = cell_names_train

#cell_names_modified = cell_names

amp_embedding_raw = read.csv("amp_for_prediction_tea_seq_analysis_bcp.csv")
amp_embedding_raw = amp_embedding_raw[,-c(1)]     
rownames(amp_embedding_raw) = paste("p", 1:1:dim(amp_embedding_raw)[1])

colnames(amp_embedding_raw) <- paste("RNA", 1:dim(amp_embedding_raw)[2])
rna <- as.sparse(t(amp_embedding_raw))
colnames(rna) <- paste("p", 1:1:dim(amp_embedding_raw)[1])
amp_embedding <- CreateSeuratObject(counts = rna)
DefaultAssay(amp_embedding) <- "RNA"
amp_embedding[["pca"]] <- CreateDimReducObject(embeddings = as.matrix(amp_embedding_raw), key = "PCA_", assay = DefaultAssay(amp_embedding))
amp_embedding  <- RunUMAP(amp_embedding, reduction = "pca", dims = 1:(dim(amp_embedding_raw)[2]-1), , return.model = TRUE, n.neighbors = 10)
umap_atlas <- Embeddings(amp_embedding, reduction = 'umap')


predicted_embedding = read.csv("predicted_embeddings_set_full_set_rna_dnt.csv")
predicted_embedding = as.matrix(predicted_embedding[,-c(1)])
rownames(predicted_embedding) = paste("p", 1:1:dim(predicted_embedding)[1])


reduction_model_proj <- amp_embedding[['umap']]

query_proj_umap <- as.matrix(predicted_embedding)
rownames(query_proj_umap) = paste("p", 1:1:dim(query_proj_umap)[1])

reference_proj_umap <- as.matrix(amp_embedding_raw)
rownames(reference_proj_umap) = paste("p", 1:1:dim(reference_proj_umap)[1])

L <- ProjectUMAP(query = query_proj_umap, reference = reference_proj_umap, reduction.model = reduction_model_proj)$proj.umap
L_embeddings <- Embeddings(L)

umap_together_2 <- rbind(umap_atlas, L_embeddings)
rownames(umap_together_2) = paste("p", 1:1:dim(umap_together_2)[1])

predicted_cell_names_3 = read.csv("labels_projected_test_full_store_full_set_rna_dnt.csv")
predicted_cell_names_3 = as.matrix(predicted_cell_names_3[,-1])
colnames(predicted_cell_names_3) <- "X0"

cell_names_with_predicted_2 <- rbind(as.matrix(cell_names_modified), predicted_cell_names_3)
cell_names_prediction_set_2 <- rbind(matrix(rep("atlas", dim(umap_atlas)[1]), ncol = 1), matrix(rep("predicted", dim(L_embeddings)[1]), ncol = 1))


cell_names_with_predicted_2[cell_names_with_predicted_2=='B cell pr'] = 'B cell progenitor'
cell_names_with_predicted_2[cell_names_with_predicted_2=='pre-B cel'] = 'pre-B cell'


p3 = DimPlot(umap_atlas,group.by=as.matrix(cell_names_modified),pt.size=0.04,cols = col_vector[11:40])+ggtitle("orchAMP atlas")+theme(plot.title = element_text(hjust = 0.5))
p4 = DimPlot(umap_together_2,group.by=cell_names_prediction_set_2,pt.size=0.04,cols = c("grey", "red"))+ theme(legend.position = "none")+ggtitle("Prediction Set: RNA (DnT)")+theme(plot.title = element_text(hjust = 0.5))



predicted_embedding = read.csv("predicted_embeddings_set_full_set_prot_dnt.csv")
predicted_embedding = as.matrix(predicted_embedding[,-c(1)])
rownames(predicted_embedding) = paste("p", 1:1:dim(predicted_embedding)[1])


reduction_model_proj <- amp_embedding[['umap']]

query_proj_umap <- as.matrix(predicted_embedding)
rownames(query_proj_umap) = paste("p", 1:1:dim(query_proj_umap)[1])

reference_proj_umap <- as.matrix(amp_embedding_raw)
rownames(reference_proj_umap) = paste("p", 1:1:dim(reference_proj_umap)[1])

L <- ProjectUMAP(query = query_proj_umap, reference = reference_proj_umap, reduction.model = reduction_model_proj)$proj.umap
L_embeddings <- Embeddings(L)

umap_together_2 <- rbind(umap_atlas, L_embeddings)
rownames(umap_together_2) = paste("p", 1:1:dim(umap_together_2)[1])

predicted_cell_names_3 = read.csv("labels_projected_test_full_store_full_set_prot_dnt.csv")
predicted_cell_names_3 = as.matrix(predicted_cell_names_3[,-1])
colnames(predicted_cell_names_3) <- "X0"

cell_names_with_predicted_2 <- rbind(as.matrix(cell_names_modified), predicted_cell_names_3)
cell_names_prediction_set_2 <- rbind(matrix(rep("atlas", dim(umap_atlas)[1]), ncol = 1), matrix(rep("predicted", dim(L_embeddings)[1]), ncol = 1))


cell_names_with_predicted_2[cell_names_with_predicted_2=='B cell pr'] = 'B cell progenitor'
cell_names_with_predicted_2[cell_names_with_predicted_2=='pre-B cel'] = 'pre-B cell'

p5 = DimPlot(umap_together_2,group.by=cell_names_prediction_set_2,pt.size=0.04,cols = c("grey", "red"))+ theme(legend.position = "none")+ggtitle("Prediction Set: Protein (DnT)")+theme(plot.title = element_text(hjust = 0.5))




predicted_embedding = read.csv("predicted_embeddings_set_full_set_atac_dnt.csv")
predicted_embedding = as.matrix(predicted_embedding[,-c(1)])
rownames(predicted_embedding) = paste("p", 1:1:dim(predicted_embedding)[1])


reduction_model_proj <- amp_embedding[['umap']]

query_proj_umap <- as.matrix(predicted_embedding)
rownames(query_proj_umap) = paste("p", 1:1:dim(query_proj_umap)[1])

reference_proj_umap <- as.matrix(amp_embedding_raw)
rownames(reference_proj_umap) = paste("p", 1:1:dim(reference_proj_umap)[1])

L <- ProjectUMAP(query = query_proj_umap, reference = reference_proj_umap, reduction.model = reduction_model_proj)$proj.umap
L_embeddings <- Embeddings(L)

umap_together_2 <- rbind(umap_atlas, L_embeddings)
rownames(umap_together_2) = paste("p", 1:1:dim(umap_together_2)[1])

predicted_cell_names_3 = read.csv("labels_projected_test_full_store_full_set_atac_dnt.csv")
predicted_cell_names_3 = as.matrix(predicted_cell_names_3[,-1])
colnames(predicted_cell_names_3) <- "X0"

cell_names_with_predicted_2 <- rbind(as.matrix(cell_names_modified), predicted_cell_names_3)
cell_names_prediction_set_2 <- rbind(matrix(rep("atlas", dim(umap_atlas)[1]), ncol = 1), matrix(rep("predicted", dim(L_embeddings)[1]), ncol = 1))


cell_names_with_predicted_2[cell_names_with_predicted_2=='B cell pr'] = 'B cell progenitor'
cell_names_with_predicted_2[cell_names_with_predicted_2=='pre-B cel'] = 'pre-B cell'

p6 = DimPlot(umap_together_2,group.by=cell_names_prediction_set_2,pt.size=0.04,cols = c("grey", "red"))+ theme(legend.position = "none")+ggtitle("Prediction Set: ATAC (DnT)")+theme(plot.title = element_text(hjust = 0.5))

p7 = (p3 + theme(legend.position = "bottom") | p6 + theme(legend.position = "none")) / (p4 + theme(legend.position = "none") | p5 + theme(legend.position = "none")) 

ggsave("tea_seq_prediction_dnt_all_three.pdf", p7, height = 10, width = 10)
ggsave("tea_seq_prediction_dnt_all_three.png", p7, height = 10, width = 10)


p8 = p7 + theme(plot.margin = margin(t = 10, r = 10, b = 50, l = 10))
ggsave("plot_with_space.png", plot = p8, width = 8, height = 10, dpi = 300)


p <- list()

p[[1]] <- p3 + theme(legend.position = "none")
p[[2]] <- p6 + theme(legend.position = "none")
p[[3]] <- p4 + theme(legend.position = "none")
p[[4]] <- p5 + theme(legend.position = "none")

legend <- get_legend(p3)

p_grid <- cowplot::plot_grid(plotlist = p, ncol = 2)
cowplot::plot_grid(p_grid, legend, ncol = 1, rel_heights = c(1, 0.2))


