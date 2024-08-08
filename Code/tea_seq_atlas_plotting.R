require(Seurat)
require(SeuratObject)
require(tidyverse)
require(patchwork)
require(cowplot)
require(RColorBrewer)
require(Biobase)
require(clusterSim)
require(fpc)
require(clusterSim)

#########################################

# Computes UMAP plots

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

#########################################

#  Sets seed for UMAP calculation

set.seed(10)

#########################################

# Brew colors for the UMAP

n = 100
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

#########################################

# Computes WNN Embedding

x_embedding_raw = read.csv("pc_adt_for_tea_seq_analysis_job_talk.csv")
x_embedding_raw = x_embedding_raw[,-1]

y_embedding_raw = read.csv("pc_atac_for_tea_seq_analysis_job_talk.csv")
y_embedding_raw = y_embedding_raw[,-1]

z_embedding_raw = read.csv("pc_rna_for_tea_seq_analysis_job_talk.csv")
z_embedding_raw = z_embedding_raw[,-1]

colnames(x_embedding_raw) <- paste("pro", 1:dim(x_embedding_raw)[2])

colnames(x_embedding_raw) <- paste("adt", 1:dim(x_embedding_raw)[2])

adt <- t(x_embedding_raw)
colnames(adt) <- paste("p", 1:1:dim(x_embedding_raw)[1])

colnames(y_embedding_raw) <- paste("atac", 1:dim(y_embedding_raw)[2])

atac <- t(y_embedding_raw)
colnames(atac) <- paste("p", 1:1:dim(y_embedding_raw)[1])

colnames(z_embedding_raw) <- paste("rna", 1:dim(z_embedding_raw)[2])

rna <- t(z_embedding_raw)
colnames(rna) <- paste("p", 1:1:dim(z_embedding_raw)[1])

all.equal(colnames(rna), colnames(atac), colnames(adt))

data <- CreateSeuratObject(counts = rna)

atac_assay <- CreateAssayObject(counts = atac)
data[["ATAC"]] <- atac_assay

atac_assay <- CreateAssayObject(counts = adt)
data[["ADT"]] <- atac_assay


DefaultAssay(data) <- 'RNA'
VariableFeatures(data) <- rownames(data[["RNA"]])

rna_pca_to_put <- as.matrix(z_embedding_raw)
rownames(rna_pca_to_put) <- colnames(data)

data[["pca"]] <- CreateDimReducObject(embeddings = rna_pca_to_put, key = "PCA_", assay = DefaultAssay(data))


DefaultAssay(data) <- 'ATAC'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(data) <- rownames(data[["ATAC"]])
atac_pca_to_put <- as.matrix(y_embedding_raw)
rownames(atac_pca_to_put) <- colnames(data)
data[["atpca"]] <- CreateDimReducObject(embeddings = atac_pca_to_put, key = "ATPCA_", assay = DefaultAssay(data))

DefaultAssay(data) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(data) <- rownames(data[["ADT"]])
adt_pca_to_put <- as.matrix(x_embedding_raw)
rownames(adt_pca_to_put) <- colnames(data)
data[["apca"]] <- CreateDimReducObject(embeddings = adt_pca_to_put, key = "APCA_", assay = DefaultAssay(data))


DefaultAssay(data) <- 'RNA'

data <- FindMultiModalNeighbors(
  data, reduction.list = list("apca", "atpca", "pca" ), 
  dims.list = list(1:29, 1:49, 1:49), modality.weight.name = "RNA.weight"
)

data <- RunUMAP(data, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

DefaultAssay(data) <- 'RNA'

data_x = Embeddings(object = data, reduction = "wnn.umap")
dims <- 1:2
dims <- paste0(Key(object = "X"), dims)
colnames(data_x) = dims


meta_data = read.csv("cleaned_cell_labels_meta_tea_seq.csv")
cell_names = meta_data$x
p1 = DimPlot(data_x,group.by=cell_names,pt.size=0.04,cols = col_vector[11:40])+ggtitle("WNN atlas")+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position = "none")

#########################################

# Computes AMP UMAP

jt_embedding_gmm = read.csv("amp_gmm.csv")
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
data_together_gmm <- RunUMAP(data_together_gmm, reduction = "pca", dims = 1:34, verbose = FALSE)

data_atlas_together_gmm = Embeddings(object = data_together_gmm, reduction = "umap")
p3 = DimPlot(data_atlas_together_gmm,group.by=cell_names,pt.size=0.04,cols = col_vector[11:40])+ggtitle("orchAMP atlas")+theme(plot.title = element_text(hjust = 0.5))


#########################################

# Plots two UMAPS side-by-side

p5 = p1+p3
ggsave("atlas_tea_seq.pdf", p5, width = 20, height = 6)

