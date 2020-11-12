library(lidR)
library(parallel)
library(rgdal)

args <- commandArgs(trailingOnly=TRUE)

data_path <- args[1]
plot_csv <- args[2]
outdir <- args[3]

files <- list.files(path=data_path, pattern="*.las", full.names=TRUE, recursive=FALSE)
cat(length(files), ' tiles in folder ', data_path, '\n')

getFieldPlotTiles <- function(filename, df, outdir) {
    # Read LAS and normalize height
    cat('processing file ', filename, '\n')
    tile <- readLAS(filename)
    tile <- normalize_height(tile, knnidw())

    # Select only field plots that are located within tile
    df <- df[which(df$x <= lasfile@header@PHB$`Max X` - 9),]
    if (nrow(df) == 0) {return()}
    df <- df[which(df$x >= lasfile@header@PHB$`Min X` + 9),]
    if (nrow(df) == 0) {return()}
    df <- df[which(df$y <= lasfile@header@PHB$`Max Y` - 9),]
    if (nrow(df) == 0) {return()}
    df <- df[which(df$y >= lasfile@header@PHB$`Min Y` + 9),]
    if (nrow(df) == 0) {return()}

    rows <- function(x) lapply(seq_len(nrow(x)), function(i) lapply(x, "[", i))
    for (row in rows(df)) {
        xleft <- row$x - 9
        xright <- row$x + 9
        ytop <- row$y - 9
        ybot <- row$y + 9

        plot_las <- clip_rectangle(tile, xleft, ytop, xright, ybot)
        plot_las@header@VLR <- list()
        outfile <- file.path(outdir, paste0(row$sampleplotid, '.las'))
        writeLAS(plot_las, paste0(row$sampleplotid, '.las'), index=FALSE)
    }
}

df <- read.table(plot_csv, header=TRUE, sep=';')

mclapply(files, getFieldPlotTiles, df=df, outdir=outdir)