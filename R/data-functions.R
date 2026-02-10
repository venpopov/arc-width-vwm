#' Convert a .mat file from honig-et-al raw data to a data frame
#' @param file Path to the .mat file
#' @return A data frame with the data and properly labeled columns
mat2dat <- function(file) {
  m <- R.matlab::readMat(file)
  dm <- m$dm[, , 1]
  fa <- m$fa[, , 1]
  ex <- m$ex[, , 1]
  N <- length(dm$betcolor[, 1])

  dat <- data.frame(
    subject = ex$subj[1],
    session = ex$session[1],
    trial = 1:N,
    resp = dm$betcolor[, 1],
    rt = dm$resptime[, 1],
    arc = dm$betarc[, 1],
    probedid = fa$probed[1:N, 1],
    probedcol = fa$probedcol[1:N, 1],
    probeddist = fa$probedist[1:N, 1]
  )

  dat$resperr <- dat$resp - dat$probedcol
  stimcols <- as.data.frame(fa$dotcols[1:N, ])
  names(stimcols) <- c("col1", "col2", "col3", "col4")
  stimdists <- as.data.frame(fa$dotdist[1:N, ])
  names(stimdists) <- c("dist1", "dist2", "dist3", "dist4")

  dat <- cbind(dat, stimcols, stimdists)
  return(dat)
}






#' Preprocess all .mat files from honig-et-al raw data
#' @param dir Directory containing the .mat files
#' @return A data frame with the preprocessed data
preprocess_honig_data <- function(dir, output_file) {
  # read files and convert to data.frame
  files <- list.files(dir, pattern = ".mat", full.names = TRUE)
  dat <- lapply(files, mat2dat)
  dat <- do.call(rbind, dat)

  # wrap response error and convert to radians
  cols <- c(
    "resperr", "arc", "probedcol", "probeddist", "resp", "col1", "col2",
    "col3", "col4", "dist1", "dist2", "dist3", "dist4"
  )
  dat[, cols] <- bmm::deg2rad(dat[, cols])
  cols <- cols[cols != "arc"]
  dat[, cols] <- bmm::wrap(dat[, cols])

  # get colors for non-probed items relative to target
  cols <- dat[, c("col1", "col2", "col3", "col4")]
  for (i in 1:nrow(cols)) {
    tid <- dat[i, ]$probedid
    cols[i, ] <- c(cols[i, tid], cols[i, -tid])
  }
  cols <- cols[, -1]
  cols <- cols
  names(cols) <- c("nt1col", "nt2col", "nt3col")
  dat <- cbind(dat, cols)

  # write to file
  write.csv(dat, output_file, row.names = FALSE)
  dat
}


preprocess_exp1_data <- function(input_file, output_file) {
  data <- read.csv(input_file)
  data[data == 999] <- NA
  col <- colnames(data)
  col <- col[grepl("p[1-8]color", col) | grepl("[a|A]ngle", col) | grepl("Color", col)]
  data[, col] <- data[, col] |>
    bmm::deg2rad() |>
    bmm::wrap()
  data <- dplyr::filter(data, arcsize <= 180) |>
    dplyr::rename(
      resperr = anglediff,
      arc = arcsize
    ) |>
    dplyr::mutate(arc = bmm::deg2rad(arc * 2))

  write.csv(data, output_file, row.names = FALSE)
  data
}
