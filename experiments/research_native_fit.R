# Native source fitting on label-isolated edge inputs; all output paths are new.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected input directory, pinned source directory, output directory")
if (getRversion() != "3.6.3" || as.character(packageVersion("drc")) != "3.0.1" ||
    as.character(packageVersion("nleqslv")) != "3.3.2") stop("Unexpected numerical environment")
options(digits = 17, warn = 1)
source_env <- new.env(parent = globalenv())
for (name in c("Reshape_data.R", "fit_dose_response_model.R", "calculate_synergy_score.R")) {
  source(file.path(args[2], "R", name), local = source_env)
}
edges <- read.csv(file.path(args[1], "edges.csv"), stringsAsFactors = FALSE)
coordinates <- read.csv(file.path(args[1], "coordinates.csv"), stringsAsFactors = FALSE)
if (dir.exists(args[3])) stop("Output exists")
dir.create(args[3], recursive = TRUE)
fits_out <- list(); cases_out <- list()
for (block in unique(edges$block_id)) {
  fits <- list()
  for (endpoint in c("a", "b")) {
    edge <- edges[edges$block_id == block & edges$endpoint == endpoint, ]
    data <- data.frame(dose = edge$dose, response = edge$response)
    warnings <- character(); error <- ""; started <- proc.time()[3]
    fit <- tryCatch(withCallingHandlers(source_env$FitDoseResponse(data),
      warning = function(w) {warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")}),
      error = function(e) {error <<- conditionMessage(e); NULL})
    valid <- inherits(fit, "drc") && all(is.finite(coef(fit)))
    model <- if (valid) as.character(source_env$FindModelType(fit)) else NA_character_
    par <- if (valid) as.numeric(coef(fit)) else rep(NA_real_, 4)
    # Predictions use the fitted model's scale, not a guessed L.4 EC50 convention.
    fitted <- if (valid) as.numeric(stats::fitted(fit)) else rep(NA_real_, nrow(edge))
    fits_out[[length(fits_out) + 1]] <- data.frame(block_id = block, endpoint = endpoint,
      success = valid, model = model, b = par[1], c = par[2], d = par[3], e = par[4],
      edge_sse = sum((edge$response - fitted)^2), warnings = paste(warnings, collapse = " | "),
      error = error, seconds = proc.time()[3] - started, stringsAsFactors = FALSE)
    fits[[endpoint]] <- list(object = fit, model = model, par = par, valid = valid, edge = edge)
  }
  selected <- coordinates[coordinates$block_id == block, ]
  if (fits$a$valid && fits$b$valid) {
    # Official Loewe sees the edges and dummy zero interiors, never measured combinations.
    mat <- matrix(0, nrow = nrow(fits$b$edge), ncol = nrow(fits$a$edge),
                  dimnames = list(fits$b$edge$dose, fits$a$edge$dose))
    mat[, 1] <- fits$b$edge$response; mat[1, ] <- fits$a$edge$response
    official <- source_env$Loewe(mat, quiet = FALSE,
      drug.col.model = fits$a$object, drug.row.model = fits$b$object)
    options(warn = 1)
    pa <- fits$a$par; pb <- fits$b$par
    equation <- get(paste("eq", gsub(".", "", fits$a$model, fixed = TRUE),
                          gsub(".", "", fits$b$model, fixed = TRUE), sep = "."), source_env)
    for (i in seq_len(nrow(selected))) {
      row <- selected[i, ]; xa <- row$dose_a; xb <- row$dose_b
      solution <- tryCatch(suppressWarnings(nleqslv::nleqslv(max(pa[2], pb[2]) + 1,
        equation, method = "Newton", x1 = xa, x2 = xb, drug.col.par = pa, drug.row.par = pb)),
        error = function(e) list(termcd = 999, x = NA_real_, fvec = NA_real_))
      fallback <- max(source_env$fun(xa, xb, pa, fits$a$model), source_env$fun(xa, xb, pb, fits$b$model))
      own <- max(source_env$fun(xa, 0, pa, fits$a$model), source_env$fun(0, xb, pb, fits$b$model))
      reference <- min(if (solution$termcd < 3) solution$x else fallback, 100)
      alternate <- min(if (solution$termcd < 3) solution$x else own, 100)
      cases_out[[length(cases_out) + 1]] <- data.frame(case_id = row$case_id,
        block_id = block, dose_a = xa, dose_b = xb, native_code = solution$termcd,
        solution = solution$x, equation_residual = solution$fvec,
        native_reference = reference, hsa_fallback_reference = alternate,
        fallback_sum = fallback, own_hsa = own,
        official_reference = -official[as.character(xb), as.character(xa)])
    }
  } else {
    for (i in seq_len(nrow(selected))) {
      row <- selected[i, ]
      cases_out[[length(cases_out) + 1]] <- data.frame(case_id = row$case_id,
        block_id = block, dose_a = row$dose_a, dose_b = row$dose_b, native_code = NA_integer_,
        solution = NA_real_, equation_residual = NA_real_, native_reference = NA_real_,
        hsa_fallback_reference = NA_real_, fallback_sum = NA_real_, own_hsa = NA_real_,
        official_reference = NA_real_)
    }
  }
}
write.csv(do.call(rbind, fits_out), file.path(args[3], "fits.csv"), row.names = FALSE)
write.csv(do.call(rbind, cases_out), file.path(args[3], "cases.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(args[3], "session.txt"))
