# Fixed-fit Newton path intervention using original, pinned native equations.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=3) stop("Expected cases CSV, pinned source directory, fresh output directory")
if (getRversion()!="3.6.3" || as.character(packageVersion("nleqslv"))!="3.3.2") stop("Unexpected environment")
options(digits=17, warn=-1)
src <- new.env(parent=globalenv())
source(file.path(args[2],"R","calculate_synergy_score.R"),local=src)
cases <- read.csv(args[1],stringsAsFactors=FALSE)
if (dir.exists(args[3])) stop("Output already exists")
dir.create(args[3],recursive=TRUE)
ans <- list()
for (i in seq_len(nrow(cases))) {
  r <- cases[i,];pa <- unlist(r[paste0("a_",c("b","c","d","e"))]);pb <- unlist(r[paste0("b_",c("b","c","d","e"))])
  eq <- get(paste("eq",gsub(".","",r$model_a,fixed=TRUE),gsub(".","",r$model_b,fixed=TRUE),sep="."),src)
  low <- max(min(pa[2:3]),min(pb[2:3]));high <- min(max(pa[2:3]),max(pb[2:3]))
  starts <- c(original=max(pa[2],pb[2])+1,lower_point_one=low+0.1,lower_five=low+5,midpoint=(low+high)/2)
  fallback <- min(max(src$fun(r$dose_a,r$dose_b,pa,r$model_a),src$fun(r$dose_a,r$dose_b,pb,r$model_b)),100)
  for (name in names(starts)) {
    sol <- tryCatch(nleqslv::nleqslv(starts[[name]],eq,method="Newton",x1=r$dose_a,x2=r$dose_b,
                                  drug.col.par=pa,drug.row.par=pb),error=function(e)list(termcd=999,x=NA_real_,fvec=NA_real_))
    ans[[length(ans)+1]] <- data.frame(case_id=r$case_id,start_rule=name,start=starts[[name]],
      termcd=sol$termcd,solution=sol$x,residual=sol$fvec,fallback_reference=fallback,
      reference=min(if(sol$termcd<3)sol$x else fallback,100),stringsAsFactors=FALSE)
  }
}
write.csv(do.call(rbind,ans),file.path(args[3],"starts.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(args[3],"session.txt"))
