# 项目根目录：请在项目根目录下运行本脚本，或手动指定绝对路径
root <- getwd()
if (!dir.exists(file.path(root, "clinical"))) {
  stop("未找到 clinical 目录，请把 root 设为项目根目录的绝对路径")
}

Sys.setlocale(category = 'LC_ALL', locale = 'English_United States.1252')
library(parallel)

subtype = "SKCM"

clin<-read.csv(file.path(root, "clinical", paste0(subtype, ".clinic")),header=T)
clust<-read.table(file.path(root, "result/cluster", paste0(subtype, ".cluster")), header=T, sep="\t")

rownames(clin)=clin$bcr_patient_barcode  #要根据实际的文本改
clin<-clin[,-1]
rownames(clust)=clust$X  #要根据实际的文本改
# clust
clust=as.matrix(clust)
rownames(clust)=rownames(clin)
clust<-clust[,-1]
# 当从一个拥有行名的数据结构（如数据框或矩阵）中删除列后，列名将会被转换为行名，同时原来的行名将被移除。
# 使用X["行名"]来提取矩阵X中的行，R会返回对应行名的数据以及行名
clinical_metadata = list(gender='DISCRETE', age_at_initial_pathologic_diagnosis='NUMERIC', pathologic_M='DISCRETE',
                         pathologic_N='DISCRETE', pathologic_T='DISCRETE', pathologic_stage='DISCRETE')
# clinical_metadata

clin <- clin[which(names(clin) %in% names(clinical_metadata))]
clin$cluster <- clust
# clin

get.empirical.clinical <- function(clustering, clinical.values, is.chisq) {
  set.seed(42)
  if (is.chisq) {
    clustering.with.clinical = cbind(clustering, clinical.values)
    tbl = table(as.data.frame(clustering.with.clinical))
    test.res = chisq.test(tbl)
  } else {
    test.res = kruskal.test(as.numeric(clinical.values), clustering)
  }
  orig.pvalue = test.res$p.value

  return(orig.pvalue)
}


pvalues = c()
params_tested = c()

for (clinical_param in names(clinical_metadata)) {

  print(clinical_param)

  if (!(clinical_param %in% colnames(clin))) {
    next
  }

  is_discrete_param <- clinical_metadata[clinical_param] == 'DISCRETE'
  is_numeric_param <- clinical_metadata[clinical_param] == 'NUMERIC'
  stopifnot(is_discrete_param | is_numeric_param)

  df = clin[,c(clinical_param, "cluster")]

  if (clinical_param == "pahtologic_M") {
    df$pathologic_M = gsub("a", "", df$pathologic_M)
    df$pathologic_M = gsub("b", "", df$pathologic_M)
    df$pathologic_M = gsub("c", "", df$pathologic_M)
    df$pathologic_M = gsub(" \\(i\\+\\)", "", df$pathologic_M)
    df[which(!df$pathologic_M %in% c("MX", "M0", "M1")),1] = NA
  } else if (clinical_param == "pahtologic_T") {
    df[which(df$pathologic_T == "[Discrepancy]"),1] = NA
    df$pathologic_T = gsub("a", "", df$pathologic_T)
    df$pathologic_T = gsub("b", "", df$pathologic_T)
    df$pathologic_T = gsub("c", "", df$pathologic_T)
    df$pathologic_T = gsub("d", "", df$pathologic_T)
    df[which(!df$pathologic_T %in% c("TX", "T0", "T1", "T2", "T3", "T4", "Tis")),1] = NA
  } else if (clinical_param == "pathologic_N") {
    df$pathologic_N = gsub("a", "", df$pathologic_N)
    df$pathologic_N = gsub("b", "", df$pathologic_N)
    df$pathologic_N = gsub("c", "", df$pathologic_N)
    df$pathologic_N = gsub("mi", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(i\\+\\)", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(i\\-\\)", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(mol\\+\\)", "", df$pathologic_N)
    df[which(!df$pathologic_N %in% c("NX", "N1", "N2", "N3", "N0")),1] = NA
  } else if (clinical_param == "pathologic_stage") {
    df[which(df$pathologic_stage == "[Discrepancy]"),1] = NA
    df$pathologic_stage = gsub("A", "", df$pathologic_stage)
    df$pathologic_stage = gsub("B", "", df$pathologic_stage)
    df$pathologic_stage = gsub("C", "", df$pathologic_stage)
    df[which(!df$pathologic_stage %in% c("Stage X", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV")),1] = NA
  }


  clinical_values <- df[,clinical_param]

  if (is_numeric_param) {
    numeric_entries = !is.na(as.numeric(clinical_values)) # Boolean list > TRUE if not NA and FALSE if is NA
    # if too many missing data (more than half), skip the param
    if (2 * sum(numeric_entries) < length(clinical_values)) {
      next
    }
  }
  else {
    not_na_entries = !is.na(clinical_values)
    should_skip = F
    if (2 * sum(not_na_entries) < length(clinical_values)) {
      should_skip = T
    } else if (length(table(clinical_values[not_na_entries])) == 1) {

      should_skip = T
    }
    if (should_skip) {
      next
    }
  }

  params_tested = c(params_tested, clinical_param)

  clustering_list <- df$cluster
  # clustering_list
  # row.names(df)
  names(clustering_list) <- row.names(df)

  if (is_discrete_param) {
    pvalue = get.empirical.clinical(clustering_list[!is.na(clinical_values)], clinical_values[!is.na(clinical_values)], T)
  } else if (is_numeric_param) {
    pvalue = get.empirical.clinical(clustering_list[numeric_entries], as.numeric(clinical_values[numeric_entries]), F)
  }

  pvalues = c(pvalues, pvalue)

}

names(pvalues) = params_tested
pvalues

sum(pvalues < 0.05)

# #pvalues
# names(pvalues) = params_tested
# sum(pvalues < 0.01)
# pvalues

