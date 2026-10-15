import torch
import os

from paths import (
    PROJECT_ROOT, RESULT_FEA_DIR, RESULT_CLUSTER_DIR, RESULT_COORD_DIR,
    RESULT_PRE_MODEL_DIR, RESULT_TRAINED_MODEL_DIR, RESULT_SUMMARY_CSV,
    r_literal,
)

# 单变量og_rank_test
def log_rank_test(subtype):
    r_root = r_literal(PROJECT_ROOT)
    r_code = f"""
    # 忽视警告
    suppressMessages(suppressWarnings({{
    Sys.setlocale(category = 'LC_ALL', locale = 'English_United States.1252')
    library(survival)
    library(glmnet)
    library(parallel)
    library(ggplot2)

    R_ROOT <- "{r_root}"
    subtype = "{subtype}"

    # 读取生存数据
    survival.data=read.table(file.path("{R_ROOT}/clinical", paste0(subtype, ".clinic")),header=T,sep=",",check.names=F,row.names=1)
    # 读取聚类结果
    clustering=read.table(file.path("{R_ROOT}/result/cluster", paste0(subtype, ".cluster")),header=T,sep="\t",check.names=F,row.names=1)

    # survival.data增加Death，Survival列
    survival.data$Death=survival.data$status
    survival.data$Survival=as.numeric(survival.data$days)

    patient.names = as.character(rownames(clustering))
    patient.names.in.file = as.character(rownames(survival.data))


    # 验证 patient.names 列表中的所有元素是否都存在于 patient.names.in.file 列表中
    stopifnot(all(patient.names %in% patient.names.in.file))
    
    # match 函数的作用是查找 patient.names 中的每个元素在 patient.names.in.file 中的第一个匹配项的位置（索引）。
    # 如果某个元素在 patient.names.in.file 中没有找到匹配项，则返回 NA。
    indices = match(patient.names, patient.names.in.file)
    ordered.survival.data = survival.data[indices,]

    # ordered.survival.data增加cluster列
    ordered.survival.data["cluster"] <- clustering

    # median_value <- median(ordered.survival.data$Survival, na.rm = TRUE)

    # 计算ordered.survival.data$Survival的均值用于填充
    mean_value <- mean(ordered.survival.data$Survival, na.rm = TRUE)

    # 将Death，Survival数值小于0或者为NA的数值改为0和均值
    ordered.survival.data$Survival[is.na(ordered.survival.data$Survival)] <- mean_value
    ordered.survival.data$Survival[ordered.survival.data$Survival <0] <- mean_value
    ordered.survival.data$Death[is.na(ordered.survival.data$Death)] <- 0

    # 单变量log_rank_test
    log_rank_test <- survdiff(Surv(Survival, Death) ~ cluster, data=ordered.survival.data)
    chisq_value <- log_rank_test$chisq
    # 提取自由度
    df <- length(log_rank_test$n) - 1
    # 计算 p 值
    p_value <- pchisq(chisq_value, df, lower.tail = FALSE)
    p_value
    }}))
    """
    return r_code

# 临床参数富集分析
def clinical_parameters_enrichment_analysis(subtype):
    r_root = r_literal(PROJECT_ROOT)
    r_code = f"""
    # 忽视警告
    suppressMessages(suppressWarnings({{
    library(parallel)
    R_ROOT <- "{r_root}"
    subtype = "{subtype}"

    clin<-read.csv(file.path("{R_ROOT}/clinical", paste0(subtype, ".clinic")),header=T)

    clust<-read.table(file.path("{R_ROOT}/result/cluster", paste0(subtype, ".cluster")),header=T,sep="\t")

    # 将列名设置为bcr_patient_barcode(第一列)
    rownames(clin)=clin$bcr_patient_barcode  #要根据实际的文本改
    # 去掉第一列
    clin<-clin[,-1]

    rownames(clust)=clust$X  #要根据实际的文本改
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
    get.clinical <- function(clustering, clinical.values, is.chisq) {{
        set.seed(42)
        if (is.chisq) {{
            clustering.with.clinical = cbind(clustering, clinical.values)
            tbl = table(as.data.frame(clustering.with.clinical))
            test.res = chisq.test(tbl)
        }} else {{
            test.res = kruskal.test(as.numeric(clinical.values), clustering)
        }}
      orig.pvalue = test.res$p.value

      return(orig.pvalue)
    }}
    pvalues = c()
    params_tested = c()
    for (clinical_param in names(clinical_metadata)) {{
        # print(clinical_param)
        if (!(clinical_param %in% colnames(clin))) {{
            next
        }}

        is_discrete_param <- clinical_metadata[clinical_param] == 'DISCRETE'
        is_numeric_param <- clinical_metadata[clinical_param] == 'NUMERIC'
        stopifnot(is_discrete_param | is_numeric_param)

        df = clin[,c(clinical_param, "cluster")]

        if (clinical_param == "pahtologic_M") {{
            df$pathologic_M = gsub("a", "", df$pathologic_M)
            df$pathologic_M = gsub("b", "", df$pathologic_M)
            df$pathologic_M = gsub("c", "", df$pathologic_M)
            df$pathologic_M = gsub(" \\\\(i\\\\+\\\\)", "", df$pathologic_M)
            df[which(!df$pathologic_M %in% c("MX", "M0", "M1")),1] = NA
        }} else if (clinical_param == "pahtologic_T") {{
            df[which(df$pathologic_T == "[Discrepancy]"),1] = NA
            df$pathologic_T = gsub("a", "", df$pathologic_T)
            df$pathologic_T = gsub("b", "", df$pathologic_T)
            df$pathologic_T = gsub("c", "", df$pathologic_T)
            df$pathologic_T = gsub("d", "", df$pathologic_T)
            df[which(!df$pathologic_T %in% c("TX", "T0", "T1", "T2", "T3", "T4", "Tis")),1] = NA
        }} else if (clinical_param == "pathologic_N") {{
            df$pathologic_N = gsub("a", "", df$pathologic_N)
            df$pathologic_N = gsub("b", "", df$pathologic_N)
            df$pathologic_N = gsub("c", "", df$pathologic_N)
            df$pathologic_N = gsub("mi", "", df$pathologic_N)
            df$pathologic_N = gsub(" \\\\(i\\\\+\\\\)", "", df$pathologic_N)
            df$pathologic_N = gsub(" \\\\(i\\\\-\\\\)", "", df$pathologic_N)
            df$pathologic_N = gsub(" \\\\(mol\\\\+\\\\)", "", df$pathologic_N)
            df[which(!df$pathologic_N %in% c("NX", "N1", "N2", "N3", "N0")),1] = NA
        }} else if (clinical_param == "pathologic_stage") {{
            df[which(df$pathologic_stage == "[Discrepancy]"),1] = NA
            df$pathologic_stage = gsub("A", "", df$pathologic_stage)
            df$pathologic_stage = gsub("B", "", df$pathologic_stage)
            df$pathologic_stage = gsub("C", "", df$pathologic_stage)
            df[which(!df$pathologic_stage %in% c("Stage X", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV")),1] = NA
        }}

        clinical_values <- df[,clinical_param]

        if (is_numeric_param) {{
            numeric_entries = !is.na(as.numeric(clinical_values)) # Boolean list > TRUE if not NA and FALSE if is NA
            # if too many missing data (more than half), skip the param
            if (2 * sum(numeric_entries) < length(clinical_values)) {{
                next
            }}
        }}
        else {{
            not_na_entries = !is.na(clinical_values)
            should_skip = F
            if (2 * sum(not_na_entries) < length(clinical_values)) {{
            should_skip = T
            }} else if (length(table(clinical_values[not_na_entries])) == 1) {{

            should_skip = T
            }}
            if (should_skip) {{
            next
            }}
        }}

        params_tested = c(params_tested, clinical_param)
        clustering_list <- df$cluster
        names(clustering_list) <- row.names(df)

        if (is_discrete_param) {{
            pvalue = get.clinical(clustering_list[!is.na(clinical_values)], clinical_values[!is.na(clinical_values)], T)
        }} else if (is_numeric_param) {{
            pvalue = get.clinical(clustering_list[numeric_entries], as.numeric(clinical_values[numeric_entries]), F)
        }}

        pvalues = c(pvalues, pvalue)

    }}

    names(pvalues) = params_tested
    # print(pvalues)
    num <- sum(pvalues < 0.05)
    num
    }}))
    """
    return r_code

# 计算软分布Q
def probability_distribution_Q(cluster_feat, cluster_centers, alpha, device):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(cluster_feat.unsqueeze(1) - cluster_centers.to(device), 2), 2) / alpha + 1e-8)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

# 计算目标分布P
def probability_distribution_P(q):
    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p

# 创建文件
def create_file(data_folder):
    """创建各类结果目录，返回本癌种对应的输出文件路径（路径随项目位置自动解析）。"""
    for d in (RESULT_FEA_DIR, RESULT_CLUSTER_DIR, RESULT_COORD_DIR,
              RESULT_PRE_MODEL_DIR, RESULT_TRAINED_MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    result_fea_file = os.path.join(RESULT_FEA_DIR, data_folder + '.fea')
    result_cluster_file = os.path.join(RESULT_CLUSTER_DIR, data_folder + '.cluster')
    result_pre_model_file = os.path.join(RESULT_PRE_MODEL_DIR, data_folder + '.pth')
    result_model_file = os.path.join(RESULT_TRAINED_MODEL_DIR, data_folder + '.pth')
    result_csv_file = RESULT_SUMMARY_CSV

    return result_fea_file, result_cluster_file, result_pre_model_file, result_model_file, result_csv_file