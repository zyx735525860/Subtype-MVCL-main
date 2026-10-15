# 项目根目录：请在项目根目录下运行本脚本，或手动指定绝对路径
root <- getwd()
if (!dir.exists(file.path(root, "clinical"))) {
  stop("未找到 clinical 目录，请把 root 设为项目根目录的绝对路径")
}

Sys.setlocale(category = 'LC_ALL', locale = 'English_United States.1252')
    library(survival)
    library(glmnet)
    library(parallel)
    library(ggplot2)

    subtype = "LUAD"

    # 读取生存数据
    survival.data=read.table(file.path(root, "clinical", paste0(subtype, ".clinic")),header=T,sep=",",check.names=F,row.names=1)
    # 读取聚类结果
    clustering=read.table(file.path(root, "result/cluster", paste0(subtype, ".cluster")),header=T,sep="\t",check.names=F,row.names=1)

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



