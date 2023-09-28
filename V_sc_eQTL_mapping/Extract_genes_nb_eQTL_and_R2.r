#import script arguments
output_workspace <- "/home/p1211536/scratch/NoFastp_Yeast/sc_eQTL/"
df_forward_search <- as.data.frame(readr::read_delim(paste0(output_workspace,"gene_id_0/cross_val.txt"), delim="\t", col_types="iddddd", skip=1,trim_ws=TRUE,col_names = F))
names(df_forward_search) <- c("nb_eQTL","likelihood","RSS_training","R2_training","RSS_validation","R2_validation")
df_forward_search$gene_id <- 0
df_forward_search$AIC <- df_forward_search$likelihood + (log(18233)*df_forward_search$nb_eQTL)
v_R2_genes <- NULL
v_R2_genes <- c(v_R2_genes,df_forward_search$R2_validation[which(df_forward_search$AIC == min(df_forward_search$AIC))])
v_nb_eQTL_genes <- NULL
v_nb_eQTL_genes <- c(v_nb_eQTL_genes,df_forward_search$nb_eQTL[which(df_forward_search$AIC == min(df_forward_search$AIC))])
v_gene_id <- 0
for (i in 1:6239){
  go_to_next <- F
  df_to_concat <- NULL
  df_to_concat <- tryCatch(
    {
      as.data.frame(readr::read_delim(paste0(output_workspace,"gene_id_",i,"/cross_val.txt"), delim="\t", col_types="iddddd", skip=1,trim_ws=TRUE,col_names = F))
    },
    error=function(error_message) {
      go_to_next <<- T
      print(error_message)
      return(NULL)
    }
  )
  if (is.null(df_to_concat)){
    v_R2_genes <- c(v_R2_genes,NA)
    v_nb_eQTL_genes <- c(v_nb_eQTL_genes,NA)
    v_gene_id <- c(v_gene_id,i)
    next()
  }
  
  if ((go_to_next)|((!is.null(df_to_concat))&&(nrow(df_to_concat)==0))){
    v_R2_genes <- c(v_R2_genes,NA)
    v_nb_eQTL_genes <- c(v_nb_eQTL_genes,NA)
    v_gene_id <- c(v_gene_id,i)
    next()
  }
  names(df_to_concat) <- c("nb_eQTL","likelihood","RSS_training","R2_training","RSS_validation","R2_validation")
  df_to_concat$gene_id <- i
  df_to_concat$AIC <- df_to_concat$likelihood + (log(18233)*df_to_concat$nb_eQTL)
  df_to_concat <- subset(df_to_concat, is.finite(AIC)&nb_eQTL>0)
  current_max_R2 <- max(df_to_concat$R2_validation[which(df_to_concat$AIC == min(df_to_concat$AIC))],na.rm=T)
  v_R2_genes <- c(v_R2_genes,current_max_R2 )
  v_nb_eQTL_genes <- c(v_nb_eQTL_genes,max(df_to_concat$nb_eQTL[which((df_to_concat$AIC == min(df_to_concat$AIC))&(df_to_concat$R2_validation == current_max_R2))],na.rm=T))
  v_gene_id <- c(v_gene_id,i)
  print(i)
}

df_out <- data.frame(Gene_id=v_gene_id,R_2=v_R2_genes,nb_eQTL=v_nb_eQTL_genes)
write.table(x=df_out,file = paste0(output_workspace,"Table_genes_R2_and_nb_eQTL.csv"),sep = ",",na = "NA",row.names = FALSE,col.names = TRUE)

