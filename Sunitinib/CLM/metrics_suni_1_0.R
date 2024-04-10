# MAPE, RMSE, MAE summary Sunitinib --------------------------------------------

#--------------------------------------------------#
# metrics summary no augmentation                  #                                                 
#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling

rm(list=ls())
library(dplyr) 

setwd("C:/Users/Olga/Suni_Pazo/Quality_metrics")


# NM parent
qual_focei_parent_1<- read.csv(("quality_measures_test_focei_parent_1_0.csv"))
qual_saem_parent_1<- read.csv(("quality_measures_test_saem_parent_1_0.csv"))
qual_focei_parent_2<- read.csv(("quality_measures_test_focei_parent_2_0.csv"))
qual_saem_parent_2<- read.csv(("quality_measures_test_saem_parent_2_0.csv"))
qual_focei_parent_3<- read.csv(("quality_measures_test_focei_parent_3_0.csv"))
qual_saem_parent_3<- read.csv(("quality_measures_test_saem_parent_3_0.csv"))
qual_focei_parent_4<- read.csv(("quality_measures_test_focei_parent_4_0.csv"))
qual_saem_parent_4<- read.csv(("quality_measures_test_saem_parent_4_0.csv"))
qual_focei_parent_5<- read.csv(("quality_measures_test_focei_parent_5_0.csv"))
qual_saem_parent_5<- read.csv(("quality_measures_test_saem_parent_5_0.csv"))
qual_focei_parent_6<- read.csv(("quality_measures_test_focei_parent_6_0.csv"))
qual_saem_parent_6<- read.csv(("quality_measures_test_saem_parent_6_0.csv"))
qual_focei_parent_7<- read.csv(("quality_measures_test_focei_parent_7_0.csv"))
qual_saem_parent_7<- read.csv(("quality_measures_test_saem_parent_7_0.csv"))
qual_focei_parent_8<- read.csv(("quality_measures_test_focei_parent_8_0.csv"))
qual_saem_parent_8<- read.csv(("quality_measures_test_saem_parent_8_0.csv"))
qual_focei_parent_9<- read.csv(("quality_measures_test_focei_parent_9_0.csv"))
qual_saem_parent_9<- read.csv(("quality_measures_test_saem_parent_9_0.csv"))
qual_focei_parent_10<- read.csv(("quality_measures_test_focei_parent_10_0.csv"))
qual_saem_parent_10<- read.csv(("quality_measures_test_saem_parent_10_0.csv"))

# NM metabolite
qual_focei_metabolite_1<- read.csv(("quality_measures_test_focei_metabolite_1_0.csv"))
qual_saem_metabolite_1<- read.csv(("quality_measures_test_saem_metabolite_1_0.csv"))
qual_focei_metabolite_2<- read.csv(("quality_measures_test_focei_metabolite_2_0.csv"))
qual_saem_metabolite_2<- read.csv(("quality_measures_test_saem_metabolite_2_0.csv"))
qual_focei_metabolite_3<- read.csv(("quality_measures_test_focei_metabolite_3_0.csv"))
qual_saem_metabolite_3<- read.csv(("quality_measures_test_saem_metabolite_3_0.csv"))
qual_focei_metabolite_4<- read.csv(("quality_measures_test_focei_metabolite_4_0.csv"))
qual_saem_metabolite_4<- read.csv(("quality_measures_test_saem_metabolite_4_0.csv"))
qual_focei_metabolite_5<- read.csv(("quality_measures_test_focei_metabolite_5_0.csv"))
qual_saem_metabolite_5<- read.csv(("quality_measures_test_saem_metabolite_5_0.csv"))
qual_focei_metabolite_6<- read.csv(("quality_measures_test_focei_metabolite_6_0.csv"))
qual_saem_metabolite_6<- read.csv(("quality_measures_test_saem_metabolite_6_0.csv"))
qual_focei_metabolite_7<- read.csv(("quality_measures_test_focei_metabolite_7_0.csv"))
qual_saem_metabolite_7<- read.csv(("quality_measures_test_saem_metabolite_7_0.csv"))
qual_focei_metabolite_8<- read.csv(("quality_measures_test_focei_metabolite_8_0.csv"))
qual_saem_metabolite_8<- read.csv(("quality_measures_test_saem_metabolite_8_0.csv"))
qual_focei_metabolite_9<- read.csv(("quality_measures_test_focei_metabolite_9_0.csv"))
qual_saem_metabolite_9<- read.csv(("quality_measures_test_saem_metabolite_9_0.csv"))
qual_focei_metabolite_10<- read.csv(("quality_measures_test_focei_metabolite_10_0.csv"))
qual_saem_metabolite_10<- read.csv(("quality_measures_test_saem_metabolite_10_0.csv"))

# ML
qual_RF_FS_1<-read.csv(("suni_params_test_Random_Forest_split_1_FS_augmentation_0.csv"))
qual_RF_No_FS_1<-read.csv(("suni_params_test_Random_Forest_split_1_No_FS_augmentation_0.csv"))
qual_GB_FS_1<-read.csv(("suni_params_test_Gradient_Boosting_split1_FS_augmentation_0.csv"))
qual_GB_No_FS_1<-read.csv(("suni_params_test_Gradient_Boosting_split1_No_FS_augmentation_0.csv"))
qual_SVM_FS_1<-read.csv(("suni_params_test_Support_Vector_Machine_split_1_FS_augmentation_0.csv"))
qual_SVM_No_FS_1<-read.csv(("suni_params_test_Support_Vector_Machine_split_1_No_FS_augmentation_0.csv"))
qual_XGB_FS_1<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_1_FS_augmentation_0.csv"))
qual_XGB_No_FS_1<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_1_No_FS_augmentation_0.csv"))
qual_LGB_FS_1<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_1_FS_augmentation_0.csv"))
qual_LGB_No_FS_1<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_1_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_1<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_1_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_1<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_1_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_1<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_1_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_1<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_1_No_FS_augmentation_0.csv"))

qual_RF_FS_2<-read.csv(("suni_params_test_Random_Forest_split_2_FS_augmentation_0.csv"))
qual_RF_No_FS_2<-read.csv(("suni_params_test_Random_Forest_split_2_No_FS_augmentation_0.csv"))
qual_GB_FS_2<-read.csv(("suni_params_test_Gradient_Boosting_split2_FS_augmentation_0.csv"))
qual_GB_No_FS_2<-read.csv(("suni_params_test_Gradient_Boosting_split2_No_FS_augmentation_0.csv"))
qual_SVM_FS_2<-read.csv(("suni_params_test_Support_Vector_Machine_split_2_FS_augmentation_0.csv"))
qual_SVM_No_FS_2<-read.csv(("suni_params_test_Support_Vector_Machine_split_2_No_FS_augmentation_0.csv"))
qual_XGB_FS_2<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_2_FS_augmentation_0.csv"))
qual_XGB_No_FS_2<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_2_No_FS_augmentation_0.csv"))
qual_LGB_FS_2<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_2_FS_augmentation_0.csv"))
qual_LGB_No_FS_2<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_2_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_2<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_2_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_2<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_2_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_2<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_2_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_2<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_2_No_FS_augmentation_0.csv"))

qual_RF_FS_3<-read.csv(("suni_params_test_Random_Forest_split_3_FS_augmentation_0.csv"))
qual_RF_No_FS_3<-read.csv(("suni_params_test_Random_Forest_split_3_No_FS_augmentation_0.csv"))
qual_GB_FS_3<-read.csv(("suni_params_test_Gradient_Boosting_split3_FS_augmentation_0.csv"))
qual_GB_No_FS_3<-read.csv(("suni_params_test_Gradient_Boosting_split3_No_FS_augmentation_0.csv"))
qual_SVM_FS_3<-read.csv(("suni_params_test_Support_Vector_Machine_split_3_FS_augmentation_0.csv"))
qual_SVM_No_FS_3<-read.csv(("suni_params_test_Support_Vector_Machine_split_3_No_FS_augmentation_0.csv"))
qual_XGB_FS_3<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_3_FS_augmentation_0.csv"))
qual_XGB_No_FS_3<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_3_No_FS_augmentation_0.csv"))
qual_LGB_FS_3<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_3_FS_augmentation_0.csv"))
qual_LGB_No_FS_3<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_3_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_3<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_3_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_3<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_3_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_3<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_3_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_3<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_3_No_FS_augmentation_0.csv"))

qual_RF_FS_4<-read.csv(("suni_params_test_Random_Forest_split_4_FS_augmentation_0.csv"))
qual_RF_No_FS_4<-read.csv(("suni_params_test_Random_Forest_split_4_No_FS_augmentation_0.csv"))
qual_GB_FS_4<-read.csv(("suni_params_test_Gradient_Boosting_split4_FS_augmentation_0.csv"))
qual_GB_No_FS_4<-read.csv(("suni_params_test_Gradient_Boosting_split4_No_FS_augmentation_0.csv"))
qual_SVM_FS_4<-read.csv(("suni_params_test_Support_Vector_Machine_split_4_FS_augmentation_0.csv"))
qual_SVM_No_FS_4<-read.csv(("suni_params_test_Support_Vector_Machine_split_4_No_FS_augmentation_0.csv"))
qual_XGB_FS_4<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_4_FS_augmentation_0.csv"))
qual_XGB_No_FS_4<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_4_No_FS_augmentation_0.csv"))
qual_LGB_FS_4<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_4_FS_augmentation_0.csv"))
qual_LGB_No_FS_4<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_4_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_4<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_4_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_4<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_4_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_4<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_4_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_4<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_4_No_FS_augmentation_0.csv"))

qual_RF_FS_5<-read.csv(("suni_params_test_Random_Forest_split_5_FS_augmentation_0.csv"))
qual_RF_No_FS_5<-read.csv(("suni_params_test_Random_Forest_split_5_No_FS_augmentation_0.csv"))
qual_GB_FS_5<-read.csv(("suni_params_test_Gradient_Boosting_split5_FS_augmentation_0.csv"))
qual_GB_No_FS_5<-read.csv(("suni_params_test_Gradient_Boosting_split5_No_FS_augmentation_0.csv"))
qual_SVM_FS_5<-read.csv(("suni_params_test_Support_Vector_Machine_split_5_FS_augmentation_0.csv"))
qual_SVM_No_FS_5<-read.csv(("suni_params_test_Support_Vector_Machine_split_5_No_FS_augmentation_0.csv"))
qual_XGB_FS_5<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_5_FS_augmentation_0.csv"))
qual_XGB_No_FS_5<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_5_No_FS_augmentation_0.csv"))
qual_LGB_FS_5<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_5_FS_augmentation_0.csv"))
qual_LGB_No_FS_5<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_5_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_5<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_5_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_5<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_5_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_5<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_5_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_5<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_5_No_FS_augmentation_0.csv"))

qual_RF_FS_6<-read.csv(("suni_params_test_Random_Forest_split_6_FS_augmentation_0.csv"))
qual_RF_No_FS_6<-read.csv(("suni_params_test_Random_Forest_split_6_No_FS_augmentation_0.csv"))
qual_GB_FS_6<-read.csv(("suni_params_test_Gradient_Boosting_split6_FS_augmentation_0.csv"))
qual_GB_No_FS_6<-read.csv(("suni_params_test_Gradient_Boosting_split6_No_FS_augmentation_0.csv"))
qual_SVM_FS_6<-read.csv(("suni_params_test_Support_Vector_Machine_split_6_FS_augmentation_0.csv"))
qual_SVM_No_FS_6<-read.csv(("suni_params_test_Support_Vector_Machine_split_6_No_FS_augmentation_0.csv"))
qual_XGB_FS_6<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_6_FS_augmentation_0.csv"))
qual_XGB_No_FS_6<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_6_No_FS_augmentation_0.csv"))
qual_LGB_FS_6<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_6_FS_augmentation_0.csv"))
qual_LGB_No_FS_6<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_6_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_6<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_6_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_6<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_6_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_6<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_6_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_6<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_6_No_FS_augmentation_0.csv"))

qual_RF_FS_7<-read.csv(("suni_params_test_Random_Forest_split_7_FS_augmentation_0.csv"))
qual_RF_No_FS_7<-read.csv(("suni_params_test_Random_Forest_split_7_No_FS_augmentation_0.csv"))
qual_GB_FS_7<-read.csv(("suni_params_test_Gradient_Boosting_split7_FS_augmentation_0.csv"))
qual_GB_No_FS_7<-read.csv(("suni_params_test_Gradient_Boosting_split7_No_FS_augmentation_0.csv"))
qual_SVM_FS_7<-read.csv(("suni_params_test_Support_Vector_Machine_split_7_FS_augmentation_0.csv"))
qual_SVM_No_FS_7<-read.csv(("suni_params_test_Support_Vector_Machine_split_7_No_FS_augmentation_0.csv"))
qual_XGB_FS_7<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_7_FS_augmentation_0.csv"))
qual_XGB_No_FS_7<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_7_No_FS_augmentation_0.csv"))
qual_LGB_FS_7<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_7_FS_augmentation_0.csv"))
qual_LGB_No_FS_7<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_7_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_7<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_7_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_7<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_7_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_7<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_7_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_7<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_7_No_FS_augmentation_0.csv"))

qual_RF_FS_8<-read.csv(("suni_params_test_Random_Forest_split_8_FS_augmentation_0.csv"))
qual_RF_No_FS_8<-read.csv(("suni_params_test_Random_Forest_split_8_No_FS_augmentation_0.csv"))
qual_GB_FS_8<-read.csv(("suni_params_test_Gradient_Boosting_split8_FS_augmentation_0.csv"))
qual_GB_No_FS_8<-read.csv(("suni_params_test_Gradient_Boosting_split8_No_FS_augmentation_0.csv"))
qual_SVM_FS_8<-read.csv(("suni_params_test_Support_Vector_Machine_split_8_FS_augmentation_0.csv"))
qual_SVM_No_FS_8<-read.csv(("suni_params_test_Support_Vector_Machine_split_8_No_FS_augmentation_0.csv"))
qual_XGB_FS_8<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_8_FS_augmentation_0.csv"))
qual_XGB_No_FS_8<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_8_No_FS_augmentation_0.csv"))
qual_LGB_FS_8<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_8_FS_augmentation_0.csv"))
qual_LGB_No_FS_8<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_8_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_8<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_8_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_8<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_8_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_8<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_8_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_8<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_8_No_FS_augmentation_0.csv"))

qual_RF_FS_9<-read.csv(("suni_params_test_Random_Forest_split_9_FS_augmentation_0.csv"))
qual_RF_No_FS_9<-read.csv(("suni_params_test_Random_Forest_split_9_No_FS_augmentation_0.csv"))
qual_GB_FS_9<-read.csv(("suni_params_test_Gradient_Boosting_split9_FS_augmentation_0.csv"))
qual_GB_No_FS_9<-read.csv(("suni_params_test_Gradient_Boosting_split9_No_FS_augmentation_0.csv"))
qual_SVM_FS_9<-read.csv(("suni_params_test_Support_Vector_Machine_split_9_FS_augmentation_0.csv"))
qual_SVM_No_FS_9<-read.csv(("suni_params_test_Support_Vector_Machine_split_9_No_FS_augmentation_0.csv"))
qual_XGB_FS_9<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_9_FS_augmentation_0.csv"))
qual_XGB_No_FS_9<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_9_No_FS_augmentation_0.csv"))
qual_LGB_FS_9<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_9_FS_augmentation_0.csv"))
qual_LGB_No_FS_9<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_9_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_9<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_9_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_9<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_9_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_9<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_9_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_9<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_9_No_FS_augmentation_0.csv"))

qual_RF_FS_10<-read.csv(("suni_params_test_Random_Forest_split_10_FS_augmentation_0.csv"))
qual_RF_No_FS_10<-read.csv(("suni_params_test_Random_Forest_split_10_No_FS_augmentation_0.csv"))
qual_GB_FS_10<-read.csv(("suni_params_test_Gradient_Boosting_split10_FS_augmentation_0.csv"))
qual_GB_No_FS_10<-read.csv(("suni_params_test_Gradient_Boosting_split10_No_FS_augmentation_0.csv"))
qual_SVM_FS_10<-read.csv(("suni_params_test_Support_Vector_Machine_split_10_FS_augmentation_0.csv"))
qual_SVM_No_FS_10<-read.csv(("suni_params_test_Support_Vector_Machine_split_10_No_FS_augmentation_0.csv"))
qual_XGB_FS_10<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_10_FS_augmentation_0.csv"))
qual_XGB_No_FS_10<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_10_No_FS_augmentation_0.csv"))
qual_LGB_FS_10<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_10_FS_augmentation_0.csv"))
qual_LGB_No_FS_10<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_10_No_FS_augmentation_0.csv"))
qual_MLP_2HL_FS_10<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_10_FS_augmentation_0.csv"))
qual_MLP_2HL_No_FS_10<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_10_No_FS_augmentation_0.csv"))
qual_MLP_1HL_FS_10<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_10_FS_augmentation_0.csv"))
qual_MLP_1HL_No_FS_10<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_10_No_FS_augmentation_0.csv"))

# FOCEI parent -----------------------------------------------------------------

focei <- rbind(qual_focei_parent_1, qual_focei_parent_2, qual_focei_parent_3, qual_focei_parent_4, qual_focei_parent_5,
               qual_focei_parent_6, qual_focei_parent_7, qual_focei_parent_8, qual_focei_parent_9,
               qual_focei_parent_10)

# Replace all Inf values with NA
focei[focei == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_focei_parent <- mean(focei$MAPE, na.rm = TRUE)
mape_sd_focei_parent <- sd(focei$MAPE, na.rm = TRUE)

rmse_mean_focei_parent <- mean(focei$RMSE, na.rm = TRUE)
rmse_sd_focei_parent <- sd(focei$RMSE, na.rm = TRUE)

mae_mean_focei_parent <- mean(focei$MAE, na.rm = TRUE)
mae_sd_focei_parent <- sd(focei$MAE, na.rm = TRUE)

# FOCEI metabolite -------------------------------------------------------------

focei <- rbind(qual_focei_metabolite_1, qual_focei_metabolite_2, qual_focei_metabolite_3, qual_focei_metabolite_4, qual_focei_metabolite_5,
               qual_focei_metabolite_6, qual_focei_metabolite_7, qual_focei_metabolite_8, qual_focei_metabolite_9,
               qual_focei_metabolite_10)

# Replace all Inf values with NA
focei[focei == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_focei_metabolite <- mean(focei$MAPE, na.rm = TRUE)
mape_sd_focei_metabolite <- sd(focei$MAPE, na.rm = TRUE)

rmse_mean_focei_metabolite <- mean(focei$RMSE, na.rm = TRUE)
rmse_sd_focei_metabolite <- sd(focei$RMSE, na.rm = TRUE)

mae_mean_focei_metabolite <- mean(focei$MAE, na.rm = TRUE)
mae_sd_focei_metabolite <- sd(focei$MAE, na.rm = TRUE)


# SAEM parent ------------------------------------------------------------------

saem <- rbind(qual_saem_parent_1, qual_saem_parent_2, qual_saem_parent_3, qual_saem_parent_4, qual_saem_parent_5,
               qual_saem_parent_6, qual_saem_parent_7, qual_saem_parent_8, qual_saem_parent_9,
               qual_saem_parent_10)

# Replace all Inf values with NA
saem[saem == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_saem_parent <- mean(saem$MAPE, na.rm = TRUE)
mape_sd_saem_parent <- sd(saem$MAPE, na.rm = TRUE)

rmse_mean_saem_parent <- mean(saem$RMSE, na.rm = TRUE)
rmse_sd_saem_parent <- sd(saem$RMSE, na.rm = TRUE)

mae_mean_saem_parent <- mean(saem$MAE, na.rm = TRUE)
mae_sd_saem_parent <- sd(saem$MAE, na.rm = TRUE)


# SAEM metabolite --------------------------------------------------------------

saem <- rbind(qual_saem_metabolite_1, qual_saem_metabolite_2, qual_saem_metabolite_3, qual_saem_metabolite_4, qual_saem_metabolite_5,
                qual_saem_metabolite_6, qual_saem_metabolite_7, qual_saem_metabolite_8, qual_saem_metabolite_9,
                qual_saem_metabolite_10)

# Replace all Inf values with NA
saem[saem == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_saem_metabolite <- mean(saem$MAPE, na.rm = TRUE)
mape_sd_saem_metabolite <- sd(saem$MAPE, na.rm = TRUE)

rmse_mean_saem_metabolite <- mean(saem$RMSE, na.rm = TRUE)
rmse_sd_saem_metabolite <- sd(saem$RMSE, na.rm = TRUE)

mae_mean_saem_metabolite <- mean(saem$MAE, na.rm = TRUE)
mae_sd_saem_metabolite <- sd(saem$MAE, na.rm = TRUE)

# RF FS ------------------------------------------------------------------------

RF_FS <- rbind(qual_RF_FS_1, qual_RF_FS_2, qual_RF_FS_3, qual_RF_FS_4, qual_RF_FS_5,
              qual_RF_FS_6, qual_RF_FS_7, qual_RF_FS_8, qual_RF_FS_9,
              qual_RF_FS_10)

# Replace all Inf values with NA
RF_FS[RF_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_RF_FS <- mean(RF_FS$MAPE, na.rm = TRUE)*100
mape_sd_RF_FS <- sd(RF_FS$MAPE, na.rm = TRUE)*100

rmse_mean_RF_FS <- mean(RF_FS$RMSE, na.rm = TRUE)
rmse_sd_RF_FS <- sd(RF_FS$RMSE, na.rm = TRUE)

mae_mean_RF_FS <- mean(RF_FS$MAE, na.rm = TRUE)
mae_sd_RF_FS <- sd(RF_FS$MAE, na.rm = TRUE)

# RF No FS ---------------------------------------------------------------------

RF_No_FS <- rbind(qual_RF_No_FS_1, qual_RF_No_FS_2, qual_RF_No_FS_3, qual_RF_No_FS_4, qual_RF_No_FS_5,
              qual_RF_No_FS_6, qual_RF_No_FS_7, qual_RF_No_FS_8, qual_RF_No_FS_9,
              qual_RF_No_FS_10)

# Replace all Inf values with NA
RF_No_FS[RF_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_RF_No_FS <- mean(RF_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_RF_No_FS <- sd(RF_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_RF_No_FS <- mean(RF_No_FS$RMSE, na.rm = TRUE)
rmse_sd_RF_No_FS <- sd(RF_No_FS$RMSE, na.rm = TRUE)

mae_mean_RF_No_FS <- mean(RF_No_FS$MAE, na.rm = TRUE)
mae_sd_RF_No_FS <- sd(RF_No_FS$MAE, na.rm = TRUE)

# GB FS ---------------------------------------------------------------------

GB_FS <- rbind(qual_GB_FS_1, qual_GB_FS_2, qual_GB_FS_3, qual_GB_FS_4, qual_GB_FS_5,
                  qual_GB_FS_6, qual_GB_FS_7, qual_GB_FS_8, qual_GB_FS_9,
                  qual_GB_FS_10)

# Replace all Inf values with NA
GB_FS[GB_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_GB_FS <- mean(GB_FS$MAPE, na.rm = TRUE)*100
mape_sd_GB_FS <- sd(GB_FS$MAPE, na.rm = TRUE)*100

rmse_mean_GB_FS <- mean(GB_FS$RMSE, na.rm = TRUE)
rmse_sd_GB_FS <- sd(GB_FS$RMSE, na.rm = TRUE)

mae_mean_GB_FS <- mean(GB_FS$MAE, na.rm = TRUE)
mae_sd_GB_FS <- sd(GB_FS$MAE, na.rm = TRUE)

# GB No FS ---------------------------------------------------------------------

GB_No_FS <- rbind(qual_GB_No_FS_1, qual_GB_No_FS_2, qual_GB_No_FS_3, qual_GB_No_FS_4, qual_GB_No_FS_5,
              qual_GB_No_FS_6, qual_GB_No_FS_7, qual_GB_No_FS_8, qual_GB_No_FS_9,
              qual_GB_No_FS_10)

# Replace all Inf values with NA
GB_No_FS[GB_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_GB_No_FS <- mean(GB_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_GB_No_FS <- sd(GB_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_GB_No_FS <- mean(GB_No_FS$RMSE, na.rm = TRUE)
rmse_sd_GB_No_FS <- sd(GB_No_FS$RMSE, na.rm = TRUE)

mae_mean_GB_No_FS <- mean(GB_No_FS$MAE, na.rm = TRUE)
mae_sd_GB_No_FS <- sd(GB_No_FS$MAE, na.rm = TRUE)


# SVM FS ------------------------------------------------------------------------

SVM_FS <- rbind(qual_SVM_FS_1, qual_SVM_FS_2, qual_SVM_FS_3, qual_SVM_FS_4, qual_SVM_FS_5,
              qual_SVM_FS_6, qual_SVM_FS_7, qual_SVM_FS_8, qual_SVM_FS_9,
              qual_SVM_FS_10)

# Replace all Inf values with NA
SVM_FS[SVM_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_SVM_FS <- mean(SVM_FS$MAPE, na.rm = TRUE)*100
mape_sd_SVM_FS <- sd(SVM_FS$MAPE, na.rm = TRUE)*100

rmse_mean_SVM_FS <- mean(SVM_FS$RMSE, na.rm = TRUE)
rmse_sd_SVM_FS <- sd(SVM_FS$RMSE, na.rm = TRUE)

mae_mean_SVM_FS <- mean(SVM_FS$MAE, na.rm = TRUE)
mae_sd_SVM_FS <- sd(SVM_FS$MAE, na.rm = TRUE)

# SVM No FS ---------------------------------------------------------------------

SVM_No_FS <- rbind(qual_SVM_No_FS_1, qual_SVM_No_FS_2, qual_SVM_No_FS_3, qual_SVM_No_FS_4, qual_SVM_No_FS_5,
              qual_SVM_No_FS_6, qual_SVM_No_FS_7, qual_SVM_No_FS_8, qual_SVM_No_FS_9,
              qual_SVM_No_FS_10)

# Replace all Inf values with NA
SVM_No_FS[SVM_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_SVM_No_FS <- mean(SVM_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_SVM_No_FS <- sd(SVM_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_SVM_No_FS <- mean(SVM_No_FS$RMSE, na.rm = TRUE)
rmse_sd_SVM_No_FS <- sd(SVM_No_FS$RMSE, na.rm = TRUE)

mae_mean_SVM_No_FS <- mean(SVM_No_FS$MAE, na.rm = TRUE)
mae_sd_SVM_No_FS <- sd(SVM_No_FS$MAE, na.rm = TRUE)

# XGB FS ---------------------------------------------------------------------

XGB_FS <- rbind(qual_XGB_FS_1, qual_XGB_FS_2, qual_XGB_FS_3, qual_XGB_FS_4, qual_XGB_FS_5,
               qual_XGB_FS_6, qual_XGB_FS_7, qual_XGB_FS_8, qual_XGB_FS_9,
               qual_XGB_FS_10)

# Replace all Inf values with NA
XGB_FS[XGB_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_XGB_FS <- mean(XGB_FS$MAPE, na.rm = TRUE)*100
mape_sd_XGB_FS <- sd(XGB_FS$MAPE, na.rm = TRUE)*100

rmse_mean_XGB_FS <- mean(XGB_FS$RMSE, na.rm = TRUE)
rmse_sd_XGB_FS <- sd(XGB_FS$RMSE, na.rm = TRUE)

mae_mean_XGB_FS <- mean(XGB_FS$MAE, na.rm = TRUE)
mae_sd_XGB_FS <- sd(XGB_FS$MAE, na.rm = TRUE)

# XGB No FS ---------------------------------------------------------------------

XGB_No_FS <- rbind(qual_XGB_No_FS_1, qual_XGB_No_FS_2, qual_XGB_No_FS_3, qual_XGB_No_FS_4, qual_XGB_No_FS_5,
              qual_XGB_No_FS_6, qual_XGB_No_FS_7, qual_XGB_No_FS_8, qual_XGB_No_FS_9,
              qual_XGB_No_FS_10)

# Replace all Inf values with NA
XGB_No_FS[XGB_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_XGB_No_FS <- mean(XGB_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_XGB_No_FS <- sd(XGB_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_XGB_No_FS <- mean(XGB_No_FS$RMSE, na.rm = TRUE)
rmse_sd_XGB_No_FS <- sd(XGB_No_FS$RMSE, na.rm = TRUE)

mae_mean_XGB_No_FS <- mean(XGB_FS$MAE, na.rm = TRUE)
mae_sd_XGB_No_FS <- sd(XGB_FS$MAE, na.rm = TRUE)

# LGB FS ---------------------------------------------------------------------

LGB_FS <- rbind(qual_LGB_FS_1, qual_LGB_FS_2, qual_LGB_FS_3, qual_LGB_FS_4, qual_LGB_FS_5,
               qual_LGB_FS_6, qual_LGB_FS_7, qual_LGB_FS_8, qual_LGB_FS_9,
               qual_LGB_FS_10)

# Replace all Inf values with NA
LGB_FS[LGB_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_LGB_FS <- mean(LGB_FS$MAPE, na.rm = TRUE)*100
mape_sd_LGB_FS <- sd(LGB_FS$MAPE, na.rm = TRUE)*100

rmse_mean_LGB_FS <- mean(LGB_FS$RMSE, na.rm = TRUE)
rmse_sd_LGB_FS <- sd(LGB_FS$RMSE, na.rm = TRUE)

mae_mean_LGB_FS <- mean(LGB_FS$MAE, na.rm = TRUE)
mae_sd_LGB_FS <- sd(LGB_FS$MAE, na.rm = TRUE)

# LGB No FS ---------------------------------------------------------------------

LGB_No_FS <- rbind(qual_LGB_No_FS_1, qual_LGB_No_FS_2, qual_LGB_No_FS_3, qual_LGB_No_FS_4, qual_LGB_No_FS_5,
              qual_LGB_No_FS_6, qual_LGB_No_FS_7, qual_LGB_No_FS_8, qual_LGB_No_FS_9,
              qual_LGB_No_FS_10)

# Replace all Inf values with NA
LGB_No_FS[LGB_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_LGB_No_FS <- mean(LGB_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_LGB_No_FS <- sd(LGB_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_LGB_No_FS <- mean(LGB_No_FS$RMSE, na.rm = TRUE)
rmse_sd_LGB_No_FS <- sd(LGB_No_FS$RMSE, na.rm = TRUE)

mae_mean_LGB_No_FS <- mean(LGB_No_FS$MAE, na.rm = TRUE)
mae_sd_LGB_No_FS <- sd(LGB_No_FS$MAE, na.rm = TRUE)

# MLP 2 HL FS ------------------------------------------------------------------

MLP_2HL_FS <- rbind(qual_MLP_2HL_FS_1, qual_MLP_2HL_FS_2, qual_MLP_2HL_FS_3, qual_MLP_2HL_FS_4, qual_MLP_2HL_FS_5,
                qual_MLP_2HL_FS_6, qual_MLP_2HL_FS_7, qual_MLP_2HL_FS_8, qual_MLP_2HL_FS_9,
                qual_MLP_2HL_FS_10)

# Replace all Inf values with NA
MLP_2HL_FS[MLP_2HL_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_2HL_FS <- mean(MLP_2HL_FS$MAPE, na.rm = TRUE)*100
mape_sd_MLP_2HL_FS <- sd(MLP_2HL_FS$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_2HL_FS <- mean(MLP_2HL_FS$RMSE, na.rm = TRUE)
rmse_sd_MLP_2HL_FS <- sd(MLP_2HL_FS$RMSE, na.rm = TRUE)

mae_mean_MLP_2HL_FS <- mean(MLP_2HL_FS$MAE, na.rm = TRUE)
mae_sd_MLP_2HL_FS <- sd(MLP_2HL_FS$MAE, na.rm = TRUE)

# MLP 2 HL No FS ---------------------------------------------------------------

MLP_2HL_No_FS <- rbind(qual_MLP_2HL_No_FS_1, qual_MLP_2HL_No_FS_2, qual_MLP_2HL_No_FS_3, qual_MLP_2HL_No_FS_4, qual_MLP_2HL_No_FS_5,
                   qual_MLP_2HL_No_FS_6, qual_MLP_2HL_No_FS_7, qual_MLP_2HL_No_FS_8, qual_MLP_2HL_No_FS_9,
                   qual_MLP_2HL_No_FS_10)

# Replace all Inf values with NA
MLP_2HL_No_FS[MLP_2HL_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_2HL_No_FS <- mean(MLP_2HL_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_MLP_2HL_No_FS <- sd(MLP_2HL_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_2HL_No_FS <- mean(MLP_2HL_No_FS$RMSE, na.rm = TRUE)
rmse_sd_MLP_2HL_No_FS <- sd(MLP_2HL_No_FS$RMSE, na.rm = TRUE)

mae_mean_MLP_2HL_No_FS <- mean(MLP_2HL_No_FS$MAE, na.rm = TRUE)
mae_sd_MLP_2HL_No_FS <- sd(MLP_2HL_No_FS$MAE, na.rm = TRUE)


# MLP 1 HL FS ------------------------------------------------------------------

MLP_1HL_FS <- rbind(qual_MLP_1HL_FS_1, qual_MLP_1HL_FS_2, qual_MLP_1HL_FS_3, qual_MLP_1HL_FS_4, qual_MLP_1HL_FS_5,
                    qual_MLP_1HL_FS_6, qual_MLP_1HL_FS_7, qual_MLP_1HL_FS_8, qual_MLP_1HL_FS_9,
                    qual_MLP_1HL_FS_10)

# Replace all Inf values with NA
MLP_1HL_FS[MLP_1HL_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_1HL_FS <- mean(MLP_1HL_FS$MAPE, na.rm = TRUE)*100
mape_sd_MLP_1HL_FS <- sd(MLP_1HL_FS$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_1HL_FS <- mean(MLP_1HL_FS$RMSE, na.rm = TRUE)
rmse_sd_MLP_1HL_FS <- sd(MLP_1HL_FS$RMSE, na.rm = TRUE)

mae_mean_MLP_1HL_FS <- mean(MLP_1HL_FS$MAE, na.rm = TRUE)
mae_sd_MLP_1HL_FS <- sd(MLP_1HL_FS$MAE, na.rm = TRUE)

# MLP 1 HL No FS ---------------------------------------------------------------

MLP_1HL_No_FS <- rbind(qual_MLP_1HL_No_FS_1, qual_MLP_1HL_No_FS_2, qual_MLP_1HL_No_FS_3, qual_MLP_1HL_No_FS_4, qual_MLP_1HL_No_FS_5,
                       qual_MLP_1HL_No_FS_6, qual_MLP_1HL_No_FS_7, qual_MLP_1HL_No_FS_8, qual_MLP_1HL_No_FS_9,
                       qual_MLP_1HL_No_FS_10)

# Replace all Inf values with NA
MLP_1HL_No_FS[MLP_1HL_No_FS == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_1HL_No_FS <- mean(MLP_1HL_No_FS$MAPE, na.rm = TRUE)*100
mape_sd_MLP_1HL_No_FS <- sd(MLP_1HL_No_FS$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_1HL_No_FS <- mean(MLP_1HL_No_FS$RMSE, na.rm = TRUE)
rmse_sd_MLP_1HL_No_FS <- sd(MLP_1HL_No_FS$RMSE, na.rm = TRUE)

mae_mean_MLP_1HL_No_FS <- mean(MLP_1HL_No_FS$MAE, na.rm = TRUE)
mae_sd_MLP_1HL_No_FS <- sd(MLP_1HL_No_FS$MAE, na.rm = TRUE)

# Final dataframe --------------------------------------------------------------

# Create a data frame to store the measures
measures <- data.frame(
  Measure = c("MAPE Mean FOCEI PARENT", "MAPE SD FOCEI PARENT", "RMSE Mean FOCEI PARENT", "RMSE SD FOCEI PARENT", "MAE Mean FOCEI PARENT", "MAE SD FOCEI PARENT",
              "MAPE Mean FOCEI METABOLITE", "MAPE SD FOCEI METABOLITE", "RMSE Mean FOCEI METABOLITE", "RMSE SD FOCEI METABOLITE", "MAE Mean FOCEI METABOLITE", "MAE SD FOCEI METABOLITE",
              "MAPE Mean SAEM PARENT", "MAPE SD SAEM PARENT", "RMSE Mean SAEM PARENT", "RMSE SD SAEM PARENT", "MAE Mean SAEM PARENT", "MAE SD SAEM PARENT",
              "MAPE Mean SAEM METABOLITE", "MAPE SD SAEM METABOLITE", "RMSE Mean SAEM METABOLITE", "RMSE SD SAEM METABOLITE", "MAE Mean SAEM METABOLITE", "MAE SD SAEM METABOLITE",
              "MAPE Mean RF FS", "MAPE SD RF FS", "RMSE Mean RF FS", "RMSE SD RF FS",  "MAE Mean RF FS", "MAE SD RF FS",
              "MAPE Mean RF No FS", "MAPE SD RF No FS", "RMSE Mean RF No FS", "RMSE SD RF No FS",  "MAE Mean RF No FS", "MAE SD RF No FS", 
              "MAPE Mean GB FS", "MAPE SD GB FS", "RMSE Mean GB FS", "RMSE SD GB FS", "MAE Mean GB FS", "MAE SD GB FS", 
              "MAPE Mean GB No FS", "MAPE SD GB No FS", "RMSE Mean GB No FS", "RMSE SD GB No FS", "MAE Mean GB No FS", "MAE SD GB No FS", 
              "MAPE Mean SVM FS", "MAPE SD SVM FS", "RMSE Mean SVM FS", "RMSE SD SVM FS", "MAE Mean SVM FS", "MAE SD SVM FS",
              "MAPE Mean SVM No FS", "MAPE SD SVM No FS", "RMSE Mean SVM No FS", "RMSE SD SVM No FS", "MAE Mean SVM No FS", "MAE SD SVM No FS",
              "MAPE Mean XGB FS", "MAPE SD XGB FS", "RMSE Mean XGB FS", "RMSE SD XGB FS", "MAE Mean XGB FS", "MAE SD XGB FS", 
              "MAPE Mean XGB No FS", "MAPE SD XGB No FS", "RMSE Mean XGB No FS", "RMSE SD XGB No FS", "MAE Mean XGB No FS", "MAE SD XGB No FS",
              "MAPE Mean GB FS", "MAPE SD LGB FS", "RMSE Mean LGB FS", "RMSE SD LGB FS", "MAE Mean LGB FS", "MAE SD LGB FS", 
              "MAPE Mean LGB No FS", "MAPE SD LGB No FS", "RMSE Mean LGB No FS", "RMSE SD LGB No FS", "MAE Mean LGB No FS", "MAE SD LGB No FS", 
              "MAPE Mean MLP 2HL FS", "MAPE SD MLP 2HL FS", "RMSE Mean MLP 2HL FS", "RMSE SD MLP 2HL FS","MAE Mean MLP 2HL FS", "MAE SD MLP 2HL FS",
              "MAPE Mean MLP 2HL No FS", "MAPE SD MLP 2HL No FS", "RMSE Mean MLP 2HL No FS", "RMSE SD MLP 2HL No FS", "MAE Mean MLP 2HL No FS", "MAE SD MLP 2HL No FS",
              "MAPE Mean MLP 1HL FS", "MAPE SD MLP 1HL FS", "RMSE Mean MLP 1HL FS", "RMSE SD MLP 1HL FS", "MAE Mean MLP 1HL FS", "MAE SD MLP 1HL FS", 
              "MAPE Mean MLP 1HL No FS", "MAPE SD MLP 1HL No FS", "RMSE Mean MLP 1HL No FS", "RMSE SD MLP 1HL No FS","MAE Mean MLP 1HL No FS", "MAE SD MLP 1HL No FS"),
  
  Value = c(mape_mean_focei_parent, mape_sd_focei_parent, rmse_mean_focei_parent, rmse_sd_focei_parent,mae_mean_focei_parent, mae_sd_focei_parent,
            mape_mean_focei_metabolite, mape_sd_focei_metabolite, rmse_mean_focei_metabolite, rmse_sd_focei_metabolite,mae_mean_focei_metabolite, mae_sd_focei_metabolite,
            mape_mean_saem_parent, mape_sd_saem_parent, rmse_mean_saem_parent, rmse_sd_saem_parent, mae_mean_saem_parent, mae_sd_saem_parent,
            mape_mean_saem_metabolite, mape_sd_saem_metabolite, rmse_mean_saem_metabolite, rmse_sd_saem_metabolite, mae_mean_saem_metabolite, mae_sd_saem_metabolite,
            mape_mean_RF_FS, mape_sd_RF_FS, rmse_mean_RF_FS, rmse_sd_RF_FS,  mae_mean_RF_FS, mae_sd_RF_FS,
            mape_mean_RF_No_FS, mape_sd_RF_No_FS, rmse_mean_RF_No_FS, rmse_sd_RF_No_FS, mae_mean_RF_FS, mae_sd_RF_FS,
            mape_mean_GB_FS, mape_sd_GB_FS, rmse_mean_GB_FS, rmse_sd_GB_FS, mae_mean_GB_FS, mae_sd_GB_FS, 
            mape_mean_GB_No_FS, mape_sd_GB_No_FS, rmse_mean_GB_No_FS, rmse_sd_GB_No_FS, mae_mean_GB_No_FS, mae_sd_GB_No_FS, 
            mape_mean_SVM_FS, mape_sd_SVM_FS, rmse_mean_SVM_FS, rmse_sd_SVM_FS, mae_mean_SVM_FS, mae_sd_SVM_FS, 
            mape_mean_SVM_No_FS, mape_sd_SVM_No_FS, rmse_mean_SVM_No_FS, rmse_sd_SVM_No_FS, mae_mean_SVM_No_FS, mae_sd_SVM_No_FS, 
            mape_mean_XGB_FS, mape_sd_XGB_FS, rmse_mean_XGB_FS, rmse_sd_XGB_FS,mae_mean_XGB_FS, mae_sd_XGB_FS, 
            mape_mean_XGB_No_FS, mape_sd_XGB_No_FS, rmse_mean_XGB_No_FS, rmse_sd_XGB_No_FS,mae_mean_XGB_No_FS, mae_sd_XGB_No_FS,
            mape_mean_LGB_FS, mape_sd_LGB_FS, rmse_mean_LGB_FS, rmse_sd_LGB_FS,mae_mean_LGB_FS, mae_sd_LGB_FS,
            mape_mean_LGB_No_FS, mape_sd_LGB_No_FS, rmse_mean_LGB_No_FS, rmse_sd_LGB_No_FS,mae_mean_LGB_No_FS, mae_sd_LGB_No_FS, 
            mape_mean_MLP_2HL_FS, mape_sd_MLP_2HL_FS, rmse_mean_MLP_2HL_FS, rmse_sd_MLP_2HL_FS, mae_mean_MLP_2HL_FS, mae_sd_MLP_2HL_FS,
            mape_mean_MLP_2HL_No_FS, mape_sd_MLP_2HL_No_FS, rmse_mean_MLP_2HL_No_FS, rmse_sd_MLP_2HL_No_FS, mae_mean_MLP_2HL_No_FS, mae_sd_MLP_2HL_No_FS,
            mape_mean_MLP_1HL_FS, mape_sd_MLP_1HL_FS, rmse_mean_MLP_1HL_FS, rmse_sd_MLP_1HL_FS,  mae_mean_MLP_1HL_FS, mae_sd_MLP_1HL_FS, 
            mape_mean_MLP_1HL_No_FS, mape_sd_MLP_1HL_No_FS, rmse_mean_MLP_1HL_No_FS, rmse_sd_MLP_1HL_No_FS,mae_mean_MLP_1HL_No_FS, mae_sd_MLP_1HL_No_FS)
)

# Save the measures to a CSV file
write.csv(measures, "measures_suni_0_augmentation.csv")
