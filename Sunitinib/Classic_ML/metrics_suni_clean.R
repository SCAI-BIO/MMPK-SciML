# MAPE, RMSE, MAE summary Suni  ------------------------------------

#--------------------------------------------------#
# metrics summary                                  #                                                 
#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling

# replace augmentation_0 with augmentation_100 in case of data augmentation and delete NONMEM part


rm(list=ls())
library(dplyr) 

setwd("C:/Users/teply/Documents/Suni_NONMEM_final/Quality")

# NM parent
qual_focei_parent_1<- read.csv(("quality_measures_test_focei_parent_1.csv"))
qual_saem_parent_1<- read.csv(("quality_measures_test_saem_parent_1.csv"))
qual_focei_parent_2<- read.csv(("quality_measures_test_focei_parent_2.csv"))
qual_saem_parent_2<- read.csv(("quality_measures_test_saem_parent_2.csv"))
qual_focei_parent_3<- read.csv(("quality_measures_test_focei_parent_3.csv"))
qual_saem_parent_3<- read.csv(("quality_measures_test_saem_parent_3.csv"))
qual_focei_parent_4<- read.csv(("quality_measures_test_focei_parent_4.csv"))
qual_saem_parent_4<- read.csv(("quality_measures_test_saem_parent_4.csv"))
qual_focei_parent_5<- read.csv(("quality_measures_test_focei_parent_5.csv"))
qual_saem_parent_5<- read.csv(("quality_measures_test_saem_parent_5.csv"))
qual_focei_parent_6<- read.csv(("quality_measures_test_focei_parent_6.csv"))
qual_saem_parent_6<- read.csv(("quality_measures_test_saem_parent_6.csv"))
qual_focei_parent_7<- read.csv(("quality_measures_test_focei_parent_7.csv"))
qual_saem_parent_7<- read.csv(("quality_measures_test_saem_parent_7.csv"))
qual_focei_parent_8<- read.csv(("quality_measures_test_focei_parent_8.csv"))
qual_saem_parent_8<- read.csv(("quality_measures_test_saem_parent_8.csv"))
qual_focei_parent_9<- read.csv(("quality_measures_test_focei_parent_9.csv"))
qual_saem_parent_9<- read.csv(("quality_measures_test_saem_parent_9.csv"))
qual_focei_parent_10<- read.csv(("quality_measures_test_focei_parent_10.csv"))
qual_saem_parent_10<- read.csv(("quality_measures_test_saem_parent_10.csv"))

# NM metabolite
qual_focei_metabolite_1<- read.csv(("quality_measures_test_focei_metabolite_1.csv"))
qual_saem_metabolite_1<- read.csv(("quality_measures_test_saem_metabolite_1.csv"))
qual_focei_metabolite_2<- read.csv(("quality_measures_test_focei_metabolite_2.csv"))
qual_saem_metabolite_2<- read.csv(("quality_measures_test_saem_metabolite_2.csv"))
qual_focei_metabolite_3<- read.csv(("quality_measures_test_focei_metabolite_3.csv"))
qual_saem_metabolite_3<- read.csv(("quality_measures_test_saem_metabolite_3.csv"))
qual_focei_metabolite_4<- read.csv(("quality_measures_test_focei_metabolite_4.csv"))
qual_saem_metabolite_4<- read.csv(("quality_measures_test_saem_metabolite_4.csv"))
qual_focei_metabolite_5<- read.csv(("quality_measures_test_focei_metabolite_5.csv"))
qual_saem_metabolite_5<- read.csv(("quality_measures_test_saem_metabolite_5.csv"))
qual_focei_metabolite_6<- read.csv(("quality_measures_test_focei_metabolite_6.csv"))
qual_saem_metabolite_6<- read.csv(("quality_measures_test_saem_metabolite_6.csv"))
qual_focei_metabolite_7<- read.csv(("quality_measures_test_focei_metabolite_7.csv"))
qual_saem_metabolite_7<- read.csv(("quality_measures_test_saem_metabolite_7.csv"))
qual_focei_metabolite_8<- read.csv(("quality_measures_test_focei_metabolite_8.csv"))
qual_saem_metabolite_8<- read.csv(("quality_measures_test_saem_metabolite_8.csv"))
qual_focei_metabolite_9<- read.csv(("quality_measures_test_focei_metabolite_9.csv"))
qual_saem_metabolite_9<- read.csv(("quality_measures_test_saem_metabolite_9.csv"))
qual_focei_metabolite_10<- read.csv(("quality_measures_test_focei_metabolite_10.csv"))
qual_saem_metabolite_10<- read.csv(("quality_measures_test_saem_metabolite_10.csv"))


# ML
qual_RF_1<-read.csv(("suni_params_test_Random_Forest_split_1_augmentation_0.csv"))
qual_GB_1<-read.csv(("suni_params_test_Gradient_Boosting_split_1_augmentation_0.csv"))
qual_SVM_1<-read.csv(("suni_params_test_Support_Vector_Machine_split_1_augmentation_0.csv"))
qual_XGB_1<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_1_augmentation_0.csv"))
qual_LGB_1<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_1_augmentation_0.csv"))
qual_MLP_2HL_1<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_1_augmentation_0.csv"))
qual_MLP_1HL_1<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_1_augmentation_0.csv"))

qual_RF_2<-read.csv(("suni_params_test_Random_Forest_split_2_augmentation_0.csv"))
qual_GB_2<-read.csv(("suni_params_test_Gradient_Boosting_split_2_augmentation_0.csv"))
qual_SVM_2<-read.csv(("suni_params_test_Support_Vector_Machine_split_2_augmentation_0.csv"))
qual_XGB_2<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_2_augmentation_0.csv"))
qual_LGB_2<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_2_augmentation_0.csv"))
qual_MLP_2HL_2<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_2_augmentation_0.csv"))
qual_MLP_1HL_2<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_2_augmentation_0.csv"))

qual_RF_3<-read.csv(("suni_params_test_Random_Forest_split_3_augmentation_0.csv"))
qual_GB_3<-read.csv(("suni_params_test_Gradient_Boosting_split_3_augmentation_0.csv"))
qual_SVM_3<-read.csv(("suni_params_test_Support_Vector_Machine_split_3_augmentation_0.csv"))
qual_XGB_3<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_3_augmentation_0.csv"))
qual_LGB_3<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_3_augmentation_0.csv"))
qual_MLP_2HL_3<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_3_augmentation_0.csv"))
qual_MLP_1HL_3<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_3_augmentation_0.csv"))

qual_RF_4<-read.csv(("suni_params_test_Random_Forest_split_4_augmentation_0.csv"))
qual_GB_4<-read.csv(("suni_params_test_Gradient_Boosting_split_4_augmentation_0.csv"))
qual_SVM_4<-read.csv(("suni_params_test_Support_Vector_Machine_split_4_augmentation_0.csv"))
qual_XGB_4<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_4_augmentation_0.csv"))
qual_LGB_4<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_4_augmentation_0.csv"))
qual_MLP_2HL_4<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_4_augmentation_0.csv"))
qual_MLP_1HL_4<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_4_augmentation_0.csv"))

qual_RF_5<-read.csv(("suni_params_test_Random_Forest_split_5_augmentation_0.csv"))
qual_GB_5<-read.csv(("suni_params_test_Gradient_Boosting_split_5_augmentation_0.csv"))
qual_SVM_5<-read.csv(("suni_params_test_Support_Vector_Machine_split_5_augmentation_0.csv"))
qual_XGB_5<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_5_augmentation_0.csv"))
qual_LGB_5<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_5_augmentation_0.csv"))
qual_MLP_2HL_5<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_5_augmentation_0.csv"))
qual_MLP_1HL_5<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_5_augmentation_0.csv"))

qual_RF_6<-read.csv(("suni_params_test_Random_Forest_split_6_augmentation_0.csv"))
qual_GB_6<-read.csv(("suni_params_test_Gradient_Boosting_split_6_augmentation_0.csv"))
qual_SVM_6<-read.csv(("suni_params_test_Support_Vector_Machine_split_6_augmentation_0.csv"))
qual_XGB_6<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_6_augmentation_0.csv"))
qual_LGB_6<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_6_augmentation_0.csv"))
qual_MLP_2HL_6<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_6_augmentation_0.csv"))
qual_MLP_1HL_6<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_6_augmentation_0.csv"))

qual_RF_7<-read.csv(("suni_params_test_Random_Forest_split_7_augmentation_0.csv"))
qual_GB_7<-read.csv(("suni_params_test_Gradient_Boosting_split_7_augmentation_0.csv"))
qual_SVM_7<-read.csv(("suni_params_test_Support_Vector_Machine_split_7_augmentation_0.csv"))
qual_XGB_7<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_7_augmentation_0.csv"))
qual_LGB_7<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_7_augmentation_0.csv"))
qual_MLP_2HL_7<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_7_augmentation_0.csv"))
qual_MLP_1HL_7<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_7_augmentation_0.csv"))

qual_RF_8<-read.csv(("suni_params_test_Random_Forest_split_8_augmentation_0.csv"))
qual_GB_8<-read.csv(("suni_params_test_Gradient_Boosting_split_8_augmentation_0.csv"))
qual_SVM_8<-read.csv(("suni_params_test_Support_Vector_Machine_split_8_augmentation_0.csv"))
qual_XGB_8<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_8_augmentation_0.csv"))
qual_LGB_8<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_8_augmentation_0.csv"))
qual_MLP_2HL_8<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_8_augmentation_0.csv"))
qual_MLP_1HL_8<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_8_augmentation_0.csv"))

qual_RF_9<-read.csv(("suni_params_test_Random_Forest_split_9_augmentation_0.csv"))
qual_GB_9<-read.csv(("suni_params_test_Gradient_Boosting_split_9_augmentation_0.csv"))
qual_SVM_9<-read.csv(("suni_params_test_Support_Vector_Machine_split_9_augmentation_0.csv"))
qual_XGB_9<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_9_augmentation_0.csv"))
qual_LGB_9<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_9_augmentation_0.csv"))
qual_MLP_2HL_9<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_9_augmentation_0.csv"))
qual_MLP_1HL_9<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_9_augmentation_0.csv"))

qual_RF_10<-read.csv(("suni_params_test_Random_Forest_split_10_augmentation_0.csv"))
qual_GB_10<-read.csv(("suni_params_test_Gradient_Boosting_split_10_augmentation_0.csv"))
qual_SVM_10<-read.csv(("suni_params_test_Support_Vector_Machine_split_10_augmentation_0.csv"))
qual_XGB_10<-read.csv(("suni_params_test_Xtreme_Gradient_Boosting_split_10_augmentation_0.csv"))
qual_LGB_10<-read.csv(("suni_params_test_Light_Gradient_Boosting_split_10_augmentation_0.csv"))
qual_MLP_2HL_10<-read.csv(("suni_params_test_MLP_two_hidden_layers_split_10_augmentation_0.csv"))
qual_MLP_1HL_10<-read.csv(("suni_params_test_MLP_one_hidden_layer_split_10_augmentation_0.csv"))

# FOCEI parent -----------------------------------------------------------------

focei_parent <- rbind(qual_focei_parent_1, qual_focei_parent_2, qual_focei_parent_3, qual_focei_parent_4, qual_focei_parent_5,
                      qual_focei_parent_6, qual_focei_parent_7, qual_focei_parent_8, qual_focei_parent_9,
                      qual_focei_parent_10)

# Replace all Inf values with NA
focei_parent[focei_parent == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_focei_parent <- mean(focei_parent$MAPE, na.rm = TRUE)
mape_sd_focei_parent <- sd(focei_parent$MAPE, na.rm = TRUE)

rmse_mean_focei_parent <- mean(focei_parent$RMSE, na.rm = TRUE)
rmse_sd_focei_parent <- sd(focei_parent$RMSE, na.rm = TRUE)

mae_mean_focei_parent <- mean(focei_parent$MAE, na.rm = TRUE)
mae_sd_focei_parent <- sd(focei_parent$MAE, na.rm = TRUE)

# FOCEI metabolite -------------------------------------------------------------

focei_metabolite <- rbind(qual_focei_metabolite_1, qual_focei_metabolite_2, qual_focei_metabolite_3, qual_focei_metabolite_4, qual_focei_metabolite_5,
                          qual_focei_metabolite_6, qual_focei_metabolite_7, qual_focei_metabolite_8, qual_focei_metabolite_9,
                          qual_focei_metabolite_10)

# Replace all Inf values with NA
focei_metabolite[focei_metabolite == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_focei_metabolite <- mean(focei_metabolite$MAPE, na.rm = TRUE)
mape_sd_focei_metabolite <- sd(focei_metabolite$MAPE, na.rm = TRUE)

rmse_mean_focei_metabolite <- mean(focei_metabolite$RMSE, na.rm = TRUE)
rmse_sd_focei_metabolite <- sd(focei_metabolite$RMSE, na.rm = TRUE)

mae_mean_focei_metabolite <- mean(focei_metabolite$MAE, na.rm = TRUE)
mae_sd_focei_metabolite <- sd(focei_metabolite$MAE, na.rm = TRUE)

# SAEM parent ------------------------------------------------------------------

saem_parent <- rbind(qual_saem_parent_1, qual_saem_parent_2, qual_saem_parent_3, qual_saem_parent_4, qual_saem_parent_5,
                     qual_saem_parent_6, qual_saem_parent_7, qual_saem_parent_8, qual_saem_parent_9,
                     qual_saem_parent_10)

# Replace all Inf values with NA
saem_parent[saem_parent == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_saem_parent <- mean(saem_parent$MAPE, na.rm = TRUE)
mape_sd_saem_parent <- sd(saem_parent$MAPE, na.rm = TRUE)

rmse_mean_saem_parent <- mean(saem_parent$RMSE, na.rm = TRUE)
rmse_sd_saem_parent <- sd(saem_parent$RMSE, na.rm = TRUE)

mae_mean_saem_parent <- mean(saem_parent$MAE, na.rm = TRUE)
mae_sd_saem_parent <- sd(saem_parent$MAE, na.rm = TRUE)

# SAEM metabolite --------------------------------------------------------------

saem_metabolite <- rbind(qual_saem_metabolite_1, qual_saem_metabolite_2, qual_saem_metabolite_3, qual_saem_metabolite_4, qual_saem_metabolite_5,
                         qual_saem_metabolite_6, qual_saem_metabolite_7, qual_saem_metabolite_8, qual_saem_metabolite_9,
                         qual_saem_metabolite_10)

# Replace all Inf values with NA
saem_metabolite[saem_metabolite == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE, and MAE while excluding NaN and Inf values
mape_mean_saem_metabolite <- mean(saem_metabolite$MAPE, na.rm = TRUE)
mape_sd_saem_metabolite <- sd(saem_metabolite$MAPE, na.rm = TRUE)

rmse_mean_saem_metabolite <- mean(saem_metabolite$RMSE, na.rm = TRUE)
rmse_sd_saem_metabolite <- sd(saem_metabolite$RMSE, na.rm = TRUE)

mae_mean_saem_metabolite <- mean(saem_metabolite$MAE, na.rm = TRUE)
mae_sd_saem_metabolite <- sd(saem_metabolite$MAE, na.rm = TRUE)


# RF  ---------------------------------------------------------------------

RF <- rbind(qual_RF_1, qual_RF_2, qual_RF_3, qual_RF_4, qual_RF_5,
              qual_RF_6, qual_RF_7, qual_RF_8, qual_RF_9,
              qual_RF_10)

# Replace all Inf values with NA
RF[RF == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_RF <- mean(RF$MAPE, na.rm = TRUE)*100
mape_sd_RF <- sd(RF$MAPE, na.rm = TRUE)*100

rmse_mean_RF <- mean(RF$RMSE, na.rm = TRUE)
rmse_sd_RF <- sd(RF$RMSE, na.rm = TRUE)

mae_mean_RF <- mean(RF$MAE, na.rm = TRUE)
mae_sd_RF <- sd(RF$MAE, na.rm = TRUE)


# GB  ---------------------------------------------------------------------

GB <- rbind(qual_GB_1, qual_GB_2, qual_GB_3, qual_GB_4, qual_GB_5,
              qual_GB_6, qual_GB_7, qual_GB_8, qual_GB_9,
              qual_GB_10)

# Replace all Inf values with NA
GB[GB == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_GB <- mean(GB$MAPE, na.rm = TRUE)*100
mape_sd_GB <- sd(GB$MAPE, na.rm = TRUE)*100

rmse_mean_GB <- mean(GB$RMSE, na.rm = TRUE)
rmse_sd_GB <- sd(GB$RMSE, na.rm = TRUE)

mae_mean_GB <- mean(GB$MAE, na.rm = TRUE)
mae_sd_GB <- sd(GB$MAE, na.rm = TRUE)



# SVM  ---------------------------------------------------------------------

SVM <- rbind(qual_SVM_1, qual_SVM_2, qual_SVM_3, qual_SVM_4, qual_SVM_5,
              qual_SVM_6, qual_SVM_7, qual_SVM_8, qual_SVM_9,
              qual_SVM_10)

# Replace all Inf values with NA
SVM[SVM == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_SVM <- mean(SVM$MAPE, na.rm = TRUE)*100
mape_sd_SVM <- sd(SVM$MAPE, na.rm = TRUE)*100

rmse_mean_SVM <- mean(SVM$RMSE, na.rm = TRUE)
rmse_sd_SVM <- sd(SVM$RMSE, na.rm = TRUE)

mae_mean_SVM <- mean(SVM$MAE, na.rm = TRUE)
mae_sd_SVM <- sd(SVM$MAE, na.rm = TRUE)


# XGB  ---------------------------------------------------------------------

XGB <- rbind(qual_XGB_1, qual_XGB_2, qual_XGB_3, qual_XGB_4, qual_XGB_5,
              qual_XGB_6, qual_XGB_7, qual_XGB_8, qual_XGB_9,
              qual_XGB_10)

# Replace all Inf values with NA
XGB[XGB == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_XGB <- mean(XGB$MAPE, na.rm = TRUE)*100
mape_sd_XGB <- sd(XGB$MAPE, na.rm = TRUE)*100

rmse_mean_XGB <- mean(XGB$RMSE, na.rm = TRUE)
rmse_sd_XGB <- sd(XGB$RMSE, na.rm = TRUE)

mae_mean_XGB <- mean(XGB$MAE, na.rm = TRUE)
mae_sd_XGB <- sd(XGB$MAE, na.rm = TRUE)


# LGB  ---------------------------------------------------------------------

LGB <- rbind(qual_LGB_1, qual_LGB_2, qual_LGB_3, qual_LGB_4, qual_LGB_5,
              qual_LGB_6, qual_LGB_7, qual_LGB_8, qual_LGB_9,
              qual_LGB_10)

# Replace all Inf values with NA
LGB[LGB == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_LGB <- mean(LGB$MAPE, na.rm = TRUE)*100
mape_sd_LGB <- sd(LGB$MAPE, na.rm = TRUE)*100

rmse_mean_LGB <- mean(LGB$RMSE, na.rm = TRUE)
rmse_sd_LGB <- sd(LGB$RMSE, na.rm = TRUE)

mae_mean_LGB <- mean(LGB$MAE, na.rm = TRUE)
mae_sd_LGB <- sd(LGB$MAE, na.rm = TRUE)


# MLP 2 HL  ---------------------------------------------------------------

MLP_2HL <- rbind(qual_MLP_2HL_1, qual_MLP_2HL_2, qual_MLP_2HL_3, qual_MLP_2HL_4, qual_MLP_2HL_5,
                   qual_MLP_2HL_6, qual_MLP_2HL_7, qual_MLP_2HL_8, qual_MLP_2HL_9,
                   qual_MLP_2HL_10)

# Replace all Inf values with NA
MLP_2HL[MLP_2HL == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_2HL <- mean(MLP_2HL$MAPE, na.rm = TRUE)*100
mape_sd_MLP_2HL <- sd(MLP_2HL$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_2HL <- mean(MLP_2HL$RMSE, na.rm = TRUE)
rmse_sd_MLP_2HL <- sd(MLP_2HL$RMSE, na.rm = TRUE)

mae_mean_MLP_2HL <- mean(MLP_2HL$MAE, na.rm = TRUE)
mae_sd_MLP_2HL <- sd(MLP_2HL$MAE, na.rm = TRUE)



# MLP 1 HL  ---------------------------------------------------------------

MLP_1HL <- rbind(qual_MLP_1HL_1, qual_MLP_1HL_2, qual_MLP_1HL_3, qual_MLP_1HL_4, qual_MLP_1HL_5,
                       qual_MLP_1HL_6, qual_MLP_1HL_7, qual_MLP_1HL_8, qual_MLP_1HL_9,
                       qual_MLP_1HL_10)

# Replace all Inf values with NA
MLP_1HL[MLP_1HL == Inf] <- NA

# Calculate means and standard deviations of MAPE, RMSE and MAE
mape_mean_MLP_1HL <- mean(MLP_1HL$MAPE, na.rm = TRUE)*100
mape_sd_MLP_1HL <- sd(MLP_1HL$MAPE, na.rm = TRUE)*100

rmse_mean_MLP_1HL <- mean(MLP_1HL$RMSE, na.rm = TRUE)
rmse_sd_MLP_1HL <- sd(MLP_1HL$RMSE, na.rm = TRUE)

mae_mean_MLP_1HL <- mean(MLP_1HL$MAE, na.rm = TRUE)
mae_sd_MLP_1HL <- sd(MLP_1HL$MAE, na.rm = TRUE)

# Final dataframe --------------------------------------------------------------

# Create a data frame to store the measures
measures <- data.frame(
  Measure = c("MAPE Mean FOCEI parent", "MAPE SD FOCEI parent", "RMSE Mean FOCEI parent", "RMSE SD FOCEI parent", "MAE Mean FOCEI parent", "MAE SD FOCEI parent",
              "MAPE Mean FOCEI metabolite", "MAPE SD FOCEI metabolite", "RMSE Mean FOCEI metabolite", "RMSE SD FOCEI metabolite", "MAE Mean FOCEI metabolite", "MAE SD FOCEI metabolite",
              "MAPE Mean SAEM parent", "MAPE SD SAEM parent", "RMSE Mean SAEM parent", "RMSE SD SAEM parent", "MAE Mean SAEM parent", "MAE SD SAEM parent",
              "MAPE Mean SAEM metabolite", "MAPE SD SAEM metabolite", "RMSE Mean SAEM metabolite", "RMSE SD SAEM metabolite", "MAE Mean SAEM metabolite", "MAE SD SAEM metabolite",
              "MAPE Mean RF ", "MAPE SD RF ", "RMSE Mean RF ", "RMSE SD RF ",  "MAE Mean RF ", "MAE SD RF ", 
              "MAPE Mean GB ", "MAPE SD GB ", "RMSE Mean GB ", "RMSE SD GB ", "MAE Mean GB ", "MAE SD GB ", 
              "MAPE Mean SVM ", "MAPE SD SVM ", "RMSE Mean SVM ", "RMSE SD SVM ", "MAE Mean SVM ", "MAE SD SVM ",
              "MAPE Mean XGB ", "MAPE SD XGB ", "RMSE Mean XGB ", "RMSE SD XGB ", "MAE Mean XGB ", "MAE SD XGB ",
              "MAPE Mean LGB ", "MAPE SD LGB ", "RMSE Mean LGB ", "RMSE SD LGB ", "MAE Mean LGB ", "MAE SD LGB ", 
              "MAPE Mean MLP 2HL ", "MAPE SD MLP 2HL ", "RMSE Mean MLP 2HL ", "RMSE SD MLP 2HL ", "MAE Mean MLP 2HL ", "MAE SD MLP 2HL ",
              "MAPE Mean MLP 1HL ", "MAPE SD MLP 1HL ", "RMSE Mean MLP 1HL ", "RMSE SD MLP 1HL ","MAE Mean MLP 1HL ", "MAE SD MLP 1HL "),
  
  Value = c(mape_mean_focei_parent, mape_sd_focei_parent, rmse_mean_focei_parent, rmse_sd_focei_parent,mae_mean_focei_parent, mae_sd_focei_parent,
            mape_mean_focei_metabolite, mape_sd_focei_metabolite, rmse_mean_focei_metabolite, rmse_sd_focei_metabolite,mae_mean_focei_metabolite, mae_sd_focei_metabolite,
            mape_mean_saem_parent, mape_sd_saem_parent, rmse_mean_saem_parent, rmse_sd_saem_parent, mae_mean_saem_parent, mae_sd_saem_parent,
            mape_mean_saem_metabolite, mape_sd_saem_metabolite, rmse_mean_saem_metabolite, rmse_sd_saem_metabolite, mae_mean_saem_metabolite, mae_sd_saem_metabolite,
            mape_mean_RF, mape_sd_RF, rmse_mean_RF, rmse_sd_RF, mae_mean_RF, mae_sd_RF,
            mape_mean_GB, mape_sd_GB, rmse_mean_GB, rmse_sd_GB, mae_mean_GB, mae_sd_GB, 
            mape_mean_SVM, mape_sd_SVM, rmse_mean_SVM, rmse_sd_SVM, mae_mean_SVM, mae_sd_SVM, 
            mape_mean_XGB, mape_sd_XGB, rmse_mean_XGB, rmse_sd_XGB,mae_mean_XGB, mae_sd_XGB,
            mape_mean_LGB, mape_sd_LGB, rmse_mean_LGB, rmse_sd_LGB,mae_mean_LGB, mae_sd_LGB, 
            mape_mean_MLP_2HL, mape_sd_MLP_2HL, rmse_mean_MLP_2HL, rmse_sd_MLP_2HL, mae_mean_MLP_2HL, mae_sd_MLP_2HL,
            mape_mean_MLP_1HL, mape_sd_MLP_1HL, rmse_mean_MLP_1HL, rmse_sd_MLP_1HL,mae_mean_MLP_1HL, mae_sd_MLP_1HL)
)

# Save the measures to a CSV file
write.csv(measures, "measures_suni_0_augmentation.csv")
