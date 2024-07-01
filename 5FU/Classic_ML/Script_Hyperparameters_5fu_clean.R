###################################################
# Hyperparameter Tables                           #
#                                                 #
# 5fu                                             #
###################################################

# Library 

library(dplyr) 

# Directory 

setwd("C:/Users/teply/Documents/5FU_NONMEM_final/Quality")

# Params Test Gradient Boost ----------------------------------------------

table1 <- read.csv("5fu_params_test_Gradient_Boosting_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_Gradient_Boosting_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_Gradient_Boosting_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_Gradient_Boosting_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_Gradient_Boosting_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_Gradient_Boosting_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_Gradient_Boosting_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_Gradient_Boosting_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_Gradient_Boosting_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_Gradient_Boosting_split_10_augmentation_0.csv")

Gradient_Boost_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(Gradient_Boost_Hyperparams)

# Transpose

tGradient_Boost_Hyperparams <- t(Gradient_Boost_Hyperparams)

# save

write.csv(tGradient_Boost_Hyperparams, file = "Gradient_Boost_Hyperparameter.csv")

# Params Test LGB  -------------------------------

table1 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_Light_Gradient_Boosting_split_10_augmentation_0.csv")


LGB_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(LGB_Hyperparams)

# Transponieren

tLGB_Hyperparams <- t(LGB_Hyperparams)

# save

write.csv(tLGB_Hyperparams, file = "Light_Gradient_Boost_Hyperparameter.csv")

# Params MLP one hidden layer -------------------

table1 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_MLP_one_hidden_layer_split_10_augmentation_0.csv")


MLP_1_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(MLP_1_Hyperparams)

# Transpose

tMLP_1_Hyperparams <- t(MLP_1_Hyperparams)

# save

write.csv(tMLP_1_Hyperparams, file = "MLP_one_Hidden_layer.csv")

# Params MLP two hidden layers -------------

table1 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_MLP_two_hidden_layers_split_10_augmentation_0.csv")


MLP_2_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(MLP_2_Hyperparams)

# Transpose

tMLP_2_Hyperparams <- t(MLP_2_Hyperparams)

# save

write.csv(tMLP_2_Hyperparams, file = "MLP_two_Hidden_layers.csv")


# Params Random Forest ------------------------------

table1 <- read.csv("5fu_params_test_Random_Forest_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_Random_Forest_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_Random_Forest_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_Random_Forest_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_Random_Forest_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_Random_Forest_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_Random_Forest_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_Random_Forest_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_Random_Forest_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_Random_Forest_split_10_augmentation_0.csv")


RF_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(RF_Hyperparams)

# Transpose

tRF_Hyperparams <- t(RF_Hyperparams)

# save

write.csv(tRF_Hyperparams, file = "RF.csv")


# Params Support Vector Machine ------------------------------



table1 <- read.csv("5fu_params_test_Support_Vector_Machine_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_Support_Vector_Machine_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_Support_Vector_Machine_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_Support_Vector_Machine_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_Support_Vector_Machine_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_Support_Vector_Machine_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_Support_Vector_Machine_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_Support_Vector_Machine_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_Support_Vector_Machine_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_Support_Vector_Machine_split_10_augmentation_0.csv")


SVM_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(SVM_Hyperparams)

# Transpose

tSVM_Hyperparams <- t(SVM_Hyperparams)

# save

write.csv(tSVM_Hyperparams, file = "SVM.csv")


# Params Xtreme Gradient Boost ------------------------

table1 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_1_augmentation_0.csv")
table2 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_2_augmentation_0.csv")
table3 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_3_augmentation_0.csv")
table4 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_4_augmentation_0.csv")
table5 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_5_augmentation_0.csv")
table6 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_6_augmentation_0.csv")
table7 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_7_augmentation_0.csv")
table8 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_8_augmentation_0.csv")
table9 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_9_augmentation_0.csv")
table10 <- read.csv("5fu_params_test_Xtreme_Gradient_Boosting_split_10_augmentation_0.csv")


XGB_Hyperparams <- rbind(table1, table2, table3, table4, table5, table6, table7, table8, table9, table10)
print(XGB_Hyperparams)

# Transpose

tXGB_Hyperparams <- t(XGB_Hyperparams)

# save

write.csv(tXGB_Hyperparams, file = "XGB.csv")





