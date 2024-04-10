# Suni Pop-PK evaluation -------------------------------------------------------


#--------------------------------------------------#
# Evaluation of PMX Suni model                     #
#--------------------------------------------------#
# Split 1                                          #
#--------------------------------------------------#


#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots
#xpose4: needed to create pc-vpc and to read nonmem tables
#MLMetrics: needed to calculate MAE (mean absolute error)


rm(list=ls())

library(dplyr) 
library(ggplot2)
library(xpose4)
library(MLmetrics)


setwd("C:/Users/Olga/Suni_Pazo/Split1")

data_suni <- read.csv("Suni_PK_final_new.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, SET_1 == 1) # test data
data_suni <- subset(data_suni, MDV != 1)

# Split number m
# Augmentation percentage n

m <- 1
n <- 0

# Section 1:  DV vs IPRED ------------------------------------------------------
# 1.1:  Parent drug ------------------------------------------------------------

# EXPIPRED is used because DV is on a log-scale in simulation
# Mean EXPIPRED from simulation ------------------------------------------------

#FOCEI -------------------------------------------------------------------------
# filter out the observations for the parent drug
data_suni <- subset(data_suni, CMT == 2) # parent

# predictions from simulation
data<-read.nm.tables("Suni_focei_sim_split_1.tab")

# Filter rows where DV is not 0 and CMT is 2 (parent)
data <- data %>%
  filter(LNDV != 0, CMT == 2)

data$EXPIPRED <- exp(data$IPRED)

# Calculate the mean of 'EXPIPRED' 
num_positions <- nrow(data_suni)
num_individuals <- nrow(data_suni)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$EXPIPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_a <- data.frame(Position = seq(1, num_positions), Mean_EXPIPRED = mean_values, REAL_values = data_suni$DV)

# Save the result data frame to a CSV file
write.csv(result_df_a, file = sprintf("mean_ipred_values_sim_focei_suni_parent_%d_%d.csv", m,n), row.names = FALSE)

#DV vs EXPIPRED for simulated results
# predictions from simulation
prediction<-result_df_a$Mean_EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: FOCE-I (test data) Parent") +
  annotate("text", x = 15, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_suni_parent_%d_%d.jpg", m,n), width=6, height=6)


#SAEM -------------------------------------------------------------------------
data<-read.nm.tables("Suni_saem_sim_split_1.tab")

# Filter rows where DV is not 0
data <- data %>%
  filter(DV != 0, CMT == 2)

data$EXPIPRED <- exp(data$IPRED)

# Calculate the mean of 'EXPIPRED' 
num_positions <- nrow(data_suni)
num_individuals <- nrow(data_suni)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$EXPIPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_b <- data.frame(Position = seq(1, num_positions), Mean_EXPIPRED = mean_values, REAL_values = data_suni$DV)

# Save the result data frame to a CSV file
write.csv(result_df_b, file = sprintf("mean_ipred_values_sim_saem_suni_parent_%d_%d.csv", m,n), row.names = FALSE)

#DV vs EXPIPRED for simulated results
# predictions from simulation
prediction<-result_df_b$Mean_EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: SAEM (test data) Parent") +
  annotate("text", x = 15, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_suni_parent_%d_%d.jpg", m,n), width=6, height=6)

# 1.2:  Metabolite -------------------------------------------------------------

#FOCEI -------------------------------------------------------------------------
# filter out the observations for the metabolite
data_suni <- read.csv("Suni_PK_final_new.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, SET_1 == 1) # test data
data_suni <- subset(data_suni, MDV != 1)
data_suni <- subset(data_suni, CMT == 3) # metabolite

# predictions from simulation
data<-read.nm.tables("Suni_focei_sim_split_1.tab")

# Filter rows where DV is not 0 and CMT is 3 (metabolite)
data <- data %>%
  filter(LNDV != 0, CMT == 3)

data$EXPIPRED <- exp(data$IPRED)

# Calculate the mean of 'EXPIPRED' 
num_positions <- nrow(data_suni)
num_individuals <- nrow(data_suni)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$EXPIPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_c <- data.frame(Position = seq(1, num_positions), Mean_EXPIPRED = mean_values, REAL_values = data_suni$DV)

# Save the result data frame to a CSV file
write.csv(result_df_c, file = sprintf("mean_ipred_values_sim_focei_suni_metabolite_%d_%d.csv", m,n), row.names = FALSE)

#DV vs EXPIPRED for simulated results
# predictions from simulation
prediction<-result_df_c$Mean_EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: FOCE-I (test data) SU12662") +
  annotate("text", x = 15, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)


#SAEM -------------------------------------------------------------------------
data<-read.nm.tables("Suni_saem_sim_split_1.tab")

# Filter rows where DV is not 0
data <- data %>%
  filter(DV != 0, CMT == 3)

data$EXPIPRED <- exp(data$IPRED)

# Calculate the mean of 'EXPIPRED' 
num_positions <- nrow(data_suni)
num_individuals <- nrow(data_suni)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$EXPIPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_d <- data.frame(Position = seq(1, num_positions), Mean_EXPIPRED = mean_values, REAL_values = data_suni$DV)

# Save the result data frame to a CSV file
write.csv(result_df_d, file = sprintf("mean_ipred_values_sim_saem_suni_metabolite_%d_%d.csv", m,n), row.names = FALSE)

#DV vs EXPIPRED for simulated results
# predictions from simulation
prediction<-result_df_d$Mean_EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: SAEM (test data) SU12662") +
  annotate("text", x = 15, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)

# Section 2: MAPE, RMSE, MAE for NONMEM -----------------------------------------

# 2.1:  Parent drug ------------------------------------------------------------

data_suni <- read.csv("Suni_PK_final_new.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, MDV != 1 & SET_1 == 1) # test
data_suni <- subset(data_suni, CMT == 2) # parent
actual<- data_suni$DV
actual <- as.double(actual)

# FOCE-I -----------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-result_df_a$Mean_EXPIPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)


# Create a data frame to store the calculated values
quality <- data.frame(
  RMSE = rmse_dv,
  MAPE = mape_dv,
  MAE = mae_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_focei_parent_%d_%d.csv", m,n), row.names = FALSE)


# SAEM flag --------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-result_df_b$Mean_EXPIPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)


# Create a data frame to store the calculated values
quality <- data.frame(
  RMSE = rmse_dv,
  MAPE = mape_dv,
  MAE = mae_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_saem_parent_%d_%d.csv", m,n), row.names = FALSE)

# 2.2:  Metabolite -------------------------------------------------------------

data_suni <- read.csv("Suni_PK_final_new.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, MDV != 1 & SET_1 == 1) # test
data_suni <- subset(data_suni, CMT == 3) # metabolite
actual<- data_suni$DV
actual <- as.double(actual)

# FOCE-I -----------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-result_df_c$Mean_EXPIPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)


# Create a data frame to store the calculated values
quality <- data.frame(
  RMSE = rmse_dv,
  MAPE = mape_dv,
  MAE = mae_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_focei_metabolite_%d_%d.csv", m,n), row.names = FALSE)


# SAEM flag --------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-result_df_d$Mean_EXPIPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)


# Create a data frame to store the calculated values
quality <- data.frame(
  RMSE = rmse_dv,
  MAPE = mape_dv,
  MAE = mae_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_saem_metabolite_%d_%d.csv", m,n), row.names = FALSE)

# Section 3 :  CWRES vs EXPIPRED -----------------------------------------------

# 3.1:  Parent drug ------------------------------------------------------------

# FOCE-I  ----------------------------------------------------------------------
data_cwres<- read.nm.tables("Suni_focei_split_1.tab")
data_cwres <- subset(data_cwres, CMT == 2) # parent
data_cwres$IPRED <- as.numeric(data_cwres$IPRED)
data_cwres$EXPIPRED <- as.numeric(data_cwres$EXPIPRED)
data_cwres$CWRES <- as.numeric(data_cwres$CWRES)
data_cwres$TIME <- as.numeric(data_cwres$TIME)

data_cwres <- data_cwres %>%
  filter(LNDV != 0 & PRED != 0)

#CWRES vs EXPIPRED
cwres_pred<- ggplot(data_cwres,aes(x=EXPIPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c((min(data_cwres$EXPIPRED)-1),(max(data_cwres$EXPIPRED)+1))) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations \n Method: FOCE-I (train data) Parent [1]") +
  theme(plot.title = element_text(size=10.5, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_focei_suni_parent_%d_%d.jpg", m,n), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time \n Method: FOCE-I (train data) Parent [1]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_focei_suni_parent_%d_%d.jpg", m,n), width=6, height=6)


# SAEM -------------------------------------------------------------------------
data_cwres<- read.nm.tables("Suni_saem_split_1.tab")
data_cwres <- subset(data_cwres, CMT == 2) # parent
data_cwres$IPRED <- as.numeric(data_cwres$IPRED)
data_cwres$EXPIPRED <- as.numeric(data_cwres$EXPIPRED)
data_cwres$CWRES <- as.numeric(data_cwres$CWRES)
data_cwres$TIME <- as.numeric(data_cwres$TIME)

data_cwres <- data_cwres %>%
  filter(LNDV != 0 & PRED != 0)

#CWRES vs EXPIPRED
cwres_pred<- ggplot(data_cwres,aes(x=EXPIPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c((min(data_cwres$EXPIPRED)-1),(max(data_cwres$EXPIPRED)+1))) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations \n Method: SAEM (train data) Parent [1]") +
  theme(plot.title = element_text(size=10.5, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_saem_suni_parent_%d_%d.jpg", m,n), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time \n Method: SAEM (train data) Parent [1]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_saem_suni_parent_%d_%d.jpg", m,n), width=6, height=6)

# 3.2:  Metabolite -------------------------------------------------------------

# FOCE-I  ----------------------------------------------------------------------
data_cwres<- read.nm.tables("Suni_focei_split_1.tab")
data_cwres <- subset(data_cwres, CMT == 3) # metabolite
data_cwres$IPRED <- as.numeric(data_cwres$IPRED)
data_cwres$EXPIPRED <- as.numeric(data_cwres$EXPIPRED)
data_cwres$CWRES <- as.numeric(data_cwres$CWRES)
data_cwres$TIME <- as.numeric(data_cwres$TIME)

data_cwres <- data_cwres %>%
  filter(LNDV != 0 & PRED != 0)

#CWRES vs EXPIPRED
cwres_pred<- ggplot(data_cwres,aes(x=EXPIPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c((min(data_cwres$EXPIPRED)-1),(max(data_cwres$EXPIPRED)+1))) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations \n Method: FOCE-I (train data) SU12662 [1]") +
  theme(plot.title = element_text(size=10.5, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_focei_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time [1] \n Method: FOCE-I (train data) SU12662 [1]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_focei_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)


# SAEM -------------------------------------------------------------------------
data_cwres<- read.nm.tables("Suni_saem_split_1.tab")
data_cwres <- subset(data_cwres, CMT == 3) # metabolite
data_cwres$IPRED <- as.numeric(data_cwres$IPRED)
data_cwres$EXPIPRED <- as.numeric(data_cwres$EXPIPRED)
data_cwres$CWRES <- as.numeric(data_cwres$CWRES)
data_cwres$TIME <- as.numeric(data_cwres$TIME)

data_cwres <- data_cwres %>%
  filter(LNDV != 0 & PRED != 0)

#CWRES vs EXPIPRED
cwres_pred<- ggplot(data_cwres,aes(x=EXPIPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c((min(data_cwres$EXPIPRED)-1),(max(data_cwres$EXPIPRED)+1))) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations \n Method: SAEM (train data) SU12662 [1]") +
  theme(plot.title = element_text(size=10.5, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_saem_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time [1] \n Method: SAEM (train data) SU12662 [1]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_saem_suni_metabolite_%d_%d.jpg", m,n), width=6, height=6)

# Section 4 :  ETA distribution  -----------------------------------------------

#FOCE-I ------------------------------------------------------------------------

data_etas <-read.nm.tables("etas_Suni_focei_split_1.tab")

# Select just one eta per patient (baseline)
data_etas <- data_etas %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Because some parameters were scaled with ASV (WT/60) and ASCL ((WT/70)**0.75), they need to be scaled back
# also: ETAS are on ln scale and ASV and ASCL on normal scale, hence we need the exp of the scaling factors

# Create the ETA distribution plot using ggplot2

# ETA distribution of IIV V2 (central vol. sunitinib)---------------------------
data_etas$ETA1sc<- data_etas$ETA1/exp(data_etas$ASV)

eta_plot <-ggplot(data_etas, aes(x=ETA1sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV V2) [1] \n Method: FOCE-I (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_v2_focei_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV V3 (central vol. metabolite SU12662)------------------
data_etas$ETA2sc<- data_etas$ETA2/exp(data_etas$ASV)
eta_plot <-ggplot(data_etas, aes(x=ETA2sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV V3) [1] \n Method: FOCE-I (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_v3_focei_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV CLP (Parent)------------------------------------------
data_etas$ETA3sc<- data_etas$ETA3/exp(data_etas$ASCL)

eta_plot <-ggplot(data_etas, aes(x=ETA3sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV CLP) [1] \n Method: FOCE-I (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_clp_focei_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV FM (Fraction metabolised to SU12662)------------------

eta_plot <-ggplot(data_etas, aes(x=ETA4)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV FM) [1] \n Method: FOCE-I (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_fm_focei_%d_%d.jpg",m,n), width= 6, height= 6)


# SAEM -------------------------------------------------------------------------

data_etas <-read.nm.tables("etas_Suni_saem_split_1.tab")

# Select just one eta per patient (baseline)
data_etas <- data_etas %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Create the ETA distribution plot using ggplot2
# ETA distribution of IIV V2 (central vol. sunitinib)---------------------------
data_etas$ETA1sc<- data_etas$ETA1/exp(data_etas$ASV)

eta_plot <-ggplot(data_etas, aes(x=ETA1sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV V2) [1] \n Method: SAEM (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_v2_saem_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV V3 (central vol. metabolite SU12662)------------------
data_etas$ETA2sc<- data_etas$ETA2/exp(data_etas$ASV)
eta_plot <-ggplot(data_etas, aes(x=ETA2sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV V3) [1] \n Method: SAEM (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_v3_saem_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV CLP (Parent)------------------------------------------
data_etas$ETA3sc<- data_etas$ETA3/exp(data_etas$ASCL)

eta_plot <-ggplot(data_etas, aes(x=ETA3sc)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.02, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV CLP) [1] \n Method: SAEM (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_clp_saem_%d_%d.jpg",m,n), width= 6, height= 6)

# ETA distribution of IIV FM (Fraction metabolised to SU12662)------------------

eta_plot <-ggplot(data_etas, aes(x=ETA4)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV FM) [1] \n Method: SAEM (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_fm_saem_%d_%d.jpg",m,n), width= 6, height= 6)

