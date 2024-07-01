# Section 1: 5-FU Pop-PK evaluation --------------------------------------------

#--------------------------------------------------#
#--------------------------------------------------#
# Evaluation of PMX 5-FU model                     #
#--------------------------------------------------#
#--------------------------------------------------#


#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots
#xpose4: needed to create pc-vpc and to read nonmem tables
#Metrics: needed to calculate MAE (mean absolute error)


rm(list=ls())

library(Metrics)
library(dplyr) 
library(tidyverse)
library(ggplot2)
library(xpose4)
library(MLmetrics)


setwd("C:/Users/teply/Documents/5FU_NONMEM_final/Split8")

# split8

data_5fu <- read.csv("corrected_NM_Data_final_clean.csv", sep=",") # for DV vs IPRED and other GOF, with 23, clean
data_train <- subset(data_5fu, SET8 == 0)
data_train <- subset(data_train, DV != 0)

data_test <-subset(data_5fu, SET8 == 1)
data_test <- subset(data_test, DV != 0)

# Split number m

m <-8


# Section 1.1 :  DV vs IPRED ---------------------------------------------------

#Mean IPRED from simulation ----------------------------------------------------


#FOCE-I ------------------------------------------------------------------------
data<-read.nm.tables("Sim_test_focei_split_8.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'IPRED' for positions 1, 116, 231, etc.
num_positions <- nrow(data_test)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$IPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_a <- data.frame(Position = seq(1, num_positions), Mean_IPRED = mean_values)

# Save the result data frame to a CSV file
write.csv(result_df_a, file = sprintf("mean_ipred_values_sim_focei_%d.csv", m), row.names = FALSE)

#DV vs IPRED for simulated results
# predictions from simulation
prediction<-result_df_a$Mean_IPRED

#true values from test data
actual<- data_test$DV

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  #geom_smooth(method = "lm", se = FALSE, color = "red", size=0.5) + ##R^2
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentration vs. Individual Predicted Concentration 5FU \n Method: FOCE-I (test data) [8]") +
  annotate("text", x = 0.8, y = 1.9,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,2)) +
  scale_y_continuous(limits=c(0,2)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [mg/L]") + ylab("Observed Concentration [mg/L]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_%d.jpg", m), width=6, height=6)


#SAEM --------------------------------------------------------------------------

data<-read.nm.tables("Sim_test_saem_split_8.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'IPRED' for positions 1, 116, 231, etc.
num_positions <- nrow(data_test)

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$IPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
}

# Print the mean values
print(mean_values)

# Create a data frame with the mean values
result_df_b <- data.frame(Position = seq(1, num_positions), Mean_IPRED = mean_values)

# Save the result data frame to a CSV file
write.csv(result_df_b, file = sprintf("mean_ipred_values_sim_saem_%d.csv", m), row.names = FALSE)

#DV vs IPRED for simulated results
# predictions from simulation
prediction<-result_df_b$Mean_IPRED

#true values from test data
actual<- data_test$DV

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  #geom_smooth(method = "lm", se = FALSE, color = "red", size=0.5) + ##R^2
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentration vs. Individual Predicted Concentration 5FU \n Method: SAEM-I (test data) [8]") +
  annotate("text", x = 0.8, y = 1.9,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,2)) +
  scale_y_continuous(limits=c(0,2)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [mg/L]") + ylab("Observed Concentration [mg/L]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_%d.jpg", m), width=6, height=6)


# Section 1.2 :  CWRES plots ---------------------------------------------------

# before, the data was preprocessed: Text in Spalten, "Table 1" header erased

# FOCE-I  ----------------------------------------------------------------------
data_cwres<- read.csv("5fu_focei_split_8.csv", sep=";")

# convert some columns to numeric
data_cwres <- data_cwres %>%
  mutate(across(c(IPRED, PRED, CWRES, TIME), as.numeric))

data_cwres <- data_cwres %>%
  filter(IPRED != 0)

#CWRES vs PRED
cwres_pred<- ggplot(data_cwres,aes(x=PRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c(0,2))+
  scale_y_continuous(limits=c(-4,6))+
  ggtitle ("Conditional Weighted Residuals (CWRES) vs. Predicted Concentration 5FU \n Method: FOCE-I (training data) [8]") +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  ylab("Conditional Weighted Residuals (CWRES)")+ xlab("Predicted Concentration [mg/L]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_focei_%d.jpg", m), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_x_continuous(limits=c(15,26))+
  scale_y_continuous()+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weighted Residuals (CWRES) vs. Time 5FU \n Method: FOCE-I (training data) [8]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weighted Residuals (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_focei_%d.jpg", m), width=6, height=6)


# SAEM  ------------------------------------------------------------------------
data_cwres<- read.csv("5fu_saem_split_8.csv", sep=";")

# convert some columns to numeric
data_cwres <- data_cwres %>%
  mutate(across(c(IPRED, PRED, CWRES, TIME), as.numeric))

data_cwres <- data_cwres %>%
  filter(IPRED != 0)

#CWRES vs PRED
cwres_pred<- ggplot(data_cwres,aes(x=PRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c(0,2))+
  scale_y_continuous(limits=c(-4,6))+
  ggtitle ("Conditional Weighted Residuals (CWRES) vs. Predicted Concentration 5FU \n Method: SAEM-I (training data) [8]") +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  ylab("Conditional Weighted Residuals (CWRES)")+ xlab("Predicted Concentration [mg/L]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_saem_%d.jpg", m), width=6, height=6)


#CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_x_continuous(limits=c(15,26))+
  scale_y_continuous()+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weighted Residuals (CWRES) vs. Time 5FU \n Method: SAEM-I (training data) [8]") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weighted Residuals (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_saem_%d.jpg", m), width=6, height=6)



# Section 1.3 :  ETA distribution  ---------------------------------------------

# FOCE-I  ----------------------------------------------------------------------
# for looking at the ETAs
data_etas <-read.nm.tables("etas_split8_focei")
data_etas <- subset(data_etas, DV != 0)

# Select only the first row per ID
data_etas <- data_etas %>%
  group_by(ID) %>%
  slice(1) %>%
  ungroup()

# Create the ETA distribution plot using ggplot2
# ETA distribution of CL (V is fixed)
eta_plot <-ggplot(data_etas, aes(x=ETA1)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA- Density (ETA CL) \n Method: FOCE-I 5FU (training data) [8]",
    x = "ETA value",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_train_focei_%d.jpg",m), width= 6, height= 6)


# SAEM -------------------------------------------------------------------------
# for looking at the ETAs
data_etas <-read.nm.tables("etas_split8_saem")
data_etas <- subset(data_etas, DV != 0)

# Select only the first row per ID
data_etas <- data_etas %>%
  group_by(ID) %>%
  slice(1) %>%
  ungroup()

# Create the ETA distribution plot using ggplot2
# ETA distribution of CL (V is fixed)
eta_plot <-ggplot(data_etas, aes(x=ETA1)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA- Density (ETA CL) \n Method: SAEM-I 5FU (training data) [8]",
    x = "ETA value",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

#Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_train_saem_%d.jpg",m), width= 6, height= 6)


# Section 1.4 MAPE, MAE, MSE for NONMEM ----------------------------------------

data_5fu <- read.csv("corrected_NM_Data_final_clean.csv", sep=",") # for DV vs IPRED and other GOF, with 23
data_test <-subset(data_5fu, SET8 == 1) # test set
data_test <- subset(data_test, DV != 0)
actual<- data_test$DV


# FOCE-I  ----------------------------------------------------------------------
# IPRED from simulation data compared with test data DV
prediction<-result_df_a$Mean_IPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate MSE and RMSE
mse_dv <- MSE(compare$pred,compare$actual)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate 100- MAPE
pmape_dv<-100-mape_dv

# Create a data frame to store the calculated values
quality <- data.frame(
  MSE = mse_dv,
  RMSE = rmse_dv,
  MAE = mae_dv,
  MAPE = mape_dv,
  PMAPE = pmape_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_focei_%d.csv", m), row.names = FALSE)

# SAEM  ------------------------------------------------------------------------
# IPRED from simulation data compared with test data DV
prediction<-result_df_b$Mean_IPRED

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

#calculate MSE and RMSE
mse_dv <- MSE(compare$pred,compare$actual)
rmse_dv <- RMSE(compare$pred,compare$actual)

#calculate MAE (mean absolute error)
mae_dv <- MAE(compare$pred,compare$actual)

#calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

#calculate 100- MAPE
pmape_dv<-100-mape_dv

# Create a data frame to store the calculated values
quality <- data.frame(
  MSE = mse_dv,
  RMSE = rmse_dv,
  MAE = mae_dv,
  MAPE = mape_dv,
  PMAPE = pmape_dv
)

# Print the data frame to see the results
print(quality)

# save results
write.csv(quality, file = sprintf("quality_measures_test_saem_%d.csv", m), row.names = FALSE)

