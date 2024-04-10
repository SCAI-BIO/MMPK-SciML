# 5FU Pop-PK evaluation -------------------------------------------------------


#--------------------------------------------------#
# Evaluation of PMX 5FU model                      #
#--------------------------------------------------#
# Split 2                                          #
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


setwd("C:/Users/lucak/Documents/Uni Bonn/Projekte mit Olga/5FU/Auswertungen/NONMEM/Split2")

data_5fu <- read.csv("NM_data_final.csv") # for DV vs IPRED and other GOF
data_5fu <- subset(data_5fu, TRAIN_2 == 0) # test data
data_5fu <- subset(data_5fu, DV != 0)


# Split number m
# Augmentation percentage n

m <- 2
n <- 0

# Section 1:  DV vs IPRED ------------------------------------------------------

# Mean IPRED from simulation 

# FOCE -------------------------------------------------------------------------
data<-read.nm.tables("5fu_foce_sim_split_2.tab")


# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'IPRED' 
num_positions <- nrow(data_5fu)
num_individuals <- nrow(data_5fu)

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
write.csv(result_df_a, file = sprintf("mean_ipred_values_sim_foce_5fu_%d_%d.csv", m,n), row.names = FALSE)

# DV vs IPRED for simulated results
# predictions from simulation
prediction<-result_df_a$Mean_IPRED

# true values from test data
actual<- data_5fu$DV

# create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [2] \n Method: FOCE (test data)") +
  annotate("text", x = min(compare$pred)+0.2, y = max(compare$actual)+1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_foce_5fu_%d_%d.jpg", m,n), width=6, height=6)

# FOCEI -------------------------------------------------------------------------
data<-read.nm.tables("5fu_focei_sim_split_2.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'IPRED' 
num_positions <- nrow(data_5fu)
num_individuals <- nrow(data_5fu)

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
write.csv(result_df_a, file = sprintf("mean_ipred_values_sim_focei_5fu_%d_%d.csv", m,n), row.names = FALSE)

# DV vs IPRED for simulated results
# predictions from simulation
prediction<-result_df_a$Mean_IPRED

# true values from test data
actual<- data_5fu$DV

# create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [2] \n Method: FOCE-I (test data)") +
  annotate("text", x = min(compare$pred)+0.2, y = max(compare$actual)+1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_5fu_%d_%d.jpg", m,n), width=6, height=6)


# SAEM -------------------------------------------------------------------------
data<-read.nm.tables("5fu_saem_sim_split_2.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'IPRED' 
num_positions <- nrow(data_5fu)
num_individuals <- nrow(data_5fu)

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
write.csv(result_df_b, file = sprintf("mean_ipred_values_sim_saem_5fu_%d_%d.csv", m,n), row.names = FALSE)

# DV vs IPRED for simulated results
# predictions from simulation
prediction<-result_df_b$Mean_IPRED

# true values from test data
actual<- data_5fu$DV

# create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [2] \n Method: SAEM (test data)") +
  annotate("text", x = min(compare$pred)+0.2, y = max(compare$actual)+1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_5fu_%d_%d.jpg", m,n), width=6, height=6)


# Section 2: MAPE, RMSE, MAE for NONMEM -----------------------------------------

data_5fu <- read.csv("NM_data_final.csv") # for DV vs IPRED and other GOF
data_5fu <- subset(data_5fu, DV != 0 & TRAIN_2 == 0) # test
actual<- data_5fu$DV

# FOCE -------------------------------------------------------------------------
# individual predictions from simulation data (compared with test!)
prediction<-result_df_a$Mean_IPRED

# create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

# calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

# calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

# calculate MAE (mean absolute error)
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
write.csv(quality, file = sprintf("5fu_quality_measures_test_foce_%d_%d.csv", m,n), row.names = FALSE)

# FOCE-I -----------------------------------------------------------------------
# individual predictions from simulation data (compared with test!)
prediction<-result_df_a$Mean_IPRED

#c reate data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

# calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

# calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

# calculate MAE (mean absolute error)
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
write.csv(quality, file = sprintf("5fu_quality_measures_test_focei_%d_%d.csv", m,n), row.names = FALSE)


# SAEM flag --------------------------------------------------------------------
# individual predictions from simulation data (compared with test!)
prediction<-result_df_b$Mean_IPRED

# create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

# calculate RMSE (root mean squared error)
rmse_dv <- RMSE(compare$pred,compare$actual)

# calculate MAPE (Mean absolute percentage error)
mape_dv<-MAPE(compare$pred,compare$actual)*100

# calculate MAE (mean absolute error)
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
write.csv(quality, file = sprintf("5fu_quality_measures_test_saem_%d_%d.csv", m,n), row.names = FALSE)


# Section 3 :  CWRES vs IPRED --------------------------------------------------

# FOCE -------------------------------------------------------------------------
data_cwres<- read.csv("5fu_foce_split_2.csv")

data_cwres <- data_cwres %>%
  filter(TIME != 'TIME' & IPRED != 0)

# CWRES vs IPRED
cwres_pred<- ggplot(data_cwres,aes(x=IPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c(0,max(compare$actual)+3)) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations [2] \n Method: FOCE (train data)") +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_foce_5fu_%d_%d.jpg", m,n), width=6, height=6)


# CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time [2] \n Method: FOCE (train data)") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_foce_5fu_%d_%d.jpg", m,n), width=6, height=6)


# FOCE-I  ----------------------------------------------------------------------
data_cwres<- read.csv("5fu_focei_split_2.csv")

data_cwres <- data_cwres %>%
  filter(TIME != 'TIME' & IPRED != 0)

#CWRES vs IPRED
cwres_pred<- ggplot(data_cwres,aes(x=IPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c(0,max(compare$actual)+3)) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations [2] \n Method: FOCE-I (train data)") +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_focei_5fu_%d_%d.jpg", m,n), width=6, height=6)


# CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time [2] \n Method: FOCE-I (train data)") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_focei_5fu_%d_%d.jpg", m,n), width=6, height=6)


# SAEM -------------------------------------------------------------------------
data_cwres<- read.csv("5fu_saem_split_2.csv")

data_cwres <- data_cwres %>%
  filter(TIME != 'TIME' & IPRED != 0)

# CWRES vs IPRED
cwres_pred<- ggplot(data_cwres,aes(x=IPRED,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  scale_x_continuous(limits=c(0,max(compare$actual)+3)) +
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Individual Predicted Concentrations [2] \n Method: SAEM (train data)") +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Individual Predicted Concentration [ng/mL]")
cwres_pred

ggsave(sprintf("cwres_pred_plot_train_saem_5fu_%d_%d.jpg", m,n), width=6, height=6)


# CWRES vs TIME
cwres_time<- ggplot(data_cwres,aes(x=TIME,y=CWRES))+
  geom_point(color="#909085", size=1, alpha=0.5)+
  scale_y_continuous(limits=c((min(data_cwres$TIME)),(max(data_cwres$TIME)+1)))+
  scale_y_continuous(limits=c((min(data_cwres$CWRES)-1),(max(data_cwres$CWRES)+1)))+
  geom_abline(intercept = 0, slope = 0,size=0.5)+ # Add line of unity
  ggtitle ("Conditional Weight Residual Errors (CWRES) vs. Time [2] \n Method: SAEM (train data)") +
  theme(plot.title = element_text(size=12, hjust=0.5)) + # Set theme
  ylab("Conditional Weight Residual Error (CWRES)")+ xlab("Time [h]")
cwres_time

ggsave(sprintf("cwres_time_plot_train_saem_5fu_%d_%d.jpg", m,n), width=6, height=6)


# Section 4 :  ETA distribution  -----------------------------------------------

# FOCE --------------------------------------------------------------------------

data_etas <-read.nm.tables("etas_5fu_foce_split_2.tab")
data_etas <- subset(data_etas, DV != 0)

# Select just one eta per patient (baseline)
data_etas <- data_etas %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Create the ETA distribution plot using ggplot2
# ETA distribution of IIV CL ---------------------------------------------------
eta_plot <-ggplot(data_etas, aes(x=ETA1)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
                    geom_density(alpha = 1, col= "#004a9f") +
                    labs(
                      title = "ETA Distribution (IIV CL) [2] \n Method: FOCE (train data)",
                      x = "ETA",
                      y = "Density"
                    ) +
                    theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
                    theme(legend.position = "top")

# Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_cl_5fu_foce_%d_%d.jpg",m,n), width= 6, height= 6)

# FOCE-I ------------------------------------------------------------------------

data_etas <-read.nm.tables("etas_5fu_focei_split_2.tab")
data_etas <- subset(data_etas, DV != 0)

# Select just one eta per patient (baseline)
data_etas <- data_etas %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Create the ETA distribution plot using ggplot2
# ETA distribution of IIV CL ---------------------------------------------------
eta_plot <-ggplot(data_etas, aes(x=ETA1)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV CL) [2] \n Method: FOCE-I (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

# Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_cl_5fu_focei_%d_%d.jpg",m,n), width= 6, height= 6)

# SAEM -------------------------------------------------------------------------

data_etas <-read.nm.tables("etas_5fu_saem_split_2.tab")
data_etas <- subset(data_etas, DV != 0)

# Select just one eta per patient (baseline)
data_etas <- data_etas %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Create the ETA distribution plot using ggplot2
# ETA distribution of IIV CL ---------------------------------------------------
eta_plot <-ggplot(data_etas, aes(x=ETA1)) +geom_histogram(aes(y = after_stat(density)), binwidth=0.05, col = "black") +  
  geom_density(alpha = 1, col= "#004a9f") +
  labs(
    title = "ETA Distribution (IIV CL) [2] \n Method: SAEM (train data)",
    x = "ETA",
    y = "Density"
  ) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  theme(legend.position = "top")

# Display the plot
print(eta_plot)

ggsave(sprintf("eta_distribution_plot_cl_5fu_saem_%d_%d.jpg",m,n), width= 6, height= 6)

