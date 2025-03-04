# Section 1: 5-FU Pop-PK evaluation --------------------------------------------

#--------------------------------------------------#
#--------------------------------------------------#
# Evaluation of PMX 5-FU model MAP                 #
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

# Split8

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
data<-read.nm.tables("Sim_map_test_focei_split_8.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

#DV vs IPRED for MAP results
# predictions from MAP estimation
prediction_a<-data$IPRED

#true values from test data
actual<- data$DV

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction_a, actual = actual)

# save results
write.csv(compare, file = sprintf("gof_table_5fu_focei_map_%d.csv", m), row.names = FALSE)

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

ggsave(sprintf("dv_ipred_plot_test_focei_map_%d.jpg", m), width=6, height=6)


#SAEM-I--------------------------------------------------------------------------

data<-read.nm.tables("Sim_map_test_saem_split_8.tab")

# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

#DV vs IPRED for MAP results
# predictions from MAP estimation
prediction_b<-data$IPRED

#true values from test data
actual<- data$DV

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction_b, actual = actual)

# save results
write.csv(compare, file = sprintf("gof_table_5fu_saem_map_%d.csv", m), row.names = FALSE)


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

ggsave(sprintf("dv_ipred_plot_test_saem_map_%d.jpg", m), width=6, height=6)


# Section 1.4 MAPE, MAE, MSE for NONMEM ----------------------------------------

data_5fu <- read.csv("corrected_NM_Data_final_clean.csv", sep=",") # for DV vs IPRED and other GOF, with 23
data_test <-subset(data_5fu, SET8 == 1) # test set
data_test <- subset(data_test, DV != 0)
actual<- data_test$DV


# FOCE-I  ----------------------------------------------------------------------
# IPRED from MAP estimation data compared with test data DV
prediction<-prediction_a

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
write.csv(quality, file = sprintf("quality_measures_test_focei_map_%d.csv", m), row.names = FALSE)

# SAEM-I  ------------------------------------------------------------------------
# IPRED from simulation data compared with test data DV
prediction<-prediction_b

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
write.csv(quality, file = sprintf("quality_measures_test_saem_map_%d.csv", m), row.names = FALSE)

