# Suni Pop-PK evaluation -------------------------------------------------------


#--------------------------------------------------#
# Evaluation of PMX Suni model MAP                 #
#--------------------------------------------------#
# Split 5                                          #
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


setwd("C:/Users/teply/Documents/Suni_NONMEM_final - Kopie/Split5")

data_suni <- read.csv("Suni_PK_final_raw.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, SET_5 == 1) # test data
data_suni <- subset(data_suni, MDV != 1)
data_suni <- subset(data_suni, C != 1) # no outliers

# Split number m
m <-5


# Section 1:  DV vs IPRED ------------------------------------------------------
# 1.1:  Parent drug ------------------------------------------------------------

#FOCE-I-------------------------------------------------------------------------
# filter out the observations for the parent drug
data_suni <- subset(data_suni, CMT == 2) # parent

# predictions from simulation
data<-read.nm.tables("Suni_focei_sim_split_map_5.tab")

# Filter rows where MDV is 0 and CMT is 2 (parent)
data <- data %>%
  filter(MDV == 0, CMT == 2)

#DV vs EXPIPRED 
# predictions from MAP-estimation
prediction<-data$EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction, actual = actual)

# save results
write.csv(compare, file = sprintf("gof_table_focei_parent_map_%d.csv", m), row.names = FALSE)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: FOCE-I (test data) Parent") +
  annotate("text", x = 18, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_suni_parent_map_%d.jpg", m), width=6, height=6)


#SAEM-I-------------------------------------------------------------------------
data<-read.nm.tables("Suni_saem_sim_split_map_5.tab")

# Filter rows where MDV is 0
data <- data %>%
  filter(MDV == 0, CMT == 2)

#DV vs EXPIPRED
# predictions from MAP-estimation
prediction_b<-data$EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction_b, actual = actual)

# save results
write.csv(compare, file = sprintf("gof_table_saemi_parent_map_%d.csv", m), row.names = FALSE)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: SAEM-I (test data) Parent") +
  annotate("text", x = 18, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_suni_parent_map_%d.jpg", m), width=6, height=6)

# 1.2:  Metabolite -------------------------------------------------------------

#FOCE-I-------------------------------------------------------------------------
# filter out the observations for the metabolite
data_suni <- read.csv("Suni_PK_final_raw.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, SET_5 == 1) # test data
data_suni <- subset(data_suni, MDV != 1)
data_suni <- subset(data_suni, CMT == 3) # metabolite
data_suni <- subset(data_suni, C != 1) # no outliers

# predictions from simulation
data<-read.nm.tables("Suni_focei_sim_split_map_5.tab")

# Filter rows where MDV is 0 and CMT is 3 (metabolite)
data <- data %>%
  filter(MDV == 0, CMT == 3)

#DV vs EXPIPRED 
# predictions from MAP estimation
prediction_c<-data$EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction_c, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: FOCE-I (test data) SU12662") +
  annotate("text", x = 18, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_focei_suni_metabolite_map_%d.jpg", m), width=6, height=6)


#SAEM-I-------------------------------------------------------------------------
data<-read.nm.tables("Suni_saem_sim_split_map_5.tab")

# Filter rows where MDV is 0
data <- data %>%
  filter(MDV==0, CMT == 3)

#DV vs EXPIPRED 
# predictions from MAP-estimation
prediction_d<-data$EXPIPRED

#true values from test data
actual<- data_suni$DV
actual <- as.double(actual)

#create data frame with a column of actual values and a column of predicted values
compare <- data.frame(pred = prediction_d, actual = actual)

model<-lm(compare$pred~compare$actual)

dv_ipred <- ggplot(data=compare,aes(x=pred,y=actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.5)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: SAEM-I (test data) SU12662") +
  annotate("text", x = 18, y = max(compare$actual)-1,
           label = paste("Adjusted R²:", round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  scale_x_continuous(limits=c(0,max(compare$pred)+1)) +
  scale_y_continuous(limits=c(0,max(compare$actual)+1)) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_test_saem_suni_metabolite_map_%d.jpg", m), width=6, height=6)

# Section 2: MAPE, RMSE, MAE for NONMEM -----------------------------------------

# 2.1:  Parent drug ------------------------------------------------------------

data_suni <- read.csv("Suni_PK_final_raw.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, MDV != 1 & SET_5 == 1) # test
data_suni <- subset(data_suni, CMT == 2) # parent
data_suni <- subset(data_suni, C != 1) # no outliers
actual<- data_suni$DV
actual <- as.double(actual)

# FOCE-I -----------------------------------------------------------------------
# exp. individual predictions from MAP-estimation data (compared with test!)
prediction<-prediction

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
write.csv(quality, file = sprintf("quality_measures_test_focei_parent_map_%d.csv", m), row.names = FALSE)


# SAEM-I------------------------------------------------------------------------
# exp. individual predictions from MAP-estimation data (compared with test!)
prediction<-prediction_b

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
write.csv(quality, file = sprintf("quality_measures_test_saem_parent_map_%d.csv", m), row.names = FALSE)

# 2.2:  Metabolite -------------------------------------------------------------

data_suni <- read.csv("Suni_PK_final_raw.csv") # for DV vs IPRED and other GOF
data_suni <- subset(data_suni, MDV != 1 & SET_5 == 1) # test
data_suni <- subset(data_suni, CMT == 3) # metabolite
data_suni <- subset(data_suni, C != 1) # no outliers
actual<- data_suni$DV
actual <- as.double(actual)

# FOCE-I -----------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-prediction_c

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
write.csv(quality, file = sprintf("quality_measures_test_focei_metabolite_map_%d.csv", m), row.names = FALSE)


# SAEM-I------------------------------------------------------------------------
# exp. individual predictions from simulation data (compared with test!)
prediction<-prediction_d

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
write.csv(quality, file = sprintf("quality_measures_test_saem_metabolite_map_%d.csv", m), row.names = FALSE)

