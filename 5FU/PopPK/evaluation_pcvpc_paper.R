# PC-VPC pSN for paper ---------------------------------------------------------

#--------------------------------------------------#
# prediction corrected VPC                         #                                                 
#--------------------------------------------------#
# 5FU: split 8, Suni: Split 5, FOCE-I              #
#--------------------------------------------------#

#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots
#xpose4: needed to create pc-vpc and to read nonmem tables

rm(list=ls())

library(xpose4)
library(dplyr) 
library(ggplot2)

setwd("C:/Users/Olga/5FU_NONMEM_neu/vpc_paper")

# Sunitinib --------------------------------------------------------------------

#--------------------------------------------------#
# FOCE-I-> vpc2                                    #
#--------------------------------------------------#

# FOCE-I (parent) --------------------------------------------------------------

Number_of_simulations_performed <-1000 ## Used as 'samples' argument in PsN vpc function. Needs to be a round number.


# Set the prediction intervals

PI <- 80 #specify prediction interval in percentage. Common is 80% interval (10-90%)
CI <- 95 #specify confidence interval in percentage. Common is 95


# Make a vector for upper and lower percentile
perc_PI <- c(0+(1-PI/100)/2, 1-(1-PI/100)/2)
perc_CI <- c(0+(1-CI/100)/2, 1-(1-CI/100)/2)


# Specify the bin times manually
bin_times <- c(48,312,336,516,1488,2700) 
Number_of_bins <- length(bin_times)-1


######### Load in simulated data 

## Search for the generated file by PsN in the sub-folders
files <- list.files(pattern = "2.npctab.dta", recursive = TRUE, include.dirs = TRUE)

# This reads the VPC npctab.dta file and save it in dataframe_simulations
dataframe_simulations <- read.nm.tables(paste('.\\',files,sep=""))
dataframe_simulations <- dataframe_simulations[dataframe_simulations$MDV == 0,]
dataframe_simulations <- dataframe_simulations[dataframe_simulations$CMT == 2,] # parent
dataframe_simulations$PRED <- exp(dataframe_simulations$PRED)
dataframe_simulations$DV <- exp(dataframe_simulations$DV)


## Set the replicate number to the simulated dataset
dataframe_simulations$replicate <- rep(1:Number_of_simulations_performed,each=nrow(dataframe_simulations)/Number_of_simulations_performed)


# Set vector with unique replicates
Replicate_vector <- unique(dataframe_simulations$replicate)

# Create bins in simulated dataset

### Set bins by backward for-loop
for(i in Number_of_bins:1){
  dataframe_simulations$BIN[dataframe_simulations$TIME <= bin_times[i+1]] <-i
} 

## Calculate median PRED per bin
PRED_BIN <- dataframe_simulations[dataframe_simulations$replicate ==1,] %>%
  group_by(BIN) %>% # Calculate the PRED per bin
  summarize(PREDBIN = median(PRED))

dataframe_simulations <- merge(dataframe_simulations,PRED_BIN,by='BIN')

# Calculate prediction corrected simulated observations (PCDV)
dataframe_simulations$PCDV <- dataframe_simulations$DV *(dataframe_simulations$PREDBIN/dataframe_simulations$PRED)

dataframe_simulations <- dataframe_simulations[order(dataframe_simulations$replicate,dataframe_simulations$ID,dataframe_simulations$TIME),]

sim_PI <- NULL

## Calculate predictions intervals per bin
for(i in Replicate_vector){
  
  # Run this for each replicate
  sim_vpc_ci <- dataframe_simulations %>%
    filter(replicate %in% i) %>% # Select an individual replicate
    group_by(BIN) %>% # Calculate everything per bin
    summarize(C_median = median(PCDV), C_lower = quantile(PCDV, perc_PI[1]), C_upper = quantile(PCDV, perc_PI[2])) %>% # Calculate prediction intervals
    mutate(replicate = i) # Include replicate number
  
  sim_PI <- rbind(sim_PI, sim_vpc_ci)
}

# Calculate confidence intervals around these prediction intervals calculated with each replicate

sim_CI <- sim_PI %>%
  group_by(BIN) %>%
  summarize(C_median_CI_lwr = quantile(C_median, perc_CI[1]), C_median_CI_upr = quantile(C_median, perc_CI[2]), # Median
            C_low_lwr = quantile(C_lower, perc_CI[1]), C_low_upr = quantile(C_lower, perc_CI[2]), # Lower percentages
            C_up_lwr = quantile(C_upper, perc_CI[1]), C_up_upr = quantile(C_upper, perc_CI[2]) # High percentages
  )

### Set bin boundaries in dataset
sim_CI$x1 <- NA
sim_CI$x2 <- NA

for(i in 1:Number_of_bins){
  sim_CI$x1[sim_CI$BIN == i] <-bin_times[i]
  sim_CI$x2[sim_CI$BIN == i] <-bin_times[i+1]
}

######### Read dataset with original observations
data <- read.csv("Suni_PK_final_new.csv")# for DV vs IPRED and other GOF
data <- subset(data, SET_5 == 1) # test data
data <- subset(data, MDV != 1)
Obs <- subset(data, CMT == 2) # parent


### Add the population prediction to each observation (only use the data from 1 replicate)
Rep1 <-  dataframe_simulations[ 
  dataframe_simulations$replicate ==1,"PRED"] 


Obs <- cbind(Obs,Rep1)
Obs$DV <- as.numeric(Obs$DV)

# Repeat the observations from Obs to match the length of dataframe_simulations
repeated_obs <- Obs[rep(seq_len(nrow(Obs)), length.out = nrow(dataframe_simulations)), ]

# Reset row names to match the repeated observations
rownames(repeated_obs) <- NULL

# Rename the column 'DV' in repeated_obs
repeated_obs <- rename(repeated_obs, DV_orig = DV)

# Rename the column 'DV' in repeated_obs and bind it to dataframe_simulations
result <- dataframe_simulations %>%
  bind_cols(DV_orig = repeated_obs$DV_orig)

# Save the result dataframe to a CSV file
write.csv(result, "Suni_vpc_spl_5.csv", row.names = FALSE)


# 5FU --------------------------------------------------------------------------

# FOCE-I -----------------------------------------------------------------------

Number_of_simulations_performed <-1000 ## Used as 'samples' argument in PsN vpc function. Needs to be a round number.


# Set the prediction intervals

PI <- 80 #specify prediction interval in percentage. Common is 80% interval (10-90%)
CI <- 95 #specify confidence interval in percentage. Common is 95


# Make a vector for upper and lower percentile
perc_PI <- c(0+(1-PI/100)/2, 1-(1-PI/100)/2)
perc_CI <- c(0+(1-CI/100)/2, 1-(1-CI/100)/2)


# Specify the bin times manually
bin_times<- c(17,18,19,24.5)
Number_of_bins <- length(bin_times)-1


######### Load in simulated data 

## Search for the generated file by PsN in the sub-folders
files <- list.files(pattern = "151.npctab.dta", recursive = TRUE, include.dirs = TRUE)

# This reads the VPC npctab.dta file and save it in dataframe_simulations
dataframe_simulations <- read.nm.tables(paste('.\\',files,sep=""))
dataframe_simulations <- dataframe_simulations[dataframe_simulations$MDV == 0,]



## Set the replicate number to the simulated dataset
dataframe_simulations$replicate <- rep(1:Number_of_simulations_performed,each=nrow(dataframe_simulations)/Number_of_simulations_performed)


# Set vector with unique replicates
Replicate_vector <- unique(dataframe_simulations$replicate)

# Create bins in simulated dataset

### Set bins by backward for-loop
for(i in Number_of_bins:1){
  dataframe_simulations$BIN[dataframe_simulations$TIME <= bin_times[i+1]] <-i
} 

## Calculate median PRED per bin
PRED_BIN <- dataframe_simulations[dataframe_simulations$replicate ==1,] %>%
  group_by(BIN) %>% # Calculate the PRED per bin
  summarize(PREDBIN = median(PRED))

dataframe_simulations <- merge(dataframe_simulations,PRED_BIN,by='BIN')

# Calculate prediction corrected simulated observations (PCDV)
dataframe_simulations$PCDV <- dataframe_simulations$DV *(dataframe_simulations$PREDBIN/dataframe_simulations$PRED)

dataframe_simulations <- dataframe_simulations[order(dataframe_simulations$replicate,dataframe_simulations$ID,dataframe_simulations$TIME),]

sim_PI <- NULL

## Calculate predictions intervals per bin
for(i in Replicate_vector){
  
  # Run this for each replicate
  sim_vpc_ci <- dataframe_simulations %>%
    filter(replicate %in% i) %>% # Select an individual replicate
    group_by(BIN) %>% # Calculate everything per bin
    summarize(C_median = median(PCDV), C_lower = quantile(PCDV, perc_PI[1]), C_upper = quantile(PCDV, perc_PI[2])) %>% # Calculate prediction intervals
    mutate(replicate = i) # Include replicate number
  
  sim_PI <- rbind(sim_PI, sim_vpc_ci)
}

# Calculate confidence intervals around these prediction intervals calculated with each replicate

sim_CI <- sim_PI %>%
  group_by(BIN) %>%
  summarize(C_median_CI_lwr = quantile(C_median, perc_CI[1]), C_median_CI_upr = quantile(C_median, perc_CI[2]), # Median
            C_low_lwr = quantile(C_lower, perc_CI[1]), C_low_upr = quantile(C_lower, perc_CI[2]), # Lower percentages
            C_up_lwr = quantile(C_upper, perc_CI[1]), C_up_upr = quantile(C_upper, perc_CI[2]) # High percentages
  )

### Set bin boundaries in dataset
sim_CI$x1 <- NA
sim_CI$x2 <- NA

for(i in 1:Number_of_bins){
  sim_CI$x1[sim_CI$BIN == i] <-bin_times[i]
  sim_CI$x2[sim_CI$BIN == i] <-bin_times[i+1]
}

######### Read dataset with original observations
data_5fu <- read.csv("corrected_NM_Data_final.csv", sep=",")
data_5fu <- subset(data_5fu, SET8 == 1) # pick test data
Obs <- subset(data_5fu, MDV != 1)

### Add the population prediction to each observation (only use the data from 1 replicate)
Rep1 <-  dataframe_simulations[ 
  dataframe_simulations$replicate ==1,"PRED"] 


Obs <- cbind(Obs,Rep1)
Obs$DV <- as.numeric(Obs$DV)

# Repeat the observations from Obs to match the length of dataframe_simulations
repeated_obs <- Obs[rep(seq_len(nrow(Obs)), length.out = nrow(dataframe_simulations)), ]

# Reset row names to match the repeated observations
rownames(repeated_obs) <- NULL

# Rename the column 'DV' in repeated_obs
repeated_obs <- rename(repeated_obs, DV_orig = DV)

# Rename the column 'DV' in repeated_obs and bind it to dataframe_simulations
result <- dataframe_simulations %>%
  bind_cols(DV_orig = repeated_obs$DV_orig)

# Save the result dataframe to a CSV file
write.csv(result, "5FU_vpc_spl_8.csv", row.names = FALSE)

