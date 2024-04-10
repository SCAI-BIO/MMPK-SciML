# Suni Augmentation ------------------------------------------------------------
#--------------------------------------------------#
# Data Augmentation Sunitinib                      #
#--------------------------------------------------#
#--------------------------------------------------#

# Split 1

#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr and tibble: data wrangling
#ggplot2: create plots
#xpose4: needed to create pc-vpc and to read nonmem tables
#mvLognCorrEst: simulate from bivariate lognormal distribution
#lubridate: Edit dates and times

rm(list=ls())

library(dplyr) 
library(ggplot2)
library(xpose4)
#install.packages("remotes")
#remotes::install_github("AlessandroDeCarlo27/mvlognCorrEst")
#install.packages("Matrix")
#install.packages("igraph")
#install.packages("qgraph")
#install.packages("pracma")
library(mvLognCorrEst)
library(lubridate)
library(tibble)

setwd("C:/Users/Olga/Suni_Pazo/Augmentation/Split1")

# Split number m
m <-1


# Section 1: Demographics of Train Data for Augmentation -----------------------

data_suni <- read.csv("Suni_PK_final_new.csv")
data_suni[data_suni == -99] <- NA

data_suni <- data_suni[complete.cases(data_suni), ]


# Define patient IDs for train and test sets
Test_ids <- unique(data_suni$ID[data_suni$SET_1 == 1])
Train_ids <- unique(data_suni$ID[data_suni$SET_1 == 0])


# Calculate individual mean values 
individual_mean_values <- aggregate(cbind(AGE, WEIGHT, HEIGHT) ~ ID, data = data_suni, FUN = mean, na.rm = TRUE)


# Calculate the mean and SD of TAD by ID (but <24h, not after breaks)
individual_tad <- aggregate(cbind(TAD) ~ ID, 
                            data = subset(data_suni, TAD < 24), 
                            FUN = mean, 
                            na.rm = TRUE)

mean_tad<- mean(individual_tad$TAD)
sd_tad <- sd(individual_tad$TAD)
                                    

# Filter data for Train and calculate mean values
Train_mean_values <- individual_mean_values %>%
  filter(ID %in% Train_ids) %>%
  summarize(
    mean_age = mean(AGE),
    sd_age = sd(AGE),
    mean_wt = mean(WEIGHT),
    sd_wt = sd(WEIGHT),
    mean_hgt = mean(HEIGHT),
    sd_hgt = sd(HEIGHT))

# Add mean and SD of TAD to dataframe
Train_mean_values$mean_tad <- mean_tad
Train_mean_values$sd_tad <- sd_tad


# Filter data and count the number of women and men for Train
Train_women_count <- data_suni %>%
  filter(ID %in% Train_ids, SEX == 0) %>%
  distinct(ID) %>%
  nrow()

# Calculate percentage of men and women 
percentage_women<- Train_women_count/length(Train_ids)
percentage_men<-1-percentage_women

Train_mean_values$women_perc<- percentage_women
Train_mean_values$men_perc<- percentage_men

# summarize range of values for train
Train_values <- individual_mean_values %>%
  filter(ID %in% Train_ids)
summary(Train_values)
range_train<- summary(Train_values)

# Save the DataFrame to a CSV file
write.csv(Train_mean_values, file = sprintf("Suni_train_demo_split_%d.CSV", m), row.names = FALSE)
write.csv(range_train, file = sprintf("range_train_demo_5fu_%d.csv", m), row.names = FALSE)


# Section 2: Simulation of Weight and Height -----------------------------------

data_suni<- data_suni %>%
  filter(ID %in% Train_ids) 

WEIGHT<- data_suni$WEIGHT
HEIGHT<- data_suni$HEIGHT
df<- data.frame(Weight = WEIGHT, Height = HEIGHT)

# calculate correlation matrix; Spearman: not neccesarily linear correlation, but monotonous
correlation <- cor(df, method = "spearman") 
cor<-correlation[1, 2] 

# Create a scatter plot
scatter_plot <- ggplot(data = df, aes(x = WEIGHT, y = HEIGHT)) +
  geom_point(color = "blue", size = 1, alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "red", size = 0.5, formula = y ~ x) +
  geom_text(aes(x = 80, y = 220, label = paste("Spearman's ρ =", round(cor, 4))),
            size = 3.5) +
  ggtitle("Weight vs. Height with Spearman's Correlation") +
  xlab("Weight [kg]") + ylab("Height [cm]") +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

print(scatter_plot)

# Save scatter plot and correlation matrix
ggsave(sprintf("Suni_hgt_wt_train_split_%d.jpg", m), width=10, height=10)
write.csv(correlation, file = sprintf("Suni_Cor_WT_HGT_split_%d.csv", m), row.names = FALSE)


# Calculate means and sds to check if conditions have been fulfilled
means <- as.array(c(Train_mean_values$mean_wt, Train_mean_values$mean_hgt))
sds <- as.array(c(Train_mean_values$sd_wt, Train_mean_values$sd_hgt))

validate_logN_corrMatrix(means, sds, correlation)

# Sample weight and height from bivariate lognormal distribution
set.seed(1234)
samples <-mvlogn(1000, mu=means, sd= sds, corrMatrix= correlation)

# Check if samples follow the original means and standard deviations
apply(samples$samples,2,mean)
apply(samples$samples,2,sd)

# Save variables to be used later in augmentation
weight<- samples[["samples"]][, "Weight"]
height<-samples[["samples"]][, "Height"]

# check if scenario fulfills conditions
validate_logN_corrMatrix(means, sds, correlation)


# Section 3: Data Augmentation -------------------------------------------------

# Template for PK --------------------------------------------------------------

# Set the seed for reproducibility
set.seed(42)

# Define the number of subjects
num_subjects <- 1000

# Dosing records
data <- tibble(ID = 1:num_subjects) %>%
  mutate(
    C=0, # NONMEM should accept all simulated patients
    SET_1=0, # always train
    SET_2=0, 
    SET_3=0, 
    SET_4=0, 
    SET_5=0,
    SET_6=0,
    SET_7=0,
    SET_8=0,
    SET_9=0,
    SET_10=0,
    ETID= "DUM",
    DAT= as.Date("2023-09-11"),
    CTIME = "15:00:00",
    TRTW= 1,
    TIME = 0,
    TAD = 0,
    AMT = 50000, # standard starting dose for colorectal and renal cancer (according to labelling)
    DOS= 50,
    DV=0,
    LNDV=0,
    EVID = 1,
    MDV = 1,
    CMT=1,
    TRTM= 1,
    F_LAG=1,
    LOQS= 0.05603,
    LOQSM=0.05683,
    LOQP=0.1086,
    STUDY=3,
    CENTER=10,
    AGE = pmin(pmax(round(rnorm(num_subjects, mean = Train_mean_values$mean_age, sd = Train_mean_values$sd_age)), min(Train_values$AGE)), max(Train_values$AGE)),
    SEX = rbinom(num_subjects, size = 1, prob = percentage_men),
    WEIGHT= round(weight,2),
    HEIGHT= round(height,2),
  )

data_2 <- data %>%
  mutate(
    EVID=0,
    CMT = 2,
    TIME=24,
    MDV=0,
    DAT= as.Date("2023-09-12"),
    TAD = pmin(pmax(round(rnorm(num_subjects, mean = Train_mean_values$mean_tad, sd = Train_mean_values$sd_tad), 2), 0.04), max(individual_tad$TAD)),
    AMT=0,
    AGE = data$AGE,
    SEX = data$SEX,
    WEIGHT = data$WEIGHT,
    HEIGHT = data$HEIGHT
  )

data_3 <- data_2 %>%
  mutate(
    CMT = 3,
    TIME=24,
    TAD = data_2$TAD,
  )


# Combine intakes with observations
combined_data <- bind_rows(data, data_2, data_3)

# Sort the data frame by ID and TAD, add 200 to synthetic patients
combined_data <- combined_data %>%
  arrange(ID, TAD)
combined_data$ID <- combined_data$ID + 200

# Relocate ID to its correct position
combined_data <- combined_data %>%
  relocate(ID, .after = SET_10)

# Save template for data augmentation to a CSV file
write.csv(combined_data, file = sprintf("Suni_Data_Augmentation_template_split_%d.csv",m), row.names = FALSE)


# PK parameters and DV means  --------------------------------------------------

#Take mean DV from simulation

data<- read.nm.tables("Suni_aug_split_1.tab")
data_2<-read.csv("Suni_Data_Augmentation_template_split_1.csv")


# Filter rows where MDV is 0
data <- data %>%
  filter(MDV == 0)

# Get the individual pharmacokinetic parameters (first values)
first_values <- data %>%
  group_by(ID) %>%
  slice(1) %>%
  select(V2, QH, CLP, CLM, V3, Q34, V4, FM, Q25, V5)

# Calculate the mean of 'DV' for positions 1000, 2000, 3000, etc.
num_positions <- 2000
num_individuals <- 1000

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values
id_values <- numeric(0)    # Initialize an empty vector to store the ID values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$LNDV[position_indices])
  mean_values <- c(mean_values, mean_value)
  id_value <- data$ID[position_indices[1]]  # Get the ID from the first row in each group
  id_values <- c(id_values, id_value)
}

# Create a data frame with the mean values and ID values
result_df <- data.frame(Position = seq(1, num_positions), ID = id_values, Mean_DV = mean_values)
result_df$Mean_DV <- round(exp(result_df$Mean_DV), digits = 3)

# Remove the 'Position' column from result_df
result_df <- result_df %>%
  select(-Position)

# Save the result data frame to a CSV file
write.csv(result_df, file = sprintf("mean_ipred_values_augmentation_split_%d.csv", m), row.names = FALSE)

# Columns to identify duplicates
duplicate_check_columns <- c("ID", "DV", "MDV")

# left join mean DV and individual parameters
data_3 <- data_2 %>%
  left_join(result_df, by = "ID") %>%
  mutate(
    DV = ifelse(MDV == 0, Mean_DV, DV)
  ) %>%
  select(-Mean_DV) %>%
left_join(first_values, by = "ID") 

# Remove duplicates based on specified columns, set CMT correctly
data_3 <- data_3 %>%
  distinct(ID, DV, MDV, .keep_all = TRUE) %>%
  mutate(CMT = (row_number() - 1) %% 3 + 1)%>%
  mutate(
    LNDV = round(log(DV), digits=3),
    LNDV = ifelse(is.infinite(LNDV), ".", LNDV)
  )

# Relocate DV and MDV to their correct position, rename columns
data_3 <- data_3 %>%
  rename("#ID" = ID) %>%
  rename("V2X" = V2) %>%
  rename("QHX" = QH) %>%
  rename("CLPX" = CLP) %>%
  rename("CLMX" = CLM) %>%
  rename("V3X" = V3) %>%
  rename("Q34X" = Q34) %>%
  rename("V4X" = V4) %>%
  rename("FMX" = FM) %>%
  rename("Q25X" = Q25) %>%
  rename("V5X" = V5) %>%
  rename("DVDR" = DV) %>%
  relocate(DVDR, .after = DOS) %>%
  relocate(MDV, .after = LNDV)

# Save the PK parameter data frame to a CSV file
write.csv(data_3, file = sprintf("Suni_PK_values_split_%d.csv", m), row.names = FALSE)

# PD templates   ---------------------------------------------------------------
data2<-read.csv("Suni_PK_values_split_1.csv")
data2 <- data2 %>%
  filter(CMT != 3)%>%
  mutate(CMT = ifelse(CMT == 2, 6, CMT))%>%
  mutate(DV = 0)%>%
  select(-DVDR, -LNDV, -LOQS, -LOQP, -LOQSM)

write.csv(data2, file = sprintf("Suni_PD_template_sim_%d.csv", m), row.names = FALSE)

# PD estimation dataset containing sVEGFR2 values-------------------------------

data_ML <- read.csv("Datensatz_AF_FK_Sunitinib_final_cc.csv") 
filtered_data_ML <- data_ML %>%
  filter(PATIENT_ID != "ET0800") %>%
  filter(!is.na(DV), DV != "BQL")
  

# NM PD dataset containing all dosing records
data_suni <- read.csv("Suni_PD_final_2.csv")

data_suni <- select(data_suni, -LOQS, -LOQSM, -LOQP, -STUDY, -CENTER, -F_LAG, -TRTM, -LNDV)

data_suni<-data_suni %>%
  filter(C != 1)

sVEGFR2 <- filtered_data_ML %>%
  pull(sVEGFR2) 

svegfr2 <- which(data_suni$CMT == 5)
data_suni$DV[svegfr2] <- sVEGFR2

data_suni <- data_suni %>%
  mutate(EVID = ifelse(CMT == 5, 0, EVID),
         MDV = ifelse(CMT == 5, 0, MDV))

# dataset with individual PK parameters
data_pk<- read.nm.tables("Suni_focei_params_split_1.tab")
data_pk <- select(data_pk, -IPRED, -EXPIPRED, -CWRES, -IWRES, -IRES, -LNDV.1, -PRED, -RES, -WRES) # remove unnecessary columns

# Select the first PK parameters per patient ID
first_pk <- data_pk %>%
  group_by(ID) %>%
  slice(1) %>%
  select(ID, V2, QH, CLP, CLM, V3, Q34, V4, FM, Q25, V5)  

data_suni_1<-data_suni %>%
  filter(SET_1 == 0)

data_suni_1 <- left_join(data_suni_1, first_pk, by = "ID")

# Save the PD template data frame to a CSV file
write.csv(data_suni_1, file = sprintf("Suni_PD_estimation_%d.csv", m), row.names = FALSE)

# PD estimation dataset containing sVEGFR3 values-------------------------------

data_ML <- read.csv("Datensatz_AF_FK_Sunitinib_final_cc.csv") 
filtered_data_ML <- data_ML %>%
  filter(PATIENT_ID != "ET0800") %>%
  filter(!is.na(DV), DV != "BQL")


# NM PD dataset containing all dosing records
data_suni <- read.csv("Suni_PD_final_3.csv")

data_suni <- select(data_suni, -LOQS, -LOQSM, -LOQP, -STUDY, -CENTER, -F_LAG, -TRTM, -LNDV)

data_suni<-data_suni %>%
  filter(C != 1)

sVEGFR3 <- filtered_data_ML %>%
  pull(sVEGFR3) 

svegfr3 <- which(data_suni$CMT == 6)
data_suni$DV[svegfr3] <- sVEGFR3

data_suni <- data_suni %>%
  mutate(EVID = ifelse(CMT == 6, 0, EVID),
         MDV = ifelse(CMT == 6, 0, MDV))

# dataset with individual PK parameters
data_pk<- read.nm.tables("Suni_focei_params_split_1.tab")
data_pk <- select(data_pk, -IPRED, -EXPIPRED, -CWRES, -IWRES, -IRES, -LNDV.1, -PRED, -RES, -WRES) # remove unnecessary columns

# Select the first PK parameters per patient ID
first_pk <- data_pk %>%
  group_by(ID) %>%
  slice(1) %>%
  select(ID, V2, QH, CLP, CLM, V3, Q34, V4, FM, Q25, V5)  

data_suni_2<-data_suni %>%
  filter(SET_1 == 0)

data_suni_2 <- left_join(data_suni_2, first_pk, by = "ID")

# Save the PD template data frame to a CSV file
write.csv(data_suni_2, file = sprintf("Suni_PD_estimation_3_%d.csv", m), row.names = FALSE)


# PD estimation values for ML data augmentation --------------------------------

data<- read.nm.tables("Suni_PD_2_augmentation.tab")
datab<-read.nm.tables("Suni_PD_3_augmentation.tab")
data_2<-read.csv("Suni_PK_values_split_1.csv")

# svEGFR2
# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'DV' for positions 1000, 2000, 3000, etc.
num_positions <- 1000
num_individuals <- 1000

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values
id_values <- numeric(0)    # Initialize an empty vector to store the ID values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$DV[position_indices])
  mean_values <- c(mean_values, mean_value)
  id_value <- data$ID[position_indices[1]]  # Get the ID from the first row in each group
  id_values <- c(id_values, id_value)
}

# Create a data frame with the mean values and ID values
result_df <- data.frame(Position = seq(1, num_positions), ID = id_values, Mean_DV = mean_values)
result_df$Mean_DV <- round(result_df$Mean_DV, digits = 3)

# Remove the 'Position' column from result_df
result_df <- result_df %>%
  select(-Position)

# Save the result data frame to a CSV file
write.csv(result_df, file = sprintf("mean_ipred_values_augmentation_pd_split_%d.csv", m), row.names = FALSE)

# svEGFR3
# Filter rows where MDV is equal to 0
datab <- datab %>%
  filter(MDV == 0)

# Calculate the mean of 'DV' for positions 1000, 2000, 3000, etc.
num_positions <- 1000
num_individuals <- 1000

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values
id_values <- numeric(0)    # Initialize an empty vector to store the ID values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(datab$DV[position_indices])
  mean_values <- c(mean_values, mean_value)
  id_value <- datab$ID[position_indices[1]]  # Get the ID from the first row in each group
  id_values <- c(id_values, id_value)
}

# Create a data frame with the mean values and ID values
result_df <- data.frame(Position = seq(1, num_positions), ID = id_values, Mean_DV = mean_values)
result_df$Mean_DV <- round(result_df$Mean_DV, digits = 3)

# Remove the 'Position' column from result_df
result_df <- result_df %>%
  select(-Position)

# Save the result data frame to a CSV file
write.csv(result_df, file = sprintf("mean_ipred_values_augmentation_split_pd2_%d.csv", m), row.names = FALSE)

# Create augmentation dataset --------------------------------------------------

mean_ipred_values_augmentation_split_pd_3_1<-read.csv("mean_ipred_values_augmentation_split_pd_3_1.csv") #svegfr3 data
mean_ipred_values_augmentation_split_pd_2_1<-read.csv("mean_ipred_values_augmentation_split_pd_2_1.csv") #svegfr2 data
mean_ipred_values_augmentation_split_1<-read.csv("mean_ipred_values_augmentation_split_1.csv") #dv and dv_met data (PK)
Suni_PK_values_split_1<-read.csv("Suni_PK_values_split_1.csv") #demographics data

# Create your new dataset

new_data <- data.frame(
  PATIENT_ID = rep("Dummy", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_1 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_2 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_3 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_4 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_5 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_6 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_7 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_8 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_9 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  Set_10 = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ID = mean_ipred_values_augmentation_split_pd_3_1$ID,
  STUDY = rep(3, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  CYC = rep(1, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  DOS = rep(50, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  DAT_MEAS = rep("11.09.2023", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  TIME_MEAS = rep("15:00", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  DAT_INTAKE = rep("11.09.2023", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  TIME_INTAKE = rep("14:00", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  TAD = round(Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(TAD = nth(TAD, 2)) %>% pull(TAD), digits = 2),
  TSB = round(Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(TAD = nth(TAD, 2)) %>% pull(TAD), digits = 2),
  DV = mean_ipred_values_augmentation_split_1 %>% group_by(ID) %>% summarise(Mean_DV = first(Mean_DV))%>% pull(Mean_DV),
  DV_MET = mean_ipred_values_augmentation_split_1 %>% group_by(ID) %>% summarise(Mean_DV = nth(Mean_DV, 2))%>% pull(Mean_DV),
  sVEGFR2 = mean_ipred_values_augmentation_split_pd_2_1$Mean_DV,
  sVEGFR3 = mean_ipred_values_augmentation_split_pd_3_1$Mean_DV,
  SEX = Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(SEX = first(SEX))%>% pull(SEX),  
  AGE = Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(AGE = first(AGE))%>% pull(AGE),  
  SMOKE = rep(0, nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  WT = Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(WT = first(WEIGHT))%>% pull(WT),
  HGT = Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(HGT = first(HEIGHT))%>% pull(HGT),  
  BSA = round(Suni_PK_values_split_1 %>% group_by(ID) %>% summarise(BSA = first(sqrt(HEIGHT * WEIGHT / 3600)))%>% pull(BSA), digits=2),
  DAT_BP = rep("11.09.2023", nrow(mean_ipred_values_augmentation_split_pd_3_1)), # dummy
  TIME_BP = rep("15:00", nrow(mean_ipred_values_augmentation_split_pd_3_1)), # dummy
  BP_SYS = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  BP_DIA = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  PULS = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ALT = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  AST = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ALB = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ALKP = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  CAL = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  SCREA = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  LDH = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  BILI = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  CYP1 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ABCR1 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ABCR2 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  ABCR3 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  VEGFA1 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  VEGFA2 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  VEGFA3 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  VEGFA4 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  KDR1 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  FLT1 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  FLT2 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  FLT3 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  IL8 = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  HAPAB = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)),
  HAPVG = rep("NA", nrow(mean_ipred_values_augmentation_split_pd_3_1)))


write.csv(new_data, file = sprintf("augmented_dataset_split_%d.csv", m), row.names = FALSE)

