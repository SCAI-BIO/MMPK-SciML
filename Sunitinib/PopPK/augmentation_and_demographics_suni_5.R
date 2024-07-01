# Suni Augmentation ------------------------------------------------------------
#--------------------------------------------------#
# Data Augmentation Sunitinib                      #
#--------------------------------------------------#
#--------------------------------------------------#

# Split 5

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

setwd("C:/Users/teply/Documents/Suni_NONMEM_final/Augmentation/Split5")

# Split number m
m <-5


# Section 1: Demographics of Train Data for Augmentation -----------------------

data_suni <- read.csv("Suni_PK_final_raw.csv")
data_suni <- subset(data_suni, C != 1) # no BLQ and missing DV

data_suni[data_suni == -99] <- NA


# Define patient IDs for train and test sets
Test_ids <- unique(data_suni$ID[data_suni$SET_5 == 1])
Train_ids <- unique(data_suni$ID[data_suni$SET_5 == 0])


# Calculate individual mean values 
individual_mean_values <- aggregate(cbind(AGE, WEIGHT, HEIGHT, SEX, TAD) ~ ID, data = data_suni, FUN = mean, na.rm = TRUE)
                                    

# Filter data for Train and calculate mean values
Train_mean_values <- individual_mean_values %>%
  filter(ID %in% Train_ids) %>%
  summarize(
    mean_age = mean(AGE),
    sd_age = sd(AGE),
    mean_wt = mean(WEIGHT),
    sd_wt = sd(WEIGHT),
    mean_hgt = mean(HEIGHT),
    sd_hgt = sd(HEIGHT),
    mean_tad = mean(TAD),
    sd_tad = sd(TAD))

# Filter data for Men and calculate mean height and weight
Men_hgt_wt <- individual_mean_values %>%
  filter(ID %in% Train_ids) %>%
  filter(SEX == 1) %>%
  summarize(
    mean_wt = mean(WEIGHT, na.rm = TRUE),
    sd_wt = sd(WEIGHT, na.rm = TRUE),
    mean_hgt = mean(HEIGHT, na.rm = TRUE),
    sd_hgt = sd(HEIGHT, na.rm = TRUE))

# Filter data for Women and calculate mean height and weight
Women_hgt_wt <- individual_mean_values %>%
  filter(ID %in% Train_ids) %>%
  filter(SEX == 0) %>%
  summarize(
    mean_wt = mean(WEIGHT, na.rm = TRUE),
    sd_wt = sd(WEIGHT, na.rm = TRUE),
    mean_hgt = mean(HEIGHT, na.rm = TRUE),
    sd_hgt = sd(HEIGHT, na.rm = TRUE))


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
write.csv(range_train, file = sprintf("range_train_demo_suni_%d.csv", m), row.names = FALSE)


# Section 2: Weight - Height Correlation ---------------------------------------

# Train men---------------------------------------------------------------------
data_suni_demo <- individual_mean_values%>%
  filter(ID %in% Train_ids) # only one mean value per patient

# Filter for male patients (SEX == 1)
Men <- data_suni_demo %>%
  filter(SEX == 1)

WT<- Men$WEIGHT
HGT<- Men$HEIGHT

# Transform weight and height using the natural logarithm
log_WT <- log(WT)
log_HGT <- log(HGT)

# Create dataframes with log-transformed Weight and Height and original Weight and Height
log_df <- data.frame(LogWeight = log_WT, LogHeight = log_HGT)
df <- data.frame(Weight = WT, Height = HGT)


# Calculate the Pearson correlation coefficient for log-transformed data 
# linear correlation of log-normal values assumed, outliers not detected

log_correlation <- cor(log_df, method = "pearson")
log_cor <- log_correlation[1, 2]

# Calculate the coefficients of variation for the original data
cv_WT <- sd(WT) / mean(WT)
cv_HGT <- sd(HGT) / mean(HGT)

# Calculate the correlation coefficient on the original scale
rho_original <- (exp(log_cor * sqrt(log(1 + cv_WT^2)) * sqrt(log(1 + cv_HGT^2))) - 1) /
  sqrt((exp(log(1 + cv_WT^2)) - 1) * (exp(log(1 + cv_HGT^2)) - 1))

# Print the correlation coefficient on the original scale
print(rho_original)

# Create a scatter plot
scatter_plot <- ggplot(data = df, aes(x = WT, y = HGT)) +
  geom_point(color = "blue", size = 1, alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "red", size = 0.5, formula = y ~ x) +
  annotate("text", x = 80, y = 220, label = paste("Pearson's ρ =", round(rho_original, 4)), size = 3.5) +
  ggtitle("Weight vs. Height with Pearson's Correlation Men Suni") +
  xlab("Weight [kg]") + ylab("Height [cm]") +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

print(scatter_plot)

ggsave(sprintf("hgt_wt_train_suni_men_%d.jpg", m), width=10, height=10)

# Create the correlation matrix
correlation_matrix <- matrix(c(1, rho_original, rho_original, 1), nrow = 2, ncol = 2)
colnames(correlation_matrix) <- c("Weight", "Height")
rownames(correlation_matrix) <- c("Weight", "Height")

# Validate simulation
means <- as.array(c(Men_hgt_wt$mean_wt, Men_hgt_wt$mean_hgt))
sds <- as.array(c(Men_hgt_wt$sd_wt, Men_hgt_wt$sd_hgt))
validate_logN_corrMatrix(means, sds, correlation_matrix)

# sample
set.seed(1234)
samples <-mvlogn(1000, mu=means, sd= sds, corrMatrix= correlation_matrix)

# check samples with respect to original mean, sd, cv
apply(samples$samples,2,mean)
apply(samples$samples,2,sd)
apply(samples$samples,2,sd)/apply(samples$samples,2,mean)

weight_men<- samples[["samples"]][, "Weight"]
height_men <-samples[["samples"]][, "Height"]


# Train women-------------------------------------------------------------------

data_suni_demo <- individual_mean_values%>%
  filter(ID %in% Train_ids) # only one mean value per patient

# Filter for female patients (SEX == 0)
Women <- data_suni_demo %>%
  filter(SEX == 0)

WT<- Women$WEIGHT
HGT<- Women$HEIGHT

# Transform weight and height using the natural logarithm
log_WT <- log(WT)
log_HGT <- log(HGT)

# Create dataframes with log-transformed Weight and Height and original Weight and Height
log_df <- data.frame(LogWeight = log_WT, LogHeight = log_HGT)
df <- data.frame(Weight = WT, Height = HGT)


# Calculate the Pearson correlation coefficient for log-transformed data 
# linear correlation of log-normal values assumed, outliers not detected

log_correlation <- cor(log_df, method = "pearson")
log_cor <- log_correlation[1, 2]

# Calculate the coefficients of variation for the original data
cv_WT <- sd(WT) / mean(WT)
cv_HGT <- sd(HGT) / mean(HGT)

# Calculate the correlation coefficient on the original scale
rho_original <- (exp(log_cor * sqrt(log(1 + cv_WT^2)) * sqrt(log(1 + cv_HGT^2))) - 1) /
  sqrt((exp(log(1 + cv_WT^2)) - 1) * (exp(log(1 + cv_HGT^2)) - 1))

# Print the correlation coefficient on the original scale
print(rho_original)

# Create a scatter plot
scatter_plot <- ggplot(data = df, aes(x = WT, y = HGT)) +
  geom_point(color = "blue", size = 1, alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "red", size = 0.5, formula = y ~ x) +
  annotate("text", x = 80, y = 220, label = paste("Pearson's ρ =", round(rho_original, 4)), size = 3.5) +
  ggtitle("Weight vs. Height with Pearson's Correlation Women Suni") +
  xlab("Weight [kg]") + ylab("Height [cm]") +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

print(scatter_plot)

ggsave(sprintf("hgt_wt_train_suni_Women_%d.jpg", m), width=10, height=10)

# Create the correlation matrix
correlation_matrix <- matrix(c(1, rho_original, rho_original, 1), nrow = 2, ncol = 2)
colnames(correlation_matrix) <- c("Weight", "Height")
rownames(correlation_matrix) <- c("Weight", "Height")

# Validate simulation
means <- as.array(c(Women_hgt_wt$mean_wt, Women_hgt_wt$mean_hgt))
sds <- as.array(c(Women_hgt_wt$sd_wt, Women_hgt_wt$sd_hgt))
validate_logN_corrMatrix(means, sds, correlation_matrix)

# sample
set.seed(1234)
samples <-mvlogn(1000, mu=means, sd= sds, corrMatrix= correlation_matrix)

# check samples with respect to original mean, sd, cv
apply(samples$samples,2,mean)
apply(samples$samples,2,sd)
apply(samples$samples,2,sd)/apply(samples$samples,2,mean)

weight_women<- samples[["samples"]][, "Weight"]
height_women <-samples[["samples"]][, "Height"]


# Section 3: Data Augmentation Template ----------------------------------------

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
    DAT= "DUM",
    CTIME = "DUM",
    TRTM= 1, # first treatment month
    TIME = sample(0:28, num_subjects, replace = TRUE) * 24, # 4 weeks-on of first treatment month
    TAD = 0,
    AMT = 50000, # common starting dose for mCRC and mRCC according to labelling
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
    WEIGHT=ifelse(SEX == 1, round((weight_men), 2), round((weight_women), 2)),
    HEIGHT=ifelse(SEX == 1, round((height_men), 2), round((height_women), 2))
  )

data_2 <- data %>%
  mutate(
    EVID=0,
    CMT = 2,
    TAD = pmin(pmax(round(rnorm(num_subjects, mean = Train_mean_values$mean_tad, sd = Train_mean_values$sd_tad), 2), min(Train_values$TAD)), max(Train_values$TAD)),
    TIME=TIME+TAD,
    MDV=0,
    DAT= "DUM",
    AMT=0,
    AGE = data$AGE,
    SEX = data$SEX,
    WEIGHT = data$WEIGHT,
    HEIGHT = data$HEIGHT,
  )

data_3 <- data_2 %>%
  mutate(
    CMT = 3,
    TIME=data_2$TIME,
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


# Section 4: EXPIPRED means  ---------------------------------------------------------

#Take mean EXPIPRED from simulation (without residual variability)

data<- read.nm.tables("Suni_aug_split_5.tab")
data_2<-read.csv("Suni_Data_Augmentation_template_split_5.csv")


# Filter rows where CMT is not 1 (no dosing records)
data <- data %>%
  filter(CMT != 1)

# Filter rows where AMT is 0 (no dosing records)
data_2 <- data_2 %>%
  filter(AMT == 0)


# Calculate the mean of 'DV' for positions 1000, 2000, 3000, etc.
num_positions <- 2000

mean_values <- numeric(0)  # Initialize an empty vector to store the mean values
id_values <- numeric(0)    # Initialize an empty vector to store the ID values

for (i in 1:num_positions) {
  position_indices <- seq(i, length.out = 1000, by = num_positions)
  mean_value <- mean(data$EXPIPRED[position_indices])
  mean_values <- c(mean_values, mean_value)
  id_value <- data$ID[position_indices[1]]  # Get the ID from the first row in each group
  id_values <- c(id_values, id_value)
}

# Create a data frame with the mean values and ID values
result_df <- data.frame(Position = seq(1, num_positions), ID = id_values, Mean_EXPIPRED = mean_values)
result_df <- result_df %>%
  mutate(Mean_EXPIPRED = round(Mean_EXPIPRED, digits = 3),
         Mean_EXPIPRED = ifelse(Mean_EXPIPRED < 0.06, 0.06, Mean_EXPIPRED)) # truncate to BLQ

# Remove the 'Position' column from result_df
result_df <- result_df %>%
  select(-Position)

# Save the result data frame to a CSV file
write.csv(result_df, file = sprintf("mean_ipred_values_augmentation_split_%d.csv", m), row.names = FALSE)

# Columns to identify duplicates
duplicate_check_columns <- c("ID", "DV", "MDV")

data_3 <- data_2 %>%
  left_join(result_df, by = "ID") %>%
  mutate(DV = Mean_EXPIPRED) %>%
  select(-Mean_EXPIPRED)

# Remove duplicates based on specified columns, set CMT correctly
data_3 <- data_3 %>%
  distinct(ID, DV, MDV, .keep_all = TRUE) %>%
  mutate(CMT = (row_number() - 1) %% 3 + 1)%>%
  mutate(
    DV = round(DV, digits=3),
    DV = ifelse(is.infinite(DV), ".", DV)
  )



# Section 5: Create augmentation dataset ---------------------------------------

summarized_data_3 <- data_3 %>%
  group_by(ID) %>%
  summarise(
    TAD = nth(TAD, 2), 
    TIME = nth(TIME, 2), 
    SEX = first(SEX),   
    AGE = first(AGE),
    WEIGHT = first(WEIGHT),
    HEIGHT = first(HEIGHT),
    DV_MET = nth(DV, 2), # Pull second DV value
    DV = first(DV),     # Pull first DV value
  )

# Create final dataset

new_data <- data.frame(
  PATIENT_ID = rep("Dummy", nrow(summarized_data_3)),
  Set_1 = rep(0, nrow(summarized_data_3)),
  Set_2 = rep(0, nrow(summarized_data_3)),
  Set_3 = rep(0, nrow(summarized_data_3)),
  Set_4 = rep(0, nrow(summarized_data_3)),
  Set_5 = rep(0, nrow(summarized_data_3)),
  Set_6 = rep(0, nrow(summarized_data_3)),
  Set_7 = rep(0, nrow(summarized_data_3)),
  Set_8 = rep(0, nrow(summarized_data_3)),
  Set_9 = rep(0, nrow(summarized_data_3)),
  Set_10 = rep(0, nrow(summarized_data_3)),
  ID = summarized_data_3$ID,
  STUDY = rep(3, nrow(summarized_data_3)),
  CYC = rep(1, nrow(summarized_data_3)),
  DOS = rep(50, nrow(summarized_data_3)),
  DAT_MEAS = rep("11.09.2023", nrow(summarized_data_3)),
  TAD = summarized_data_3$TAD,
  TIME = summarized_data_3$TIME,
  DV = summarized_data_3$DV,
  DV_MET = summarized_data_3$DV_MET,
  sVEGFR2 = NA,
  sVEGFR3 = NA,
  SEX = summarized_data_3$SEX,  
  AGE = summarized_data_3$AGE,  
  SMOKE = rep(0, nrow(summarized_data_3)),
  WT = summarized_data_3$WEIGHT,
  HGT = summarized_data_3$HEIGHT, 
  BSA = round(sqrt(summarized_data_3$HEIGHT * summarized_data_3$WEIGHT / 3600), 2), # Calculate BSA using the Mosteller formula
  DAT_BP = rep("11.09.2023", nrow(summarized_data_3)), # dummy
  TIME_BP = rep("15:00", nrow(summarized_data_3)), # dummy
  BP_SYS = rep("NA", nrow(summarized_data_3)),
  BP_DIA = rep("NA", nrow(summarized_data_3)),
  PULS = rep("NA", nrow(summarized_data_3)),
  ALT = rep("NA", nrow(summarized_data_3)),
  AST = rep("NA", nrow(summarized_data_3)),
  ALB = rep("NA", nrow(summarized_data_3)),
  ALKP = rep("NA", nrow(summarized_data_3)),
  CAL = rep("NA", nrow(summarized_data_3)),
  SCREA = rep("NA", nrow(summarized_data_3)),
  LDH = rep("NA", nrow(summarized_data_3)),
  BILI = rep("NA", nrow(summarized_data_3)),
  CYP1 = rep("NA", nrow(summarized_data_3)),
  ABCR1 = rep("NA", nrow(summarized_data_3)),
  ABCR2 = rep("NA", nrow(summarized_data_3)),
  ABCR3 = rep("NA", nrow(summarized_data_3)),
  VEGFA1 = rep("NA", nrow(summarized_data_3)),
  VEGFA2 = rep("NA", nrow(summarized_data_3)),
  VEGFA3 = rep("NA", nrow(summarized_data_3)),
  VEGFA4 = rep("NA", nrow(summarized_data_3)),
  KDR1 = rep("NA", nrow(summarized_data_3)),
  FLT1 = rep("NA", nrow(summarized_data_3)),
  FLT2 = rep("NA", nrow(summarized_data_3)),
  FLT3 = rep("NA", nrow(summarized_data_3)),
  IL8 = rep("NA", nrow(summarized_data_3)),
  HAPAB = rep("NA", nrow(summarized_data_3)),
  HAPVG = rep("NA", nrow(summarized_data_3)))


write.csv(new_data, file = sprintf("augmented_dataset_split_%d.csv", m), row.names = FALSE)

