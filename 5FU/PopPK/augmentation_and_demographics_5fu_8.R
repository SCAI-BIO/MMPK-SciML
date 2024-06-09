#--------------------------------------------------#
#--------------------------------------------------#
# Demographics and Data Augmentation 5-FU                                                                                                                                                                                                                                          #
#--------------------------------------------------#
#--------------------------------------------------#

# Split 8

#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots
#xpose4: needed to create pc-vpc and to read nonmem tables

rm(list=ls())


library(dplyr) 
library(tidyverse)
library(ggplot2)
library(xpose4)
#install.packages("remotes")
#remotes::install_github("AlessandroDeCarlo27/mvlognCorrEst")
#install.packages("Matrix")
#install.packages("igraph")
#install.packages("qgraph")
#install.packages("pracma")
library(mvLognCorrEst)

setwd("C:/Users/Olga/5FU_ML_neu/Augmentation_datasets/Split8")

# Split number m
m <-8


# Section 1: Demographics -------------------------------------------------------

data_5fu_demo <- read.csv("corrected_full_10fold_data_5fu_fi_cyc_split_check.csv")

# Train data -------------------------------------------------------------------

# Define patient IDs for Test set

Test_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 1])
Train_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 0])

# Calculate individual mean values 
individual_mean_values <- aggregate(cbind(Age,BSA_new, AMT, AUC_new, Difference_Start_End_Infusion, HGT, WT) ~ ID, data = data_5fu_demo, FUN = mean, na.rm = TRUE)

# Calculate the standard deviation for each variable by ID, handling NA values
individual_sd_values <- aggregate(. ~ ID, data = individual_mean_values, FUN = function(x) sd(x, na.rm = TRUE))

# Filter data for Train and calculate mean values
Train_mean_values <- individual_mean_values %>%
  filter(ID %in% Train_ids) %>%
  summarize(
    mean_age = mean(Age, na.rm = TRUE),
    sd_age = sd(Age, na.rm = TRUE),
    mean_bsa = mean(BSA_new, na.rm = TRUE),
    sd_bsa = sd(BSA_new, na.rm = TRUE),
    mean_dose = mean(AMT, na.rm = TRUE),
    sd_dose = sd(AMT, na.rm = TRUE),
    mean_auc = mean(AUC_new, na.rm = TRUE),
    sd_auc = sd(AUC_new, na.rm = TRUE),
    mean_time = mean(Difference_Start_End_Infusion, na.rm = TRUE),
    sd_time = sd(Difference_Start_End_Infusion, na.rm = TRUE),
    mean_wt = mean(WT, na.rm = TRUE),
    sd_wt = sd(WT, na.rm = TRUE),
    mean_hgt = mean(HGT, na.rm = TRUE),
    sd_hgt = sd(HGT, na.rm = TRUE))

# summarize range of values for train
Train_values <- individual_mean_values %>%
  filter(ID %in% Train_ids)
summary(Train_values)

# Filter data and count the number of women and men for Train
Train_women_count <- data_5fu_demo %>%
  filter(ID %in% Train_ids, Sex == 0) %>%
  distinct(ID) %>%
  nrow()
percentage_women<- Train_women_count/125
percentage_men<-1-percentage_women

Train_men_count <- data_5fu_demo %>%
  filter(ID %in% Train_ids, Sex == 1) %>%
  distinct(ID) %>%
  nrow()

Train_mean_values$womencount<- Train_women_count
Train_mean_values$mencount<- Train_men_count

range_train<- summary(Train_values)

# Save the result data frame to a CSV file
write.csv(Train_mean_values, file = sprintf("Train_demo_5fu_%d.csv", m), row.names = FALSE)
write.csv(range_train, file = sprintf("range_train_demo_5fu_%d.csv", m), row.names = FALSE)

# Section 2: BSA vs Dose -------------------------------------------------------
data_5fu_demo <- read.csv("corrected_full_10fold_data_5fu_fi_cyc_split_check.csv")
data_5fu_demo <- data_5fu_demo[complete.cases(data_5fu_demo), ]

Test_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 1])
Train_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 0])

# Train ------------------------------------------------------------------------
data_5fu_demo <- data_5fu_demo %>%
  filter(ID %in% Train_ids) 

#lm function
model <- lm(AMT ~ BSA_new, data = data_5fu_demo)
summary(model)

# Extract the coefficients
slope <- coef(model)[["BSA_new"]]
intercept <- coef(model)[["(Intercept)"]]

# Calculate the adjusted R-squared value
adj_r_squared <- summary(model)$adj.r.squared

# Create the equation
equation <- paste("Dose [mg] =", round(intercept, 2), "+", round(slope, 2), "* Body surface area [qm]")

# Create the plot with the equation and adjusted R-squared
bsa_dose <- ggplot(data = data_5fu_demo, aes(x = BSA_new, y = AMT)) +
  geom_point(color = "blue", size = 1, alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "red", size = 0.5, formula = y ~ x) +
  ggtitle("BSA vs. Dose for fluorouracil model \n (train data) [8]") +
  geom_text(aes(x = 1, y = 7700, label = equation), size = 3.5) +
  geom_text(aes(x = 1, y = 8000,
                label = paste("Adjusted R²:", round(adj_r_squared, 4))), size=3.5) +
  theme(plot.title = element_text(size = 12, hjust = 0.5)) +
  scale_x_continuous(limits = c(0, 3)) +
  scale_y_continuous(limits = c(0, 8000)) +
  xlab("Body surface area [qm]") + ylab("Dose [mg]") +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    plot.title = element_text(hjust = 0.5)
  )

# Print the plot and save it
print(bsa_dose)

ggsave(sprintf("bsa_dose_train_5fu_%d.jpg", m), width=10, height=10)


# Section 3: Weight - Height Correlation ---------------------------------------

# Train-------------------------------------------------------------------------
data_5fu_demo <- read.csv("corrected_full_10fold_data_5fu_fi_cyc_split_check.csv")
data_5fu_demo <- data_5fu_demo[complete.cases(data_5fu_demo), ]

Test_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 1])
Train_ids <- unique(data_5fu_demo$ID[data_5fu_demo$Set_Run8 == 0])

data_5fu_demo <- data_5fu_demo %>%
  filter(ID %in% Train_ids) 

WT<- data_5fu_demo$WT
HGT<- data_5fu_demo$HGT
df<- data.frame(Weight = WT, Height = HGT)
correlation <- cor(df, method = "spearman") #not necc. linear, monotonous
cor<-correlation[1, 2] 

# Create a scatter plot
scatter_plot <- ggplot(data = df, aes(x = WT, y = HGT)) +
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

ggsave(sprintf("hgt_wt_train_5fu_%d.jpg", m), width=10, height=10)
write.csv(Train_mean_values, file = sprintf("Cor_WT_HGT_5fu_%d.csv", m), row.names = FALSE)

# Validate simulation
means <- as.array(c(Train_mean_values$mean_wt, Train_mean_values$mean_hgt))
sds <- as.array(c(Train_mean_values$sd_wt, Train_mean_values$sd_hgt))
validate_logN_corrMatrix(means, sds, correlation)

# sample
set.seed(1234)
samples <-mvlogn(1000, mu=means, sd= sds, corrMatrix= correlation)

# check samples with respect to original mean, sd, cv
apply(samples$samples,2,mean)
apply(samples$samples,2,sd)
apply(samples$samples,2,sd)/apply(samples$samples,2,mean)

weight_with<- samples[["samples"]][, "Weight"]
height_with <-samples[["samples"]][, "Height"]

# Section 4 Data Augmentation Template -----------------------------------------

# Train ------------------------------------------------------------------------
# Set the seed for reproducibility
set.seed(42)

# Define the number of subjects
num_subjects <- 1000
# Simulate the dataset using dplyr
data <- tibble(ID = 1:num_subjects) %>%
  mutate(
    TIME = 0,
    REGIME = 24,
    CYC = 1,
    MDV = 1,
    DV=0,
    EVID = 1,
    AGE = pmin(pmax(round(rnorm(num_subjects, mean = Train_mean_values$mean_age, sd = Train_mean_values$sd_age)), min(Train_values$Age)), max(Train_values$Age)),
    WT= round(weight_with,2),
    HGT= round(height_with,2),
    SEX = rbinom(num_subjects, size = 1, prob = percentage_men),
    BSA = round(0.007184 * (HGT^0.725) * (WT^0.425), 2), # Du Bois and Du Bois formula
    AMT = round(intercept + slope *BSA,2), # equation after linear regression
    RATE = round(AMT / REGIME, 2),
    LBM = ifelse(SEX == 1, round((0.32810 * WT + 0.33929 * HGT - 29.5336), 1), round((0.29569 * WT + 0.41813 * HGT - 43.2933), 1)), # Hume formula, sex=1 is male
    FM = round(WT - LBM, 1)
  )

# Add observation records
data_2 <- data %>%
  bind_rows(
    tibble(
      ID = 1:num_subjects,
      AMT = 0,
      TIME = pmin(pmax(round(rnorm(num_subjects, mean = 18, sd = 0.5) * 2) / 2, 17.0), 21.5),
      REGIME = 24,
      RATE = 0,
      CYC = 1,
      MDV = 0,
      DV=0,
      EVID = 0,
      AGE = data$AGE,
      WT = data$WT,
      HGT = data$HGT,
      SEX = data$SEX,
      BSA = data$BSA,
      LBM = data$LBM,
      FM = data$FM
    )
  )

# Sort the data frame by ID and TIME, add + 200 to the IDs of the synthetic patients
data_2 <- data_2 %>%
  arrange(ID, TIME)
data_2$ID <- data_2$ID + 200

# Add # to ID column for NONMEM
colnames(data_2)[colnames(data_2) == "ID"] <- "#ID"

# Save data_2 to a CSV file
write.csv(data_2, file = sprintf("DataAugmentation_dataset_split%d.csv", m), row.names = FALSE)

# Section 5 Data Augmentation Datasets -----------------------------------------

data<- read.nm.tables("Data_Augmentation_NM_split8.tab")
data_2<-read.csv("DataAugmentation_dataset_split8.csv")


# Filter rows where MDV is equal to 0
data <- data %>%
  filter(MDV == 0)

# Calculate the mean of 'DV' for positions 1000, 2000, 3000, etc.
num_positions <- 1000

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
write.csv(result_df, file = sprintf("mean_dv_values_augmentation_split_%d.csv", m), row.names = FALSE)

# Rename X.ID column from data_2
data_2<-data_2 %>% 
  rename(
    ID = X.ID
  )

# Join data_2 and result_df by the 'ID' column
data_3 <- data_2 %>%
  left_join(result_df, by = "ID") %>%
  mutate(
    DV = ifelse(MDV == 0, Mean_DV, DV)
  ) %>%
  select(-Mean_DV)  

#Create dataframe for machine learning augmentation
library(lubridate) 

# In NONMEM, we have one dosing event and a seperate observation event, but we 
# want it combined in one row for ML
# also, some NM items need to be dropped

combined_dataset <- data_3 %>%
  group_by(ID) %>%
  arrange(ID, TIME) %>%
  mutate(
    TIME = last(TIME),  # Set TIME to the value from the last row within each group
    DV = last(DV),      # Set DV to the value from the last row within each group
    EVID = NULL,        # Drop the EVID column
    MDV = NULL,         # Drop the MDV column
    RATE = NULL         # Drop the RATE column
  ) %>%
  filter(AMT != 0) %>%  # Remove rows where AMT is equal to 0
  ungroup()            # Ungroup the data frame

# next, we might need dates and times, this is done with lubridate
combined_dataset <- combined_dataset %>%
  mutate(
    `Date_Infusion_Start` = as.Date("2023-09-11"),    # 11.09.23
    `Time_Infusion_Start` = hms("15:00:00"),        # 15:00:00 as time
    `Date_Infusion_End` = as.Date("2023-09-12"),      # 12.09.23
    `Time_Infusion_End` = hms("15:00:00"),          # 15:00:00 as time
    `Date_Sampling` = `Date_Infusion_Start` + days(floor(TIME / 24)),
    `Time_Sampling` = `Date_Sampling` + hours(15) + minutes((TIME %% 24) * 60),  # Calculate Time_Sampling
    `AUC_new` = round(DV*24,2),  # Calculate AUC (Rectangle Rule: Infusion duration x Steady State concentration)
    `CL_new` = round(AMT/AUC_new,2)  # Calculate AUC (Rectangle Rule: Infusion duration x Steady State concentration)
  )

# next, all times should be in the "%H:%M:%S" format, like 15:00:00
combined_dataset <- combined_dataset %>%
  mutate(
    `Date_Sampling` = as.Date(`Date_Sampling`),
    `Time_Sampling` = format(`Time_Sampling`, format = "%H:%M:%S")
  )

combined_dataset <- combined_dataset %>%
  mutate(
    `Time_Infusion_Start` = sprintf("%02d:%02d:%02d",
                                    hour(`Time_Infusion_Start`), minute(`Time_Infusion_Start`), second(`Time_Infusion_Start`)),
    `Time_Infusion_End` = sprintf("%02d:%02d:%02d",
                                  hour(`Time_Infusion_End`), minute(`Time_Infusion_End`), second(`Time_Infusion_End`))
  ) 

# Add a few columns present in the original data, but which are not required and will be removed in preprocessing
combined_dataset$Age_60 <- 0
combined_dataset$Therapy <- 0
combined_dataset$Indication <- 0
combined_dataset$Leukocytes <- 0
combined_dataset$Neutrophils_Abs <- 0
combined_dataset$Neutrophils <- 0
combined_dataset$Erythrocytes <- 0
combined_dataset$Haemoglobin <- 0
combined_dataset$Haematocrit <- 0
combined_dataset$Thrombocytes <- 0
combined_dataset$Nausea <- 0
combined_dataset$Emesis <- 0
combined_dataset$Erythrocytes <- 0
combined_dataset$Fatigue <- 0
combined_dataset$Diarrhoea <- 0
combined_dataset$Neutropenia_new <- 0
combined_dataset$Thrombocytopenia <- 0
combined_dataset$Hand_Foot <- 0
combined_dataset$Stomatitis <- 0
combined_dataset$Set_Run0 <- 0


# Set DV to same unit as in dataset
combined_dataset$DV <- combined_dataset$DV*1000

# Since the train data is to be augmented, Set_Run1-Set_Run10 are all 0
# Create new columns with all values set to 0
for (i in 1:10) {
  column_name <- paste0("Set_Run", i)
  combined_dataset[[column_name]] <- 0
}

# next, we want to sort the dataframe to what we are used to
combined_dataset <- combined_dataset %>%
  select(
    ID, SEX, AGE, Age_60, Therapy, REGIME,
    `Date_Infusion_Start`, `Time_Infusion_Start`, 
    `Date_Infusion_End`, `Time_Infusion_End`, 
    `Date_Sampling`, `Time_Sampling`, TIME, DV, AMT, CL_new, AUC_new, Indication, 
    Leukocytes,	Neutrophils_Abs,	Neutrophils,	Erythrocytes,	Haemoglobin,	
    Haematocrit,	Thrombocytes, HGT, WT, LBM, FM, BSA, Nausea,	Emesis,	Fatigue,
    Diarrhoea,	Neutropenia_new,	Thrombocytopenia,	Hand_Foot,	Stomatitis, CYC,
    Set_Run1,	Set_Run2,	Set_Run3,	Set_Run4,	Set_Run5,	Set_Run6,	Set_Run7,	Set_Run8,	
    Set_Run9,	Set_Run10, Set_Run0
  )

# finally, we want to rename the columns of the dataframe to what we are used to
combined_dataset <- combined_dataset %>%
  rename(
    `Sex` = SEX,
    `Age` = AGE,
    `Infusion_dur` = REGIME,
    `Difference_Start_End_Infusion` = TIME,
    `LBM_new` = LBM,
    `FM_new` = FM,
    `BSA_new` = BSA,
    `Cycle` = CYC
  )


# Save the result data frame to a CSV file for the ML augmentation
write.csv(combined_dataset, file = sprintf("corrected_10fold_data_5fu_fi_cyc_split_check_augmentation_split%d.csv", m), row.names = FALSE)

