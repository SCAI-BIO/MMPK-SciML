#--------------------------------------------------#
# Baseline Demographics and Scaling                #
#--------------------------------------------------#

library(dplyr)
setwd("C:/Users/teply/Documents/Baseline_scaling_5fu_suni")

# 5FU --------------------------------------------------------------------------

data_fu <- read.csv("corrected_10fold_5fu_clean_check.csv")

# Select the first value per patient (baseline)
first_values <- data_fu %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Calculate median baseline demographics and the range -------------------------
variables_of_interest <- first_values %>%
  group_by(ID) %>%
  select(Age, AUC_new, BSA_new, AMT, Cycle)  

# Calculate the median, min and max 
median <- apply(variables_of_interest, 2, median)
min <- apply(variables_of_interest, 2, min)
max <- apply(variables_of_interest, 2, max)

statistics_df <- data.frame(
  median = median,
  min = min,
  max = max
)

# Write to CSV
write.csv(statistics_df, file = "baseline_statistics_5fu.csv", row.names = TRUE)

# Count the number of women and men --------------------------------------------
women_count <- data_fu %>%
  filter(Sex == 0) %>%
  distinct(ID) %>%
  nrow()

men_count <- data_fu %>%
  filter(Sex == 1) %>%
  distinct(ID) %>%
  nrow()

women_percentage <- (women_count/(women_count+men_count))*100
men_percentage <- 100 - women_percentage

# Therapy regimens at baseline -------------------------------------------------

# aio =  Weekly 5FU infusion (2600 mg/m2) over 24 h in combination with folinate (500 mg/m2).
aio_count <- first_values %>%
  filter(Therapy == 5) %>%
  distinct(ID) %>%
  nrow()

# fufox = Weekly 5FU infusion (2000 mg/m2) over 24 h in combination with folinate (500 mg/m2) and oxaliplatin (50 mg/m2).
# including monoclonal antibodies
fufox_count <- first_values %>%
  filter(Therapy %in% c(3, 9, 15)) %>%
  distinct(ID) %>%
  nrow()


# Paclitaxel/cisplatin/5FU/folinate
pacli_count <- first_values %>%
  filter(Therapy == 18) %>%
  distinct(ID) %>%
  nrow()

# Other therapies 
other_therapy <- (women_count + men_count)-(aio_count + fufox_count + pacli_count)


# Indications at baseline ------------------------------------------------------

# colorectal cancer: cancer from the colon or rectum (not includng adenocarcinoma)
colorectal <- first_values %>%
  filter(Indication %in% c(5, 6, 14)) %>%
  distinct(ID) %>%
  nrow()
  
# gastroesophagal: cancers of the esophagus, gastroesophageal junction (GEJ), and stomach
gastroesophagal <- first_values %>%
  filter(Indication %in% c(7, 11, 15)) %>%
  distinct(ID) %>%
  nrow()

# pancreatic
pancreatic <- data_fu %>%
  filter(Indication == 13) %>%
  distinct(ID) %>%
  nrow()

# other
other_indication <- (women_count + men_count)-(colorectal + gastroesophagal + pancreatic)

# Create a dataframe for demographics
demographics_df <- data.frame(
  metric = c("Women Count", "Men Count", "Women Percentage", "Men Percentage",
             "AIO Count", "FIFOX Count", "PACLI Count", "Other Therapy Count",
             "Colorectal Indication Count", "Gastroesophagal Indication Count",
             "Pancreatic Indication Count", "Other Indication Count"),
  value = c(women_count, men_count, women_percentage, men_percentage,
            aio_count, fufox_count, pacli_count, other_therapy,
            colorectal, gastroesophagal, pancreatic, other_indication)
)

# Write to CSV
write.csv(demographics_df, file = "baseline_counts_5fu.csv", row.names = FALSE)


# BSA median calculation for each training dataset -----------------------------
# (body surface area)

# Calculate the median BSA value per ID
median_bsa_per_id <- data_fu %>%
  group_by(ID) %>%
  summarise(median_bsa = median(BSA_new, na.rm = TRUE))

# Initialize a list to store the results
results <- list()

# Loop through Set_Run1 to Set_Run10
for (i in 1:10) {
  set_run_column <- paste0("Set_Run", i)
  
  # Step 4: Filter the data for Set_Run == 0
  set_run_zero_ids <- data_fu %>%
    filter(!!sym(set_run_column) == 0) %>%
    select(ID) %>%
    distinct()
  
  # Calculate the median of these median BSA values
  median_of_medians <- median_bsa_per_id %>%
    filter(ID %in% set_run_zero_ids$ID) %>%
    summarise(median_of_median_bsa = median(median_bsa, na.rm = TRUE))
  
  # Store the result in the list
  results[[set_run_column]] <- median_of_medians$median_of_median_bsa
}


# Save the data frame to a CSV file
write.csv(results, "median_bsa_5fu_results.csv", row.names = FALSE)

# Sunitinib --------------------------------------------------------------------

# See publication 
# Population Modeling Integrating Pharmacokinetics, Pharmacodynamics, Pharmacogenetics, 
# and Clinical Outcome in Patients With Sunitinib-Treated Cancer
# MH Diekstra, A Fritsch et al.

# (demographics unchanged)

# WT and HGT mean calculation per sex for each training dataset ----------------
# (weight and height)

data_suni <- read.csv("Datensatz_AF_FK_Sunitinib_final_raw.csv")

# Remove entries with DV == "BQL", DV == NA, and CYC == "BASELINE"
filtered_data <- data_suni %>%
  filter(!(DV == "BQL" | is.na(DV)) & CYC != "BASELINE")

# Calculate the mean WT and HGT by SEX, grouping by ID
mean_by_sex <- filtered_data %>%
  group_by(ID, SEX) %>%
  summarise(mean_WT = mean(WT, na.rm = TRUE),
            mean_HGT = mean(HGT, na.rm = TRUE))

print(mean_by_sex)

# Initialize a list to store the results
results_list <- list()

# Loop through Set_1 to Set_10
for (i in 1:10) {
  set_column <- paste0("Set_", i)
  
  # Filter the data for Set_ == 0 (train data), grouping by SEX
  result <- mean_by_sex %>%
    inner_join(filtered_data %>% select(ID, SEX, !!sym(set_column)), by = c("ID", "SEX")) %>%
    filter(!!sym(set_column) == 0) %>%
    group_by(SEX) %>%
    summarise(mean_WT = mean(mean_WT, na.rm = TRUE),
              mean_HGT = mean(mean_HGT, na.rm = TRUE))
  
  # Round the results to two decimal places
  result <- result %>%
    mutate(mean_WT = round(mean_WT, 2),
           mean_HGT = round(mean_HGT, 2))
  
  # Store the result in the list with the set column name as key
  results_list[[set_column]] <- result
}

# Convert the list of results to a dataframe for easier saving
results_df <- do.call(rbind, lapply(names(results_list), function(x) {
  df <- results_list[[x]]
  df$Set_Column <- x
  return(df)
}))

write.csv(results_df, "mean_wt_hgt_by_sex_sunitinib_results.csv", row.names = FALSE)