#--------------------------------------------------#
# Baseline Demographics                            #
#--------------------------------------------------#

library(dplyr)

# 5FU --------------------------------------------------------------------------

data_fu <- read.csv("10fold_data_5fu_fi_cyc_split_check.csv")

# Select only complete cases
data_fu <- data_fu[complete.cases(data_fu), ]

# Select the first value per patient (baseline)
first_values <- data_fu %>% 
  group_by(ID) %>%
  slice(1)  # Select the first row for each patient

# Calculate median baseline demographics and the range -------------------------
variables_of_interest <- first_values %>%
  group_by(ID) %>%
  select(Age, AUC_new, BSA_new, AMT, AUC_new, Cycle)  

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

# Therapy regimens -------------------------------------------------------------

# aio =  Weekly 5FU infusion (2600 mg/m2) over 24 h in combination with folinate (500 mg/m2).
aio_count <- data_fu %>%
  filter(Therapy == 5) %>%
  distinct(ID) %>%
  nrow()

# fufox = Weekly 5FU infusion (2000 mg/m2) over 24 h in combination with folinate (500 mg/m2) and oxaliplatin (50 mg/m2).
# including monoclonal antibodies
fufox_count <- data_fu %>%
  filter(Therapy %in% c(3, 9, 15)) %>%
  distinct(ID) %>%
  nrow()


# Paclitaxel/cisplatin/5FU/folinate
pacli_count <- data_fu %>%
  filter(Therapy == 18) %>%
  distinct(ID) %>%
  nrow()

# Other therapies 
other_therapy <- (women_count + men_count)-(aio_count + fufox_count + pacli_count)


# Indications ------------------------------------------------------------------

# colorectal cancer: cancer from the colon or rectum (not includng adenocarcinoma)
colorectal <- data_fu %>%
  filter(Indication %in% c(5, 6, 12, 15)) %>%
  distinct(ID) %>%
  nrow()
  
# gastroesophagal: cancers of the esophagus, gastroesophageal junction (GEJ), and stomach
gastroesophagal <- data_fu %>%
  filter(Indication %in% c(7, 11, 16)) %>%
  distinct(ID) %>%
  nrow()

# pancreatic
pancreatic <- data_fu %>%
  filter(Indication == 14) %>%
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

# Sunitinib --------------------------------------------------------------------

# See publication 
# Population Modeling Integrating Pharmacokinetics, Pharmacodynamics, Pharmacogenetics, 
# and Clinical Outcome in Patients With Sunitinib-Treated Cancer
# MH Diekstra, A Fritsch et al.

# (demographics unchanged)