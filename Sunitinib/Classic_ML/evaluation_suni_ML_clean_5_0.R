# Suni Simple ML evaluation ----------------------------------------------------

#--------------------------------------------------#
# Evaluation of ML Suni models                     #
#--------------------------------------------------#
# Split 5                                          #
#--------------------------------------------------#


#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots

rm(list=ls())
 
library(dplyr)
library(ggplot2)

setwd("C:/Users/teply/Documents/Suni_NONMEM_final/ML_results/Split5")

# Split number m
# Augmentation percentage n, replace with 100 if 100% augmentation
# additionally, set augmentation_100 in all dataset titles

m <- 5
n <- 0

# Section 1: Random Forests-----------------------------------------------------

#suni_results_Random_Forest_split_5_augmentation_0

# No FS ------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Random_Forest_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: RF (test data)") +
  annotate("text", x = 5, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_RF_%d_%d.jpg", m,n), width=6, height=6)


# Section 2: SVM ---------------------------------------------------------------

#suni_results_Support_Vector_Machine_split_5_augmentation_0

# No FS ---------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Support_Vector_Machine_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: SVM  (test data)") +
  annotate("text", x = 5, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_SVM_%d_%d.jpg", m,n), width=6, height=6)

# Section 3: Gradient Boosting -------------------------------------------------

#suni_results_Gradient_Boosting_split_5_augmentation_0

#No FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Gradient_Boosting_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: GB  (test data)") +
  annotate("text", x = 5, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_GB_%d_%d.jpg", m,n), width=6, height=6)


# Section 4: XGBoost -----------------------------------------------------------

#suni_results_Xtreme_Gradient_Boosting_split_5_augmentation_0

#No FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Xtreme_Gradient_Boosting_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: XGB  (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_XGB_%d_%d.jpg", m,n), width=6, height=6)


# Section 5: LightGBM ----------------------------------------------------------

#suni_results_Light_Gradient_Boosting_split_5_augmentation_0

# No FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Light_Gradient_Boosting_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: LGB  (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred
ggsave(sprintf("dv_ipred_plot_LGB_%d_%d.jpg", m,n), width=6, height=6)


# Section 6: MLP ---------------------------------------------------------------

#suni_results_MLP_two_hidden_layers_split_5_augmentation_0
#suni_results_MLP_one_hidden_layer_split_5_augmentation_0

# No FS 2 HL --------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_two_hidden_layers_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: MLP 2 HL  (test data)") +
  annotate("text", x = 5, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_2HL_%d_%d.jpg", m,n), width=6, height=6)


# No FS 1 HL --------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_one_hidden_layer_split_5_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [5] \n Method: MLP 1 HL  (test data)") +
  annotate("text", x = 10, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_1HL_%d_%d.jpg", m,n), width=6, height=6)


