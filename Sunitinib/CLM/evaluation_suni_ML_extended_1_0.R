# Suni Simple ML evaluation ----------------------------------------------------

#--------------------------------------------------#
# Evaluation of ML Suni models                     #
#--------------------------------------------------#
# Split 1                                          #
#--------------------------------------------------#


#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots

rm(list=ls())
 
library(dplyr)
library(ggplot2)

setwd("C:/Users/Olga/Suni_Pazo/ML/Split1")

# Split number m
# Augmentation percentage n

m <- 1
n <- 0

# Section 1: Random Forests-----------------------------------------------------

#suni_results_Random_Forest_split_1_FS_augmentation_0
#suni_results_Random_Forest_split_1_No_FS_augmentation_0

# FS ---------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Random_Forest_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: RF with FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_RF_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS ------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Random_Forest_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: RF without FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_RF_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 2: SVM ---------------------------------------------------------------

#suni_results_Support_Vector_Machine_split_1_FS_augmentation_0
#suni_results_Support_Vector_Machine_split_1_No_FS_augmentation_0

# FS ---------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Support_Vector_Machine_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: SVM with FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_SVM_FS_%d_%d.jpg", m,n), width=6, height=6)


# No FS ---------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Support_Vector_Machine_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: SVM without FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_SVM_No_FS_%d_%d.jpg", m,n), width=6, height=6)

# Section 3: Gradient Boosting -------------------------------------------------

#suni_results_Gradient_Boosting_split_1_FS_augmentation_0
#suni_results_Gradient_Boosting_split_1_No_FS_augmentation_0

# FS----------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Gradient_Boosting_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: GB with FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_GB_FS_%d_%d.jpg", m,n), width=6, height=6)


#No FS--------------------------------------------------------------------------

#suni_results_Gradient_Boosting_split_1_No_FS_augmentation_0

data_suni <- read.csv("suni_results_Gradient_Boosting_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: GB without FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_GB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 4: XGBoost -----------------------------------------------------------

#suni_results_Xtreme_Gradient_Boosting_split_1_FS_augmentation_0
#suni_results_Xtreme_Gradient_Boosting_split_1_No_FS_augmentation_0

# FS----------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Xtreme_Gradient_Boosting_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: XGB with FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_XGB_FS_%d_%d.jpg", m,n), width=6, height=6)


#No FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Xtreme_Gradient_Boosting_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: XGB without FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_XGB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 5: LightGBM ----------------------------------------------------------

#suni_results_Light_Gradient_Boosting_split_1_FS_augmentation_0
#suni_results_Light_Gradient_Boosting_split_1_No_FS_augmentation_0

# FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Light_Gradient_Boosting_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: LGB with FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred
ggsave(sprintf("dv_ipred_plot_LGB_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS--------------------------------------------------------------------------

data_suni <- read.csv("suni_results_Light_Gradient_Boosting_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: LGB without FS (test data)") +
  annotate("text", x = 15, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred
ggsave(sprintf("dv_ipred_plot_LGB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 6: MLP ---------------------------------------------------------------

#suni_results_MLP_two_hidden_layers_split_1_FS_augmentation_0
#suni_results_MLP_two_hidden_layers_split_1_No_FS_augmentation_0
#suni_results_MLP_one_hidden_layer_split_1_FS_augmentation_0
#suni_results_MLP_one_hidden_layer_split_1_No_FS_augmentation_0

# FS 2 HL -----------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_two_hidden_layers_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: MLP 2 HL with FS (test data)") +
  annotate("text", x = 20, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_2HL_FS_%d_%d.jpg", m,n), width=6, height=6)


# No FS 2 HL --------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_two_hidden_layers_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: MLP 2 HL without FS (test data)") +
  annotate("text", x = 20, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_2HL_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# FS 1 HL -----------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_one_hidden_layer_split_1_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: MLP 1 HL with FS (test data)") +
  annotate("text", x = 7, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_1HL_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS 1 HL --------------------------------------------------------------------

data_suni <- read.csv("suni_results_MLP_one_hidden_layer_split_1_No_FS_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_suni$Predicted~data_suni$Actual)

dv_ipred <- ggplot(data=data_suni,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ # Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [1] \n Method: MLP 1 HL without FS (test data)") +
  annotate("text", x = 10, y = max(data_suni$Actual)-1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_suni$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_suni$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("dv_ipred_plot_MLP_1HL_No_FS_%d_%d.jpg", m,n), width=6, height=6)


