# 5FU Simple ML evaluation ----------------------------------------------------

#--------------------------------------------------#
# Evaluation of ML 5FU models                      #
#--------------------------------------------------#
# Split 8                                          #
#--------------------------------------------------#


#--------------------------------------------------#
# packages                                         #
#--------------------------------------------------#

#dplyr: data wrangling
#ggplot2: create plots


rm(list=ls())

library(dplyr) 
library(ggplot2)

setwd("C:/Users/teply/Documents/5FU_NONMEM_final/ML_results/Split8")

# Split number m
# Augmentation percentage n, replace with 100 if 100% augmentation
# additionally, set augmentation_100 in all dataset titles

m <- 8
n <- 0

# Section 1: Random Forests-----------------------------------------------------

data_5fu <- read.csv("5fu_results_Random_Forest_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: RF (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_RF_%d_%d.jpg", m,n), width=6, height=6)


# Section 2: SVM ---------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Support_Vector_Machine_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: SVM (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_SVM_%d_%d.jpg", m,n), width=6, height=6)

# Section 3: Gradient Boosting -------------------------------------------------

data_5fu <- read.csv("5fu_results_Gradient_Boosting_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: GB (test data)  5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_GB_%d_%d.jpg", m,n), width=6, height=6)


# Section 4: XGBoost -----------------------------------------------------------

data_5fu <- read.csv("5fu_results_Xtreme_Gradient_Boosting_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: XGB (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_XGB_%d_%d.jpg", m,n), width=6, height=6)


# Section 5: LightGBM ----------------------------------------------------------

data_5fu <- read.csv("5fu_results_Light_Gradient_Boosting_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: LGB (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_LGB_%d_%d.jpg", m,n), width=6, height=6)


# Section 6: MLP ---------------------------------------------------------------


# 2 HL -------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_two_hidden_layers_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 2 HL (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_2HL_%d_%d.jpg", m,n), width=6, height=6)


# 1 HL ------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_one_hidden_layer_split_8_augmentation_0.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 1 HL (test data) 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_1HL_%d_%d.jpg", m,n), width=6, height=6)

