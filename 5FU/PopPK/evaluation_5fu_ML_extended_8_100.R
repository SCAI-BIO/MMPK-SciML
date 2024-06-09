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

setwd("C:/Users/Olga/5FU_ML_neu/Augmentation/Split8")


# Split number m
# Augmentation percentage n

m <- 8
n <- 100

# Section 1: Random Forests-----------------------------------------------------

#5fu_results_Random_Forest_split_8_FS_augmentation_100
#5fu_results_Random_Forest_split_8_No_FS_augmentation_100

# FS ---------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Random_Forest_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: RF with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_RF_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS ------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Random_Forest_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: RF without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_RF_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 2: SVM ---------------------------------------------------------------

#5fu_results_Support_Vector_Machine_split_8_FS_augmentation_100
#5fu_results_Support_Vector_Machine_split_8_No_FS_augmentation_100

# FS ---------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Support_Vector_Machine_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: SVM with FS (test data) Augmentation 5FU") +
  annotate("text", x = 1.0, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_SVM_FS_%d_%d.jpg", m,n), width=6, height=6)


# No FS ---------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Support_Vector_Machine_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: SVM without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(-0.5,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_SVM_No_FS_%d_%d.jpg", m,n), width=6, height=6)

# Section 3: Gradient Boosting -------------------------------------------------

#5fu_results_Gradient_Boosting_split_8_FS_augmentation_100
#5fu_results_Gradient_Boosting_split_8_No_FS_augmentation_100

# FS----------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Gradient_Boosting_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: GB with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_GB_FS_%d_%d.jpg", m,n), width=6, height=6)


#No FS--------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Gradient_Boosting_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: GB without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_GB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 4: XGBoost -----------------------------------------------------------

#5fu_results_Xtreme_Gradient_Boosting_split_8_FS_augmentation_100
#5fu_results_Xtreme_Gradient_Boosting_split_8_No_FS_augmentation_100

# FS----------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Xtreme_Gradient_Boosting_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: XGB with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_XGB_FS_%d_%d.jpg", m,n), width=6, height=6)


#No FS--------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Xtreme_Gradient_Boosting_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: XGB without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_XGB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 5: LightGBM ----------------------------------------------------------

#5fu_results_Light_Gradient_Boosting_split_8_FS_augmentation_100
#5fu_results_Light_Gradient_Boosting_split_8_No_FS_augmentation_100

# FS--------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Light_Gradient_Boosting_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: LGB with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_LGB_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS-------------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_Light_Gradient_Boosting_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: LGB without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_LGB_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# Section 6: MLP ---------------------------------------------------------------

#5fu_results_MLP_two_hidden_layers_split_8_FS_augmentation_100
#5fu_results_MLP_two_hidden_layers_split_8_No_FS_augmentation_100
#5fu_results_MLP_one_hidden_layer_split_8_FS_augmentation_100
#5fu_results_MLP_one_hidden_layer_split_8_No_FS_augmentation_100

# FS 2 HL -----------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_two_hidden_layers_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 2 HL with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_2HL_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS 2 HL --------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_two_hidden_layers_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 2 HL without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_2HL_No_FS_%d_%d.jpg", m,n), width=6, height=6)


# FS 1 HL -----------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_one_hidden_layer_split_8_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 1 HL with FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_1HL_FS_%d_%d.jpg", m,n), width=6, height=6)

# No FS 1 HL --------------------------------------------------------------------

data_5fu <- read.csv("5fu_results_MLP_one_hidden_layer_split_8_No_FS_augmentation_100.csv")

#DV vs IPRED
model<-lm(data_5fu$Predicted~data_5fu$Actual)

dv_ipred <- ggplot(data=data_5fu,aes(x=Predicted,y=Actual))+ # Depict individual prediction
  geom_point(color="#004e9f", size=1, alpha=0.3)+ ## Add points
  geom_abline(intercept = 0, slope = 1,size=0.5)+ # Add line of unity
  ggtitle ("Observed Concentrations vs. Individual Predicted Concentrations [8] \n Method: MLP 1 HL without FS (test data) Augmentation 5FU") +
  annotate("text", x = 0.75, y = max(data_5fu$Actual)+1,
           label = paste("Adjusted R²:",round(summary(model)$adj.r.squared,4)),
           hjust = 1, vjust = 1) +
  theme(plot.title = element_text(size=11, hjust=0.5)) + # Set theme
  scale_x_continuous(limits=c(0,max(data_5fu$Predicted)+1)) +
  scale_y_continuous(limits=c(0,max(data_5fu$Actual)+1)) +
  xlab("Individual Predicted Concentration [ng/mL]") + ylab("Observed Concentration [ng/mL]") # Set axis labels
dv_ipred

ggsave(sprintf("5fu_dv_ipred_plot_MLP_1HL_No_FS_%d_%d.jpg", m,n), width=6, height=6)
