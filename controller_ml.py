import train_model
import classify_data
import sys


#arg1 = sys.argv[1]


#if arg1 == "Train":
train_model.train_model.TrainModelResponsability()
train_model.train_model.TrainModelChargeability()


classify_data.classify_data.ClassifyResponsability()


#train_model.train_model.TrainModelAction()
#if arg1 == "Classify":


#classify_data.classify_data.ClassifyFailCode()
