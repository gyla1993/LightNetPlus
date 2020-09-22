
# LightNetPlus

data_dir　　　　　\# Validation and test data are stored. <br>
　|-AWS　　　　　\# Automatic weather station data. <br>
　|-LIG　　　　　　\# Lightning observation data. <br>
　|-WRF　　　　　\# WRF simulation data: micro-physical parameters and maximum vertical velocity. <br>
　|-WRF_ncl　　　　\# WRF simulation data: radar reflectivity. <br>

test_dir　　　　　\# Files related to testing are stored. <br>
　|-curves　　　　\# Performance curves in hours. <br>
　|-results　　　　\# Prediction results by every model. <br>
　|-scores　　　　\# Performance scores (POD, FAR, TS, ETS) for every model. <br>
　|-visualization　　\# Case visualization results for every test period. <br>

train_dir　　　　　\# Files related to training are stored. <br>
　|-models　　　　\# Trained models. <br>
　|-records　　　　\# Training log files. <br>

data_generator.py　　　\# Read data from data_dir and formats them into numpy arrays.

draw_each_hour_curve.py　　　\# Draw performance curves in hours.

global_var.py　　　\# Define global variables which will be used in the whole project.

model_def.py　　　\# Define structure of all models.

score.py　　　\# Calculate performance scores for prediction results.

test.py　　　\# Test a trained model and calculate performance scores for the model.

test_periods.txt　　　\# The periods used for test. 2017.08-09.

train.py　　　\# Train a deep neural network model.

training_periods.txt　　　\# The periods used for training. 2015.06-09 & 2016.05-09 & 2017.05-06

validation_periods.txt　　　\# The periods used for validation. 2017.07

visualization_case.py　　　\# Case visualization for every test period.
