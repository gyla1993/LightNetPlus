
# LightNetPlus
### Directory description
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

data_generator.py　　　\# Load data from data_dir and formats them into numpy arrays.

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
### Runtime environment
Our code requires python 3.6 with packages: tensorflow 1.13.1, keras 2.2.4 and numpy. If you use [Anaconda](https://www.anaconda.com/), the following commands will help you create a feasible runtime environment.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conda create -n py36_keras224 python=3.6

conda activate py36_keras224 

conda install tensorflow-gpu==1.13.1

pip install keras==2.2.4

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Then, you may need to remove the following code 
```python
    inputs, initial_state, constants = _standardize_args(
         inputs, initial_state, constants, self._num_constants)
```
from "keras/layers/convolutional_recurrent.py", due to a bug in ConvLSTM2D of keras. cf. https://github.com/keras-team/keras/issues/9761

### Reference  
```
@article{geng2021deep,
  title={A deep learning framework for lightning forecasting with multi-source spatiotemporal data},
  author={Geng, Yangli-ao and Li, Qingyong and Lin, Tianyang and Yao, Wen and Xu, Liangtao and Zheng, Dong and Zhou, Xinyuan and Zheng, Liming and Lyu, Weitao and Zhang, Yijun},
  journal={Quarterly Journal of the Royal Meteorological Society},
  volume={147},
  number={741},
  pages={4048--4062},
  year={2021},
  publisher={Wiley Online Library}
}
```
