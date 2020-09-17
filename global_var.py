# This file defines global variables which will be used in the whole project.

num_WRF = 6     # The number of hours of WRF simulation used in one prediction case.
num_LIG = 3     # The number of hours of lightning observations used in one prediction case.
num_AWS = 3     # The number of hours of AWS observations used in one prediction case.
num_PRED = 6    # The number of prediction hours.
la_grids = 159  # The number of grid intervals in the latitude direction.
lo_grids = 159  # The number of grid intervals in the longitude direction.
dim_WRF = 37    # The number of WRF simulated parameters.
dim_AWS = 3     # The number of AWS observed parameters.
use_gpu = False  # Whether use gpu or cpu for training, validation and test.
train_set_file = 'training_periods.txt'     # The periods used for training. 2015.06-09 & 2016.05-09 & 2017.05-06
val_set_file = 'validation_periods.txt'     # The periods used for validation. 2017.07
test_set_file = 'test_periods.txt'          # The periods used for test. 2017.08-09
model_file_dir = 'train_dir/models/'        # Trained models are saved in this directory.
record_file_dir = 'train_dir/records/'      # Training logs are saved in this directory.
result_file_dir = 'test_dir/results/'       # Prediction results are saved in this directory.
score_file_dir = 'test_dir/scores/'         # Test scores are saved in this directory.
visualization_file_dir = 'test_dir/visualization/'  # Visualization results are saved in this directory.
curve_file_dir = 'test_dir/curves/'         # Performance curves are saved in this directory.
WRF_file_dir = 'data_dir/WRF/'              # The WRF simulation data (except WRF ncl) are saved in this directory.
WRF_ncl_file_dir = 'data_dir/WRF_ncl/'      # The WRF ncl data are saved in this directory.
LIG_file_dir = 'data_dir/LIG/'              # The lightning observation data are saved in this directory.
AWS_file_dir = 'data_dir/AWS/'              # The AWS observation data are saved in this directory.


# The following variables are used for data pre-processing and can be ignored.

time_shift = 1
use_good_start = False

def get_time_period(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6:
        nchour = '00'
    elif 6 <= hour < 12:
        nchour = '06'
    elif 12 <= hour < 18:
        nchour = '12'
    elif 18 <= hour <= 23:
        nchour = '18'
    else:
        print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour
