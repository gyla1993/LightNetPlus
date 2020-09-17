# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import keras.backend as K
import datetime
import numpy as np
import score
from keras.models import load_model
from global_var import dim_WRF, dim_AWS, num_LIG, num_AWS, num_PRED, use_good_start, test_set_file, model_file_dir,\
    time_shift, get_time_period, la_grids, lo_grids, result_file_dir, use_gpu
import train
from data_generator import DataGenerator_LightNetPlus, PredictDataGenerator_LightNetPlus,\
    DataGenerator_StepDeep, PredictDataGenerator_StepDeep,\
    DataGenerator_LightNetPlus_WRF, PredictDataGenerator_LightNetPlus_WRF,\
    DataGenerator_LightNetPlus_LIG, PredictDataGenerator_LightNetPlus_LIG,\
    DataGenerator_LightNetPlus_AWS, PredictDataGenerator_LightNetPlus_AWS,\
    DataGenerator_LightNetPlus_WRF_LIG, PredictDataGenerator_LightNetPlus_WRF_LIG,\
    DataGenerator_LightNetPlus_WRF_AWS, PredictDataGenerator_LightNetPlus_WRF_AWS,\
    DataGenerator_LightNetPlus_LIG_AWS, PredictDataGenerator_LightNetPlus_LIG_AWS

def do_test(test_list, model, model_file, model_num):
    # --------------------------------------------------
    test_batch_size = 2

    if model_num == 1:
        test_gen = PredictDataGenerator_LightNetPlus(test_list, test_batch_size)
    elif model_num == 2:
        test_gen = PredictDataGenerator_StepDeep(test_list, test_batch_size)
    elif model_num == 3:
        test_gen = PredictDataGenerator_LightNetPlus_WRF(test_list, test_batch_size)
    elif model_num == 4:
        test_gen = PredictDataGenerator_LightNetPlus_LIG(test_list, test_batch_size)
    elif model_num == 5:
        test_gen = PredictDataGenerator_LightNetPlus_AWS(test_list, test_batch_size)
    elif model_num == 6:
        test_gen = PredictDataGenerator_LightNetPlus_WRF_LIG(test_list, test_batch_size)
    elif model_num == 7:
        test_gen = PredictDataGenerator_LightNetPlus_WRF_AWS(test_list, test_batch_size)
    elif model_num == 8:
        test_gen = PredictDataGenerator_LightNetPlus_LIG_AWS(test_list, test_batch_size)

    print('generating test data and predicting...')
    y_pred = model.predict_generator(test_gen, workers=3, verbose=1)  # [len(test_list),num_frames,159*159,1]
    ypred = y_pred
    ypred = 1.0 / (1.0 + np.exp(-ypred))  # if model for prediction doesn't include a sigmoid layer
    with tf.device('/cpu:0'):
        for id, ddt_item in enumerate(test_list):
            ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)  # 北京时间转换成UTC时间
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # 时间偏移
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            for hour_plus in range(num_PRED):
                ypred_i = ypred[id][hour_plus]
                dt = ddt + datetime.timedelta(hours=hour_plus)
                dt_item = dt.strftime('%Y%m%d%H%M')
                result_path = result_file_dir+model_file+'/'
                if not os.path.isdir(result_path):
                    os.makedirs(result_path)
                with open(result_path + '%s_h%d' % (dt_item, hour_plus), 'w') as rfile:
                    for i in range(la_grids*lo_grids):
                        rfile.write('%f\r\n' % ypred_i[i])    # the probability value
                print(dt_item)

if __name__ == "__main__":

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    test_list = []
    with open(test_set_file, 'r') as file:
        for line in file:
            test_list.append(line.rstrip('\n'))

    model_file = 'LightNetPlus-09.hdf5'     # The model file to be tested.
    model_num = 1   # The model number corresponds to model_file. See do_test for more details.
    trained_model = load_model(model_file_dir + model_file,
                               custom_objects={'dim_WRF': dim_WRF, 'dim_AWS': dim_AWS, 'num_LIG': num_LIG,
                                               'num_AWS': num_AWS, 'num_PRED': num_PRED,
                                               'use_good_start': use_good_start,
                                               'weighted_loss': train.weighted_loss,
                                               'POD': train.POD, 'FAR': train.FAR, 'TS': train.TS,
                                               'ETS': train.ETS, 'binary_acc': train.binary_acc})
    do_test(test_list, trained_model, model_file, model_num)
    score.eva(model_file, 0.5)
    score.eva_each_hour(model_file, 0.5)
    sess.close()

