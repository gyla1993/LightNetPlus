# This file reads data from data source files and formats them into numpy arrays for training, validation and test.


import keras
import numpy as np
from global_var import num_WRF, num_LIG, num_AWS, num_PRED, dim_WRF, dim_AWS, la_grids, lo_grids, time_shift,\
    WRF_file_dir, LIG_file_dir, AWS_file_dir, WRF_ncl_file_dir, use_good_start, get_time_period
import datetime
import random
import math


variables3d = ['U', 'V', 'W', 'T', 'P','QVAPOR','QCLOUD','QRAIN','QICE','QHAIL',
               'QGRAUP','QSNOW','QEI','QEG','QEC','QES','QER','QEH','QESUM','REFL_10CM',
               'QNICE', 'QNSNOW', 'QNGRAUPEL']
variables2d = ['Q2', 'T2', 'TH2', 'PSFC', 'U10', 'V10', 'OLR', 'PBLH', 'W_max']
sumVariables2d = ['RAINC','RAINNC','HAILNC','FN']
variables3d_ave3 = ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3']
AWS_para_list = ['TEM', 'RHU', 'PRE_1h']
WRF_para_list = ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3', 'W_max']
E_3d = ['QESUM','QEI','QEG','QES']
mode_3d = 'select'
ncl_layers = 27
ncl_ave3 = True
ncl_disp = 'dbz'
num_WRF_Q = 28
label_type = 'bin'
high_levels = [i for i in range(27)]


def getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, dim):
    grid = np.zeros(shape=[num_WRF, la_grids, lo_grids, dim], dtype=np.float32)
    s_idx = 0
    delta_hour -= 6   # important!!!!
    for s in WRF_para_list:
        npy_grid = np.load(npyFileDir + '%s.npy' % s)
        if s in variables3d_ave3:
            temp = npy_grid[delta_hour:delta_hour + num_WRF, 0:9, 0:159, 0:159]
            temp = np.transpose(temp, (0, 2, 3, 1))
            if s in E_3d:
                temp = np.abs(temp)
            if s in ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3']:     # no negative values
                temp[temp < 0] = 0
                # temp = temp * 100000         #!!!!!!!!!!!!
            grid[:, :, :, s_idx:s_idx+9] = temp
            s_idx += 9
        elif s in variables3d:
            if mode_3d == 'ave':
                temp = np.zeros((num_WRF, la_grids, lo_grids))
                for t in range(27):
                    temp += npy_grid[delta_hour:delta_hour+num_WRF, t, 0:159, 0:159]
                grid[:,:,:,s_idx] = temp / 27.0
                s_idx += 1
            elif mode_3d == 'select':
                temp = npy_grid[delta_hour:delta_hour+num_WRF, high_levels[0]:high_levels[-1]+1, 0:159, 0:159]
                temp = np.transpose(temp, (0, 2, 3, 1))
                if s in E_3d:
                    temp = np.abs(temp)
                if s in ['QICE','QGRAUP','QSNOW']: # no negative values
                    temp[temp < 0] = 0
                    # temp = temp * 100000

                grid[:, :, :, s_idx:s_idx + len(high_levels)] = temp
                s_idx += len(high_levels)
        elif s in variables2d or s in sumVariables2d:
            grid[:,:,:,s_idx] = npy_grid[delta_hour:delta_hour+num_WRF, 0:159, 0:159]
            s_idx += 1
    return grid

def getHoursNCLGridFromNPY(ft, nchour, delta_hour):
    if ncl_ave3 and ncl_layers == 27:
        grid = np.zeros(shape=[num_WRF, la_grids, lo_grids, 9], dtype=np.float32)
    else:
        grid = np.zeros(shape=[num_WRF, la_grids, lo_grids, ncl_layers], dtype=np.float32)
    for t in range(num_PRED):
        if ncl_ave3 and ncl_layers == 27:
            nclfilepath = WRF_ncl_file_dir + ft.strftime('%Y%m%d') + '/' + nchour + '/' + '%d_dbz_ave3.npy' % (delta_hour + t)
        else:
            nclfilepath = WRF_ncl_file_dir + ft.strftime('%Y%m%d') + '/' + nchour + '/' + '_%d_%s.npy' % (delta_hour + t, ncl_disp)
        tmp = np.load(nclfilepath)    # (27, 159, 159) or (9, 159, 159)
        grid[t, :, :, :] = np.transpose(tmp, (1, 2, 0))
    return grid

class DataGenerator_LightNetPlus(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6

            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)


            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours = hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid > 1] = 1
                labels_batch[i, hour_plus,:,:] = truth_grid[:,np.newaxis]
            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1
            # read history observations
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                LIG_batch[i, hour_plus, :, :, :] = truth_grid[:, :, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, LIG_batch, AWS_batch, truth_m1_batch], labels_batch

class PredictDataGenerator_LightNetPlus(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lists, batch_size, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1
            # if use_ncl:
            #     WRF_batch[i, :, :, :, -ncl_layers:] = getHoursNCLGrid(ft, nchour, delta_hour)
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                LIG_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(la_grids, lo_grids))[:, :, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, LIG_batch, AWS_batch, truth_m1_batch]

class DataGenerator_StepDeep(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6

            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)


            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours = hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid > 1] = 1
                labels_batch[i, hour_plus,:,:] = truth_grid[:,np.newaxis]
            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1
            # read history observations
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                LIG_batch[i, hour_plus, :, :, :] = truth_grid[:, :, np.newaxis]
        return [WRF_batch, LIG_batch, AWS_batch], labels_batch

class PredictDataGenerator_StepDeep(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lists, batch_size, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1
            # if use_ncl:
            #     WRF_batch[i, :, :, :, -ncl_layers:] = getHoursNCLGrid(ft, nchour, delta_hour)
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                LIG_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(la_grids, lo_grids))[:, :, np.newaxis]
        return [WRF_batch, LIG_batch, AWS_batch]

class DataGenerator_LightNetPlus_WRF(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)


        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6

            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_WRF(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, truth_m1_batch]

class DataGenerator_LightNetPlus_LIG(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)


        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')

            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]
            # read history observations
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                LIG_batch[i, hour_plus, :la_grids, :lo_grids, :] = truth_grid[:, :, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [LIG_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_LIG(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')


            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                LIG_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(la_grids, lo_grids))[:, :, np.newaxis]

        if use_good_start:
            hour_plus = num_LIG-1
            dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
            tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
            with open(tFilePath) as tfile:
                truth_grid = np.array(tfile.readlines(), dtype=np.float32)
            truth_grid.resize(la_grids, lo_grids)
            truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [LIG_batch, truth_m1_batch]

class DataGenerator_LightNetPlus_AWS(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]

            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            if use_good_start:
                hour_plus = num_LIG - 1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [AWS_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_AWS(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')

            # read sta discrete
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [AWS_batch, truth_m1_batch]

class DataGenerator_LightNetPlus_WRF_LIG(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)
            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]

            # read history observations
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                LIG_batch[i, hour_plus, :la_grids, :lo_grids, :] = truth_grid[:, :, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, LIG_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_WRF_LIG(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                LIG_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(la_grids, lo_grids))[:, :, np.newaxis]

        if use_good_start:
            hour_plus = num_LIG-1
            dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
            tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
            with open(tFilePath) as tfile:
                truth_grid = np.array(tfile.readlines(), dtype=np.float32)
            truth_grid.resize(la_grids, lo_grids)
            truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, LIG_batch, truth_m1_batch]

class DataGenerator_LightNetPlus_WRF_AWS(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read WRF
            utc = ddt + datetime.timedelta(hours=-8)  # Beijing time to UTC
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # Shift time to select a proper WRF file
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)
            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]
            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, AWS_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_WRF_AWS(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        WRF_batch = np.zeros(shape=[batchsize, num_WRF, la_grids, lo_grids, dim_WRF], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6) * time_shift)
            nchour, delta_hour = get_time_period(ft)
            delta_hour += time_shift * 6
            date_str = ft.date().strftime("%Y%m%d")
            npyFileDir = WRF_file_dir + '%s/' % date_str + '%s/' % nchour
            WRF_batch[i,:,:,:, 0:num_WRF_Q] = getHoursGridFromNPY(npyFileDir, delta_hour, WRF_para_list, num_WRF_Q)
            WRF_batch[i, :, :, :, num_WRF_Q:] = getHoursNCLGridFromNPY(ft, nchour, delta_hour)

            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [WRF_batch, AWS_batch, truth_m1_batch]

class DataGenerator_LightNetPlus_LIG_AWS(keras.utils.Sequence):
    def __init__(self, lists, batch_size, n_labels, generator_type, PRE_1h_scale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.lists = lists
        self.n_classes = n_labels
        self.shuffle = shuffle
        self.type = generator_type
        self.PRE_1h_scale = PRE_1h_scale
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(list_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.type != 'val':
            random.shuffle(self.lists)

    def __data_generation(self, list_batch):
        # print('__data_generation begin')
        # time_start = TI.time()
        batchsize = len(list_batch)
        labels_batch = np.zeros(shape=[batchsize, num_PRED, la_grids * lo_grids, 1], dtype=np.float32)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)


        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')

            for hour_plus in range(num_PRED):
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                # convert to binary
                if label_type == 'bin':
                    truth_grid[truth_grid >= 1] = 1
                labels_batch[i, hour_plus, :, :] = truth_grid[:, np.newaxis]
            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            # read history observations
            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                LIG_batch[i, hour_plus, :la_grids, :lo_grids, :] = truth_grid[:, :, np.newaxis]

        if use_good_start:
            hour_plus = num_LIG - 1
            dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
            tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
            with open(tFilePath) as tfile:
                truth_grid = np.array(tfile.readlines(), dtype=np.float32)
            truth_grid.resize(la_grids, lo_grids)
            truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [LIG_batch, AWS_batch, truth_m1_batch], [labels_batch]

class PredictDataGenerator_LightNetPlus_LIG_AWS(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, lists, batch_size, PRE_1h_scale=1.0, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.lists = lists
        self.PRE_1h_scale = PRE_1h_scale
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.lists) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.lists[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(list_batch)
        return X

    def __data_generation(self, list_batch):
        'Generates data containing batch_size samples'
        batchsize = len(list_batch)
        LIG_batch = np.zeros(shape=[batchsize, num_LIG, la_grids, lo_grids, 1], dtype=np.float32)
        AWS_batch = np.zeros(shape=[batchsize, num_AWS, la_grids, lo_grids, dim_AWS], dtype=np.float32)
        truth_m1_batch = np.zeros(shape=[batchsize, 1, la_grids, lo_grids, 1], dtype=np.float32)

        for i, datetime_peroid in enumerate(list_batch):
            ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
            # read sta interp
            for hour_plus in range(num_AWS):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_AWS)
                c_idx = 0
                for pa in AWS_para_list:
                    sFilePath = AWS_file_dir + '%s/' % pa + dt.strftime('%Y%m%d%H') + '.npy'
                    sta_grid = np.load(sFilePath)
                    AWS_batch[i, hour_plus, :, :, c_idx] = sta_grid
                    c_idx += 1

            for hour_plus in range(num_LIG):
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                LIG_batch[i, hour_plus, :, :, :] = (truth_grid.reshape(la_grids, lo_grids))[:, :, np.newaxis]

            if use_good_start:
                hour_plus = num_LIG-1
                dt = ddt + datetime.timedelta(hours=hour_plus - num_LIG)
                tFilePath = LIG_file_dir + dt.strftime('%Y%m%d%H%M') + '_truth'
                with open(tFilePath) as tfile:
                    truth_grid = np.array(tfile.readlines(), dtype=np.float32)
                truth_grid.resize(la_grids, lo_grids)
                truth_m1_batch[i, 0, :, :, :] = truth_grid[:, :, np.newaxis]

        return [LIG_batch, AWS_batch, truth_m1_batch]




