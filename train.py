import os
import model_def
import tensorflow as tf
import keras.backend as K
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import datetime
import numpy as np
from global_var import num_PRED, train_set_file, val_set_file, test_set_file, la_grids, lo_grids,\
    model_file_dir, LIG_file_dir, record_file_dir, use_gpu
from data_generator import DataGenerator_LightNetPlus, PredictDataGenerator_LightNetPlus,\
    DataGenerator_StepDeep, PredictDataGenerator_StepDeep,\
    DataGenerator_LightNetPlus_WRF, PredictDataGenerator_LightNetPlus_WRF,\
    DataGenerator_LightNetPlus_LIG, PredictDataGenerator_LightNetPlus_LIG,\
    DataGenerator_LightNetPlus_AWS, PredictDataGenerator_LightNetPlus_AWS,\
    DataGenerator_LightNetPlus_WRF_LIG, PredictDataGenerator_LightNetPlus_WRF_LIG,\
    DataGenerator_LightNetPlus_WRF_AWS, PredictDataGenerator_LightNetPlus_WRF_AWS,\
    DataGenerator_LightNetPlus_LIG_AWS, PredictDataGenerator_LightNetPlus_LIG_AWS



global_val_sumETS_max = -1e10
global_val_sumETS_max_loss = 1e10
global_val_sumETS_max_epoch = -1
global_val_loss_min = 1e10
global_val_loss_min_ETS = -1e10
global_val_loss_min_epoch = -1
model_record_name = None

def POD(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    possible_positives = K.sum(ytrue)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def FAR(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    predicted_positives = K.sum(ypred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return 1 - precision

def TS(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    N1 = K.sum(ytrue * ypred)
    N1pN2 = K.sum(ypred)
    N1pN3 = K.sum(ytrue)
    N2 = N1pN2 - N1
    N3 = N1pN3 - N1
    TS = N1 / (N1 + N2 + N3 + K.epsilon())
    return TS

def ETS(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    N1 = K.sum(ytrue * ypred)
    N1pN2 = K.sum(ypred)
    N1pN3 = K.sum(ytrue)
    N2 = N1pN2 - N1
    N3 = N1pN3 - N1
    r = (N1+N2)*(N1+N3)/(la_grids*lo_grids)
    TS = (N1-r) / (N1 + N2 + N3 - r + K.epsilon())
    return TS

def binary_acc(y_true,y_pred):
    ypred = K.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(ypred)), axis=-1)

def weighted_loss(y_true, y_pred):  # binary classification
    positive_weight = 18
    ytrue = K.flatten(y_true)
    ypred = K.flatten(y_pred)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=ypred, targets=ytrue, pos_weight=positive_weight))

def do_train(train_list, val_list, test_list, model_num):

    # train setting
    train_batch_size = 8
    val_batchsize = 1
    class_num = 2
    epochs_num = 25
    initial_epoch_num = 0
    init_learning_rate = 0.0001
    training_patience = 5

    # global_para_name_list = ['train_batchs_ize', 'epochs_num', 'init_learning_rate', 'training_patience']
    # global_para_list = [train_batch_size, epochs_num, init_learning_rate, training_patience]

    '''
    Prepare data and model according to model_num.
    '''
    if model_num == 1:
        train_gen = DataGenerator_LightNetPlus(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus(test_list, val_batchsize)
        model = model_def.LightNetPlus()

    elif model_num == 2:
        train_gen = DataGenerator_StepDeep(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_StepDeep(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_StepDeep(val_list, val_batchsize)
        test_gen = PredictDataGenerator_StepDeep(test_list, val_batchsize)
        model = model_def.StepDeep()

    elif model_num == 3:
        train_gen = DataGenerator_LightNetPlus_WRF(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_WRF(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_WRF(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_WRF(test_list, val_batchsize)
        model = model_def.LightNetPlus_WRF()

    elif model_num == 4:
        train_gen = DataGenerator_LightNetPlus_LIG(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_LIG(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_LIG(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_LIG(test_list, val_batchsize)
        model = model_def.LightNetPlus_LIG()

    elif model_num == 5:
        train_gen = DataGenerator_LightNetPlus_AWS(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_AWS(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_AWS(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_AWS(test_list, val_batchsize)
        model = model_def.LightNetPlus_AWS()

    elif model_num == 6:
        train_gen = DataGenerator_LightNetPlus_WRF_LIG(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_WRF_LIG(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_WRF_LIG(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_WRF_LIG(test_list, val_batchsize)
        model = model_def.LightNetPlus_WRF_LIG()

    elif model_num == 7:
        train_gen = DataGenerator_LightNetPlus_WRF_AWS(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_WRF_AWS(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_WRF_AWS(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_WRF_AWS(test_list, val_batchsize)
        model = model_def.LightNetPlus_WRF_AWS()

    elif model_num == 8:
        train_gen = DataGenerator_LightNetPlus_LIG_AWS(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_LightNetPlus_LIG_AWS(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_LightNetPlus_LIG_AWS(val_list, val_batchsize)
        test_gen = PredictDataGenerator_LightNetPlus_LIG_AWS(test_list, val_batchsize)
        model = model_def.LightNetPlus_LIG_AWS()


    adam = optimizers.Adam(lr=init_learning_rate)
    model.compile(
        loss=weighted_loss,
        optimizer=adam,
        metrics=[POD, FAR, TS, ETS, binary_acc])
    model_file_name = "%s-{epoch:02d}.hdf5" % model.name

    checkpoint = ModelCheckpoint(model_file_dir + model_file_name, monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')

    # get label of validation set
    val_label = np.zeros(shape=[len(val_list), num_PRED, 159 * 159, 1], dtype=np.float32)
    print('generating val label...')
    for id, ddt_item in enumerate(val_list):
        ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
        for hour_plus in range(num_PRED):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_item = dt.strftime('%Y%m%d%H%M')
            with open(LIG_file_dir + dt_item + '_truth') as tfile:
                y_test = np.array(tfile.readlines(), dtype=int)
            y_test[y_test >= 1] = 1
            val_label[id, hour_plus, :, 0] = y_test

    test_label = np.zeros(shape=[len(test_list), num_PRED, 159 * 159, 1], dtype=np.float32)
    print('generating test label...')
    for id, ddt_item in enumerate(test_list):
        ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
        for hour_plus in range(num_PRED):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_item = dt.strftime('%Y%m%d%H%M')
            with open(LIG_file_dir + dt_item + '_truth') as tfile:
                y_test = np.array(tfile.readlines(), dtype=int)
            y_test[y_test >= 1] = 1
            test_label[id, hour_plus, :, 0] = y_test

    global model_record_name

    model_record_name = model.name

    class RecordMetricsAfterEpoch(Callback):
        def on_epoch_end(self, epoch, logs={}):
            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('epoch %d:\r\n' % (epoch + 1))
                for key in ['loss', 'POD', 'FAR', 'TS', 'binary_acc', 'val_loss', 'val_POD', 'val_FAR', 'val_TS']:
                    f.write('%s: %f   ' % (key, logs[key]))

    class calVALmetrics(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('generating val data...')
            y_pred = self.model.predict_generator(val_gen_2, workers=5, verbose=1)
            y_pred = y_pred
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            y_pred = np.round(y_pred)
            y_pred = y_pred.reshape(len(val_list), num_PRED, 159 * 159, 1)  # (len(val_list),12,159*159,1)
            y_true = val_label  # (len(val_list),12,159*159,1)
            y_pred = np.sum(y_pred, axis=1)  # (len(val_list),1,159*159,1)
            y_true = np.sum(y_true, axis=1)  # (len(val_list),1,159*159,1)
            tot = np.zeros(4, dtype=np.float)
            for id in range(len(val_list)):
                print(val_list[id])
                ypred = y_pred[id]  # (1,159*159,1)
                ytrue = y_true[id]  # (1,159*159,1)
                tN = np.zeros(4, dtype=np.float)
                tN[0] = np.sum((ypred > 0) & (ytrue > 0))
                tN[1] = np.sum((ypred > 0) & (ytrue < 1))
                tN[2] = np.sum((ypred < 1) & (ytrue > 0))
                tN[3] = np.sum((ypred < 1) & (ytrue < 1))
                tot += tN
            POD = tot[0] / (tot[0] + tot[2] + 10e-7)
            FAR = tot[1] / (tot[0] + tot[1] + 10e-7)
            TS = tot[0] / (tot[0] + tot[1] + tot[2] + 10e-7)
            R = (tot[0] + tot[1]) * (tot[0] + tot[2]) / (tot[0] + tot[1] + tot[2] + tot[3])
            ETS = (tot[0] - R) / ((tot[0] + tot[1] + tot[2]) - R)


            global global_val_sumETS_max, global_val_sumETS_max_loss, global_val_sumETS_max_epoch, global_val_loss_min, global_val_loss_min_ETS, global_val_loss_min_epoch
            if ETS > global_val_sumETS_max:
                global_val_sumETS_max = ETS
                global_val_sumETS_max_loss = logs['val_loss']
                global_val_sumETS_max_epoch = epoch + 1
            if logs['val_loss'] < global_val_loss_min:
                global_val_loss_min = logs['val_loss']
                global_val_loss_min_ETS = ETS
                global_val_loss_min_epoch = epoch + 1

            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('val_sumPOD: %f   ' % POD)
                f.write('val_sumFAR: %f   ' % FAR)
                f.write('val_sumTS: %f   ' % TS)
                f.write('val_sumETS: %f   ' % ETS)
                f.write('\r\n')

    class calTestmetrics(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('generating test data...')
            y_pred = self.model.predict_generator(test_gen, workers=5, verbose=1)
            y_pred = y_pred
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            y_pred = np.round(y_pred)
            y_pred = y_pred.reshape(len(test_list), num_PRED, 159 * 159, 1)  # (len(val_list),12,159*159,1)
            y_true = test_label  # (len(val_list),12,159*159,1)
            y_pred = np.sum(y_pred, axis=1)  # (len(val_list),1,159*159,1)
            y_true = np.sum(y_true, axis=1)  # (len(val_list),1,159*159,1)
            tot = np.zeros(4, dtype=np.float)
            for id in range(len(test_list)):
                print(test_list[id])
                ypred = y_pred[id]  # (1,159*159,1)
                ytrue = y_true[id]  # (1,159*159,1)
                tN = np.zeros(4, dtype=np.float)
                tN[0] = np.sum((ypred > 0) & (ytrue > 0))
                tN[1] = np.sum((ypred > 0) & (ytrue < 1))
                tN[2] = np.sum((ypred < 1) & (ytrue > 0))
                tN[3] = np.sum((ypred < 1) & (ytrue < 1))
                tot += tN
            POD = tot[0] / (tot[0] + tot[2] + 10e-7)
            FAR = tot[1] / (tot[0] + tot[1] + 10e-7)
            TS = tot[0] / (tot[0] + tot[1] + tot[2] + 10e-7)
            R = (tot[0] + tot[1]) * (tot[0] + tot[2]) / (tot[0] + tot[1] + tot[2] + tot[3])
            ETS = (tot[0] - R) / ((tot[0] + tot[1] + tot[2]) - R)

            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('test_sumPOD: %f   ' % POD)
                f.write('test_sumFAR: %f   ' % FAR)
                f.write('test_sumTS: %f   ' % TS)
                f.write('test_sumETS: %f   ' % ETS)
                f.write('\r\n')

    RMAE = RecordMetricsAfterEpoch()
    CVALM = calVALmetrics()
    CTESTM = calTestmetrics()
    DECAY = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=training_patience, verbose=1, mode='min', min_delta=0.0001, cooldown=3,
                              min_lr=1e-6)
    model.fit_generator(train_gen,
                        validation_data=val_gen,
                        epochs=epochs_num,
                        initial_epoch=initial_epoch_num,
                        workers=5,
                        max_queue_size=10,
                        callbacks=[DECAY, RMAE, CVALM, CTESTM, checkpoint]
                        )

if __name__ == "__main__":

    model_num = 5   # Choose the model to be trained. Can be set to 1, 2, ..., 8. See the do_train function for more details.

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)



    train_list = []
    with open(train_set_file, 'r') as file:
        for line in file:
            train_list.append(line.rstrip('\n'))

    val_list = []
    with open(val_set_file, 'r') as file:
        for line in file:
            val_list.append(line.rstrip('\n'))

    test_list = []
    with open(test_set_file, 'r') as file:
        for line in file:
            test_list.append(line.rstrip('\n'))

    # global global_val_sumETS_max, global_val_sumETS_max_loss, global_val_sumETS_max_epoch, global_val_loss_min, global_val_loss_min_ETS, global_val_loss_min_epoch
    global_val_sumETS_max = -1e10
    global_val_sumETS_max_loss = 1e10
    global_val_sumETS_max_epoch = -1
    global_val_loss_min = 1e10
    global_val_loss_min_ETS = -1e10
    global_val_loss_min_epoch = -1

    do_train(train_list, val_list, test_list, model_num)

    with open(record_file_dir + model_record_name + '.txt', 'a') as f:
        f.write('global_val_loss_min: %f   ' % global_val_loss_min)
        f.write('global_val_loss_min_ETS: %f   ' % global_val_loss_min_ETS)
        f.write('global_val_loss_min_epoch: %d   ' % global_val_loss_min_epoch)
        f.write('\r\n')
        f.write('global_val_sumETS_max: %f   ' % global_val_sumETS_max)
        f.write('global_val_sumETS_max_loss: %f   ' % global_val_sumETS_max_loss)
        f.write('global_val_sumETS_max_epoch: %d   ' % global_val_sumETS_max_epoch)
        f.write('\r\n')
