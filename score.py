import datetime
import os
import numpy as np
from global_var import LIG_file_dir, result_file_dir, score_file_dir

global_scores_list = []
global_scores_neig_list = []
global_score_formal = []
global_score_neig_formal = []

def convert_to_binary(data, threshold):
    if data.ndim == 1:
        for i in range(len(data)):
            if data[i] >= threshold:
                data[i] = 1
            else:
                data[i] = 0
    elif data.ndim == 2:
        mm,nn = data.shape
        for i in range(mm):
            for j in range(nn):
                if data[i][j] >= threshold:
                    data[i][j] = 1
                else:
                    data[i][j] = 0

def calc_four_situation(tN, y_pred, y_test):
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    tN[0] = np.sum((y_pred > 0) & (y_test > 0))
    tN[1] = np.sum((y_pred > 0) & (y_test < 1))
    tN[2] = np.sum((y_pred < 1) & (y_test > 0))
    tN[3] = np.sum((y_pred < 1) & (y_test < 1))
    print("N1=%d,  N2=%d,  N3=%d,  N4=%d" % (tN[0], tN[1], tN[2], tN[3]))
def calc_four_situation_neighborhood(tN, y_pred, y_test, r, side):  # side:  side-length of one grid, km
    y_pred = y_pred.reshape(159, 159)
    y_test = y_test.reshape(159, 159)
    # 0: unmarked, 1: hit, 2: miss, 3: false alarm, 4: correct negative
    y_sign = np.zeros((159, 159))
    for i in range(159):
        for j in range(159):
            if y_sign[i][j] != 0:
                continue
            elif y_test[i][j] == 0 and y_pred[i][j] == 0:
                y_sign[i][j] = 4
            elif y_test[i][j] > 0 and y_pred[i][j] > 0:
                y_sign[i][j] = 1
            elif y_test[i][j] > 0 and y_pred[i][j] == 0:
                flag = 0
                for di in range(-(r//side),(r//side)+1):
                    if flag == 1: break
                    for dj in range(-(r//side),(r//side)+1):
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                            continue
                        if ((di*side)**2 + (dj*side)**2) > r**2:
                            continue
                        if y_pred[ii][jj] > 0:
                            y_sign[i][j] = 1
                            flag = 1
                            break
                if flag == 0:
                    y_sign[i][j] = 2
            elif y_pred[i][j] > 0 and y_test[i][j] == 0:
                flag = 0
                for di in range(-(r//side), (r//side)+1):
                    if flag == 1: break
                    for dj in range(-(r//side), (r//side)+1):
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                            continue
                        if ((di*side)**2 + (dj*side)**2) > r**2:
                            continue
                        if y_test[ii][jj] > 0:
                            y_sign[i][j] = 1
                            flag = 1
                            break
                if flag == 0:
                    y_sign[i][j] = 3

    tN[0] = np.sum(y_sign == 1)
    tN[1] = np.sum(y_sign == 3)
    tN[2] = np.sum(y_sign == 2)
    tN[3] = np.sum(y_sign == 4)
    print("N1=%d,  N2=%d,  N3=%d,  N4=%d" % (tN[0], tN[1], tN[2], tN[3]))

def calc_TS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    return tN[0]/(tN[0]+tN[1]+tN[2])
def calc_ETS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    R = (tN[0]+tN[1])*(tN[0]+tN[2])/(tN[0]+tN[1]+tN[2]+tN[3])
    return (tN[0]-R)/((tN[0]+tN[1]+tN[2])-R)
def calc_POD(tN):
    if tN[0] == 0:
        return 0
    return tN[0]/(tN[0]+tN[2])
def calc_FAR(tN):
    if tN[1] == 0:
        return 0
    return tN[1]/(tN[0]+tN[1])
def calc_MAR(tN):
    if tN[2] == 0:
        return 0
    return tN[2]/(tN[0]+tN[2])
def calc_BS(tN):
    if tN[0] + tN[1] + tN[2] == 0:
        return -1
    elif tN[1] != 0 and tN[0]+tN[2] == 0:
        return 0
    return (tN[0]+tN[1])/((tN[0]+tN[2]))
def calc_AC(tN):
    return (tN[0]+tN[3])/sum(tN)
def calc_evaluation(tN, Eval):
    Eval[0] = calc_TS(tN)
    Eval[1] = calc_ETS(tN)
    Eval[2] = calc_POD(tN)
    Eval[3] = calc_FAR(tN)
    Eval[4] = calc_MAR(tN)
    Eval[5] = calc_BS(tN)
    Eval[6] = calc_AC(tN)

def cal_scores(ypred,ytest):
    # convert_to_binary(ypred,1)
    # convert_to_binary(ytest,1)
    tN = np.zeros(4,dtype=float)
    Eval = np.zeros(7,dtype=float)
    calc_four_situation(tN,ypred,ytest)
    calc_evaluation(tN, Eval)
    return tN,Eval
def cal_scores_neighborhood(ypred,ytest,r,side):
    # convert_to_binary(ypred,1)
    # convert_to_binary(ytest,1)
    tN = np.zeros(4,dtype=float)
    Eval = np.zeros(7,dtype=float)
    calc_four_situation_neighborhood(tN,ypred,ytest,r,side)
    calc_evaluation(tN, Eval)
    return tN,Eval

def cal_7_scores_0_6h(st, et, result_path, score_path, threshold, frame_start, frame_end):
    tt = st
    DateTimeList = []
    while (tt < et):
        DateTimeList.append(tt)
        tt += datetime.timedelta(hours=1)
    tot1 = np.zeros(4,dtype=np.float)
    ifile = open(score_path + '7_scores_%d-%dh_t%.2f.txt' % (frame_start,frame_end,threshold), 'w')

    eve_s1 = np.zeros(7, dtype=float)
    n1 = 0

    for ddt in DateTimeList:
        ddt_str = ddt.strftime('%Y%m%d%H%M')
        flag = 0
        truthgrid = np.zeros(159*159,dtype=float)
        predgrid = np.zeros(159*159,dtype=float)
        for hour_plus in range(frame_start,frame_end):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthfilepath = LIG_file_dir + dt_str + '_truth'
            predfilepath = result_path + dt_str + '_h%d' % hour_plus
            if not os.path.exists(predfilepath) or not os.path.exists(truthfilepath):
                flag = 1
                break
            with open(truthfilepath) as tfile:
                truthgrid += np.array(tfile.readlines(), dtype=float)
            with open(predfilepath) as pfile:
                tmp = np.array(pfile.readlines(), dtype=float)
                predgrid += np.round(tmp - (threshold - 0.5))
        if flag == 1: continue

        print('Calculating scores for datetime peroid %s' % (ddt_str))
        tN_1, Eval_1 = cal_scores(predgrid, truthgrid)

        tot1 += tN_1
        eve_1 = np.zeros(7, dtype=float)
        calc_evaluation(tN_1, eve_1)
        if -1 not in eve_1:      # -1 means the denominators of one or more scores are zero
            eve_s1 += eve_1
            n1 += 1

    eve1 = np.zeros(7,dtype=float)
    calc_evaluation(tot1, eve1)

    # accumulate N1-N4 of all test samples and calculate scores one time
    ifile.write('Total(sum):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot1))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve1))
    global_scores_list.extend(eve1[0:5].tolist())
    global_score_formal.extend(eve1[[2, 3, 0, 1]].tolist())
    ifile.write('\n')

    # calculate scores for each time periods
    ifile.write('Total(average):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s1 / (n1+1e-6)))
    ifile.close()

    return eve1

def cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path,score_path , threshold, frame_start, frame_end):
    tt = st
    DateTimeList = []
    while (tt < et):
        DateTimeList.append(tt)
        tt += datetime.timedelta(hours=1)
    tot1 = np.zeros(4,dtype=float)

    ifile = open(score_path + '7_scores_%d-%dh_nbh_r%d_s%d_t%.2f.txt' % (frame_start,frame_end,radius,side_length,threshold), 'w')
    eve_s1 = np.zeros(7, dtype=float)
    n1 = 0
    for ddt in DateTimeList:
        ddt_str = ddt.strftime('%Y%m%d%H%M')
        flag = 0
        truthgrid = np.zeros(159*159,dtype=float)
        predgrid = np.zeros(159*159,dtype=float)
        for hour_plus in range(frame_start, frame_end):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthfilepath = LIG_file_dir + dt_str + '_truth'
            predfilepath = result_path + dt_str + '_h%d' % hour_plus
            if not os.path.exists(predfilepath) or not os.path.exists(truthfilepath):
                flag = 1
                break
            with open(truthfilepath) as tfile:
                truthgrid += np.array(tfile.readlines(), dtype=float)
            with open(predfilepath) as pfile:
                tmp = np.array(pfile.readlines(), dtype=float)
                predgrid += np.round(tmp - (threshold - 0.5))
        if flag == 1: continue
        # print(predgrid)
        print('Calculating scores for datetime peroid %s ' % (ddt_str))
        tN_1, Eval_1 = cal_scores_neighborhood(predgrid,truthgrid, radius, side_length)

        tot1 += tN_1
        eve_1 = np.zeros(7, dtype=float)
        calc_evaluation(tN_1, eve_1)
        if -1 not in eve_1:      # -1 means the denominators of one or more scores are zero
            eve_s1 += eve_1
            n1 += 1

    eve1 = np.zeros(7,dtype=float)
    calc_evaluation(tot1,eve1)

    # accumulate N1-N4 of all test samples and calculate scores one time
    ifile.write('Total(sum):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('N1:%d\tN2:%d\tN3:%d\tN4:%d\n' % tuple(tot1))
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve1))
    global_scores_neig_list.extend(eve1[0:5].tolist())
    global_score_neig_formal.extend(eve1[[2, 3, 0, 1]].tolist())
    ifile.write('\n')

    # calculate scores for each time periods
    ifile.write('Total(average):\n')
    ifile.write('MODEL PRED:\n')
    ifile.write('TS:%f\tETS:%f\tPOD:%f\tFAR:%f\tMAR:%f\tBS:%f\tAC:%f\n' % tuple(eve_s1 / (n1+1e-6)))
    ifile.close()
    return eve1

def eva(model_file, threshold):

    result_path = result_file_dir+model_file+'/'

    radius = 6
    side_length = 4

    st = datetime.datetime(2017, 8, 1, 0, 0, 0, 0)
    et = datetime.datetime(2017, 10, 1, 0, 0, 0, 0)

    score_path = score_file_dir+model_file+'/'

    if not os.path.isdir(score_path):
        os.makedirs(score_path)

    global global_score_formal, global_score_neig_formal
    global_score_formal = []
    global_score_neig_formal = []
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 0, 1)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 0, 3)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 3, 6)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 0, 6)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 0, 1)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 0, 3)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 3, 6)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 0, 6)


    with open(score_path + 'summary_grid_neig' + '.txt', 'w') as f:
        for i in range(4):
            for j in range(4):
                f.write('%.6f\t' % global_score_formal[i*4+j])
            for j in range(4):
                f.write('%.6f\t' % global_score_neig_formal[i*4+j])

def eva_each_hour(model_file, threshold):
    result_path = result_file_dir+model_file+'/'

    radius = 6
    side_length = 4

    st = datetime.datetime(2017, 8, 1, 0, 0, 0, 0)
    et = datetime.datetime(2017, 10, 1, 0, 0, 0, 0)

    score_path = score_file_dir + model_file + '/'

    if not os.path.isdir(score_path):
        os.makedirs(score_path)

    global global_score_formal, global_score_neig_formal
    global_score_formal = []
    global_score_neig_formal = []
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 0, 1)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 1, 2)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 2, 3)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 3, 4)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 4, 5)
    eve = cal_7_scores_0_6h(st, et, result_path, score_path, threshold, 5, 6)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 0, 1)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 1, 2)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 2, 3)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 3, 4)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 4, 5)
    eve = cal_7_scores_0_6h_neighborhood(st, et, radius, side_length, result_path, score_path, threshold, 5, 6)

    num_para = 4
    num_frames = 6
    with open(score_path + 'score_each_hour' + '.txt', 'w') as f:
        for i in range(num_para):
            for j in range(num_frames):
                f.write('%.6f\t' % global_score_formal[j * num_para + i])
            f.write('\n')
    with open(score_path + 'score_each_hour_neig' + '.txt', 'w') as f:
        for i in range(num_para):
            for j in range(num_frames):
                f.write('%.6f\t' % global_score_neig_formal[j*num_para+i])
            f.write('\n')
