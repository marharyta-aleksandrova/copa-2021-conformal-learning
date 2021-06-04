# imports
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory
from nonconformist.nc import InverseProbabilityErrFunc
from nonconformist.nc import MarginErrFunc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sklearn.datasets import load_iris

from sklearn.preprocessing import LabelEncoder

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

"""
Some constants
"""
np.random.seed(seed=1)
num_folds = 10
eps_err = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

"""
Analysis 
"""


def get_fold(fold, idx, test_len, cal_len):
    idx = np.array(idx)
    start_idx = (fold - 1) * test_len
    end_idx = min(fold * test_len, len(idx))
    idx_test = idx[start_idx:end_idx]
    idx_train = idx[0:start_idx].tolist()
    idx_train.extend(idx[end_idx:].tolist())
    idx_cal = idx_train[0:cal_len]
    idx_train = idx_train[cal_len:]
    return np.array(idx_train), np.array(idx_test), np.array(idx_cal)


# get size of training, calibration and testing sets
def get_train_cal_test_len(N, test_frac=0.1, cal_frac=0.2):
    # N - size of the dataset
    # test_frac - fraction of test data from the whole dataset, we'll use test_frac=0.1
    # test_frac depends on the number of folds, test_frac=1/num_folds
    # whole dataset = test_data + all_training_data
    # cal_frac - fraction of calibration data from the all_training_data, we'll use cal_frac=0.2
    # all_training_data = calibration_data + training_data
    test_len = round(N * test_frac)
    all_train_len = N - test_len
    cal_len = round(all_train_len * cal_frac)
    train_len = all_train_len - cal_len
    return (train_len, cal_len, test_len)


# calculating metrics: oneC & avgC
def get_oneC_avgC(prediction):
    arr = np.array(prediction)
    oneC = 0
    avgC = 0
    for i in range(0, len(arr)):
        num_predicted = arr[i].sum()
        avgC += num_predicted
        if num_predicted == 1:
            oneC += 1
        pass
    oneC /= len(arr)
    avgC /= len(arr)
    return oneC, avgC


# calculating metrics: error_rate
def get_accuracy(prediction, real_class):
    correct = 0
    eff_oneC = 0
    N = len(prediction)
    for i in range(0, N):
        if real_class[i] < len(prediction[i]):
            if prediction[i][real_class[i]]:
                correct += 1
            if (prediction[i][real_class[i]]) and (sum(prediction[i]) == 1):
                eff_oneC += 1
        pass
    return correct / N, eff_oneC / N


def filter_class_res(prediction, epsilon):
    result = []
    for line in prediction:
        tmp = copy.deepcopy(line)
        removed = 0
        to_continue = True
        while to_continue:
            min_idx = np.argmin(tmp)
            if tmp[min_idx] + removed <= epsilon:
                removed += tmp[min_idx]
                # removed_idx.append(min_idx)
                tmp[min_idx] = 2.
            else:
                to_continue = False
                pass
            pass
        # now set remove all values set to 2.
        result.append(tmp < 2)
        # print(line)
        pass
    return np.array(result)


def filter_class_res_total(prediction, epsilon):
    tmp = copy.deepcopy(prediction).flatten()
    tot_epsilon = epsilon * len(prediction)
    removed = 0
    to_continue = True
    while to_continue:
        min_idx = np.argmin(tmp)
        if tmp[min_idx] + removed < tot_epsilon:
            removed += tmp[min_idx]
            tmp[min_idx] = 2.
            pass
        else:
            to_continue = False
            pass
        pass
    result = tmp < 2.
    result = result.reshape(prediction.shape)
    return result


def plot_2_df(df_1, df_2, columns, title_str, to_extend=None):
    # model 1
    df = df_1
    color = 'blue'

    if columns[0] is not None:
        plt.plot(eps_err, df[columns[0]], label=df_1.name, color=color, lw=3)  # thick line for original
        plt.plot(eps_err, df[columns[1]], label='- margin', color=color, )  # normal line for margin
        plt.plot(eps_err, df[columns[2]], label='- inv prob', color=color, linestyle='--')  # dashed line for inv_prob
        plt.plot(eps_err, df[columns[3]], label='- simulated', color=color,
                 linestyle=':')  # dotted line for simulation
    else:
        plt.plot(eps_err, df[columns[1]], label=df_1.name + '- margin', color=color, )  # normal line for margin error
        plt.plot(eps_err, df[columns[2]], label='- inv prob', color=color,
                 linestyle='--')  # dashed line for inv_prob error
        plt.plot(eps_err, df[columns[3]], label='- simulated', color=color,
                 linestyle=':')  # dotted line for simulation error

    # model 2
    df = df_2
    color = 'orange'

    if columns[0] is not None:
        plt.plot(eps_err, df[columns[0]], label=df_2.name, color=color, lw=3)  # thik line for original
        plt.plot(eps_err, df[columns[1]], label='_nolegend_', color=color, )  # normal line for margin
        plt.plot(eps_err, df[columns[2]], label='_nolegend_', color=color, linestyle='--')  # dashed line for inv_prob
        plt.plot(eps_err, df[columns[3]], label='_nolegend_', color=color, linestyle=':')  # dotted line for simulation
    else:
        plt.plot(eps_err, df[columns[1]], label=df_2.name, color=color, )  # normal line for margin error
        plt.plot(eps_err, df[columns[2]], label='_nolegend_', color=color,
                 linestyle='--')  # dashed line for inv_prob error
        plt.plot(eps_err, df[columns[3]], label='_nolegend_', color=color,
                 linestyle=':')  # dotted line for simulation error

    plt.xticks(eps_err)
    if columns[0] is not None:
        y_ticks = eps_err[:]
        if to_extend is not None:
            y_ticks.extend(to_extend)
        plt.yticks(y_ticks)
    plt.grid(True)
    plt.legend()
    plt.title(title_str)
    plt.xlabel('Error, $\epsilon$')
    plt.ylabel(title_str)
    pass


"""
Data loading
"""


def load_wine_red():
    print('Loading wine_red')
    # load red wine dataset
    file = 'datasets/winequality-red.csv'
    df = pd.read_csv(file, sep=';')
    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    target = df[df.columns[-1]].values
    target -= min(target)
    data = {'target': df[df.columns[-1]].values, 'data': df[df.columns[:-1]].values,
            'target_names': target_names}
    return data


def load_wine_white():
    print('Loading wine_white')
    # load red wine dataset
    file = 'datasets/winequality-white.csv'
    df = pd.read_csv(file, sep=';')
    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    target = df[df.columns[-1]].values
    target -= min(target)
    data = {'target': df[df.columns[-1]].values, 'data': df[df.columns[:-1]].values,
            'target_names': target_names}
    return data


def load_ecoli():
    print('Loading ecoli')
    file = 'datasets/ecoli.csv'
    df = pd.read_csv(file, sep=',', header=None)
    df.drop(0, axis=1, inplace=True)
    # print(df.dtypes)
    target_names = df[df.columns[-1]].unique()
    target_names.sort()

    target = df[df.columns[-1]].values
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)

    data = {'target': target, 'data': df[df.columns[:-1]].values,
            'target_names': target_names}
    return data


def load_balance():
    print('Loading balance')
    # load balance dataset
    file = 'datasets/balance.csv'
    df = pd.read_csv(file, header=None, sep=',')
    le = LabelEncoder()
    le.fit(df[0].values)
    target = le.transform(df[0].values)
    data = {'target': target, 'data': df[df.columns[1:]].values,
            'target_names': le.classes_}
    return data


def load_cars():
    print('Loading cars')
    # load cars dataset
    file = 'datasets/cars.csv'
    df = pd.read_csv(file, header=None, sep=',')
    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    data_points = df[df.columns[:-1]]
    target = df[df.columns[-1]].values
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)
    encoded_df = pd.DataFrame({})
    for i in range(6):
        encoded_df[i] = label_encoder.fit_transform(data_points[i])
    data = {'target': target, 'data': encoded_df.values,
            'target_names': target_names}
    return data


def load_glass():
    print('Loading glass')
    file = 'datasets/glass.csv'
    df = pd.read_csv(file, sep=',', header=None)
    df.drop(0, axis=1, inplace=True)

    target_names = df[df.columns[-1]].unique()
    target_names.sort()

    target = df[df.columns[-1]].values
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)

    data = {'target': target, 'data': df[df.columns[:-1]].values,
            'target_names': target_names}
    return data


def load_user():
    print('Loading user')
    # load user dataset
    file = 'datasets/user.csv'
    df = pd.read_csv(file, header=0, sep='\t')
    target_names = df.iloc[:, -1].unique()
    target_names.sort()

    target = df.iloc[:, -1].values
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)
    data_points = df.iloc[:, 0:-1]
    data = {'target': target, 'data': data_points.values,
            'target_names': target_names}
    return data


def load_wave():
    print('Loading wave')
    # load wave dataset
    file = 'datasets/wave.csv'
    df = pd.read_csv(file, header=None, sep=',')
    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    target = df[df.columns[-1]].values
    data = {'target': target, 'data': df[df.columns[:-1]].values, 'target_names': target_names}
    return data


def load_wine():
    print('Loading wine')
    # load wne dataset
    file = 'datasets/wine.csv'
    df = pd.read_csv(file, header=None, sep=',')
    target_names = df.iloc[:, 0].unique()
    target_names.sort()
    target = df.iloc[:, 0].values
    data = {'target': target, 'data': df.iloc[:, 1:-1].values, 'target_names': target_names}
    return data


def load_yeast():
    # cols_to_exclude = ['ERL', 'POX', 'VAC']
    cols_to_exclude = None
    print('Loading yeast')
    # load yeast dataset
    file = 'datasets/yeast.csv'

    df = pd.read_csv(file, header=None, sep=' ')
    df.drop(columns=[0], inplace=True)

    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    target = df[df.columns[-1]].values
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)
    data = {'target': target, 'data': df[df.columns[:-1]].values,
            'target_names': target_names}
    return data


def load_zoo():
    # cols_to_exclude = [3, 4, 5, 6, 7]
    cols_to_exclude = None
    print('Loading zoo')
    # load ecoli dataset
    file = 'datasets/zoo.csv'
    df = pd.read_csv(file, header=None)
    df.drop(columns=[0], inplace=True)

    target_names = df[df.columns[-1]].unique()
    target_names.sort()
    target = df[df.columns[-1]].values
    # target_encoder = LabelEncoder()
    # target = target_encoder.fit_transform(target)

    data = {'target': target, 'data': df[df.columns[0:-1]].values,
            'target_names': target_names}
    return data


def plotClasses(df):
    cols = ["col" + str(x) for x in range(1, len(df.columns))]
    cols.append("target")
    df.columns = cols  # add column names to the data frame
    counts = df.groupby(by="target").count()[['col1']]
    labels = counts.index.to_list()
    data_points = counts.values.reshape(len(counts), )
    return plt.bar(labels, data_points)


"""
This is the main function, runs conformal prediction
"""


def run_conformal_inv_prob_margin_3_in_1(data, epsilon_arr, model_obj=SVC, params={'probability': True},
                                         smoothing=False, seed=1, verbose=True, num_folds=10,
                                         min_max_mode=0):
    """
    min_max_mode = 0 -> min + max
                 = 1 -> min + 1 rand
                 = 2 -> min + 0 rand
                 = 3 -> min only if p_val >= (1- eps)
    """
    np.random.seed(seed=seed)

    data_size = len(data['target'])
    idx = np.random.permutation(data_size)
    train_len, cal_len, test_len = get_train_cal_test_len(data_size, 0.1, 0.2)

    if verbose:
        print('Info about dataset: ')
        print('classes: {}'.format(list(data['target_names'])))
        print('size of the dataset: {}'.format(len(data['target'])))
        print('train = {}, cal = {}, test = {}'.format(train_len, cal_len, test_len))
        pass

    # define arrays to store information in
    res_epsilon_arr = []
    fold_arr = []
    # error rate for normal SVM
    error_rate_trad_arr = []
    # info for conformal learning
    # inv_prob
    ip_oneC_arr = []
    ip_eff_oneC_arr = []
    ip_avgC_arr = []
    ip_error_rate_arr = []
    # margin
    m_oneC_arr = []
    m_eff_oneC_arr = []
    m_avgC_arr = []
    m_error_rate_arr = []
    # inv_prob + margin
    ip_m_oneC_arr = []
    ip_m_eff_oneC_arr = []
    ip_m_avgC_arr = []
    ip_m_error_rate_arr = []
    # info for simulation of conformal learning (exclude labels until you get 1-eps total probability)
    svm_oneC_arr = []
    svm_eff_oneC_arr = []
    svm_avgC_arr = []
    svm_error_rate_arr = []

    for fold in range(1, num_folds + 1):
        idx_train, idx_test, idx_cal = get_fold(fold, idx, test_len, cal_len)
        # print('effective training set')
        # print(idx_train[:10])

        # first run normal SVM
        # put train and calibration together
        idx_all_train = np.concatenate((idx_train, idx_cal))
        clf = model_obj(**params)
        clf.fit(data['data'][idx_all_train, :], data['target'][idx_all_train])
        prediction = clf.predict(data['data'][idx_test, :])
        error_rate_svm = (data['target'][idx_test] != prediction).sum() / len(prediction)

        # now for different values of eps do conformal prediction

        # build conformal model
        model = model_obj(**params)  # Create the underlying model

        nc_inv_prob = NcFactory.create_nc(model,
                                          err_func=InverseProbabilityErrFunc())  # specify non-conformity function
        icp_inv_prob = IcpClassifier(nc_inv_prob, smoothing=smoothing)
        # Fit the ICP using the proper training set
        icp_inv_prob.fit(data['data'][idx_train, :], data['target'][idx_train])
        # Calibrate the ICP using the calibration set
        icp_inv_prob.calibrate(data['data'][idx_cal, :], data['target'][idx_cal])
        p_vals_inv_prob = icp_inv_prob.predict(data['data'][idx_test, :])

        nc_margin = NcFactory.create_nc(model, err_func=MarginErrFunc())  # specify non-conformity function
        icp_margin = IcpClassifier(nc_margin, smoothing=smoothing)
        # Fit the ICP using the proper training set
        icp_margin.fit(data['data'][idx_train, :], data['target'][idx_train])
        # Calibrate the ICP using the calibration set
        icp_margin.calibrate(data['data'][idx_cal, :], data['target'][idx_cal])
        p_vals_margin = icp_margin.predict(data['data'][idx_test, :])

        for i in range(0, len(epsilon_arr)):
            epsilon = epsilon_arr[i]
            # save current info to arrays
            fold_arr.append(fold)
            res_epsilon_arr.append(epsilon)
            # save info about traditional SVM error rate
            error_rate_trad_arr.append(error_rate_svm)
            # run adjusted conformal prediction: (exclude labels until you get 1-eps total probability)
            # see function filter_class_res
            prediction_svm_prob = clf.predict_proba(data['data'][idx_test, :])
            prediction_svm = filter_class_res(prediction_svm_prob, epsilon)
            oneC, avgC = get_oneC_avgC(prediction_svm)
            real_class = data['target'][idx_test]
            accuracy, eff_oneC = get_accuracy(prediction_svm, real_class)
            svm_eff_oneC_arr.append(eff_oneC)
            error_rate = 1 - accuracy
            svm_oneC_arr.append(oneC)
            svm_avgC_arr.append(avgC)
            svm_error_rate_arr.append(error_rate)
            # run conformal prediction itself
            # Produce predictions for the test set
            prediction_margin = p_vals_margin > epsilon / 2
            prediction_margin_original = p_vals_margin > epsilon
            prediction_inv_prob = p_vals_inv_prob > epsilon
            prediction = []
            if min_max_mode == 0:
                num_pred_mar = prediction_margin.sum(axis=1)
                idx_one = np.where(num_pred_mar == 1)[0]
                idx_max = np.argsort(num_pred_mar)[-len(idx_one):]
                idx_margin = np.concatenate([idx_one, idx_max])
                for i in range(0, len(prediction_margin)):
                    if i in idx_margin:
                        prediction.append(prediction_margin[i])
                    else:
                        prediction.append(prediction_inv_prob[i])
                    pass
                pass
            elif min_max_mode == 1:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
                        prediction.append(prediction_margin[i])
                        if i + 1 < len(prediction_margin):
                            prediction.append(prediction_margin[i + 1])
                        i += 2
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass
            elif min_max_mode == 2:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
                        prediction.append(prediction_margin[i])
                        i += 1
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass
            elif min_max_mode == 3:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1) and (
                            max(p_vals_margin[i]) >= 1 - epsilon):
                        prediction.append(prediction_margin[i])
                        i += 1
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass

            # for i in range(0, len(prediction_margin)):
            #    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
            #        prediction.append(prediction_margin[i])
            #    else:
            #        prediction.append(prediction_inv_prob[i])
            oneC, avgC = get_oneC_avgC(prediction)
            accuracy, eff_oneC = get_accuracy(prediction, real_class)
            error_rate = 1 - accuracy
            ip_m_eff_oneC_arr.append(eff_oneC)
            ip_m_oneC_arr.append(oneC)
            ip_m_avgC_arr.append(avgC)
            ip_m_error_rate_arr.append(error_rate)
            # inv_prob
            oneC, avgC = get_oneC_avgC(prediction_inv_prob)
            accuracy, eff_oneC = get_accuracy(prediction_inv_prob, real_class)
            error_rate = 1 - accuracy
            ip_eff_oneC_arr.append(eff_oneC)
            ip_oneC_arr.append(oneC)
            ip_avgC_arr.append(avgC)
            ip_error_rate_arr.append(error_rate)
            # margin_prob
            oneC, avgC = get_oneC_avgC(prediction_margin_original)
            accuracy, eff_oneC = get_accuracy(prediction_margin_original, real_class)
            error_rate = 1 - accuracy
            m_eff_oneC_arr.append(eff_oneC)
            m_oneC_arr.append(oneC)
            m_avgC_arr.append(avgC)
            m_error_rate_arr.append(error_rate)
            pass
        pass

    df_res = pd.DataFrame({})
    # save data into a data frame
    df_res['eps'] = res_epsilon_arr
    df_res['fold'] = fold_arr
    df_res['origin_err'] = error_rate_trad_arr

    # info for conformal learning: margin
    df_res['marg_err'] = m_error_rate_arr
    df_res['marg_oneC'] = m_oneC_arr
    df_res['marg_eff_oneC'] = m_eff_oneC_arr
    df_res['marg_avgC'] = m_avgC_arr

    # info for conformal learning: ip
    df_res['inv_err'] = ip_error_rate_arr
    df_res['inv_oneC'] = ip_oneC_arr
    df_res['inv_eff_oneC'] = ip_eff_oneC_arr
    df_res['inv_avgC'] = ip_avgC_arr

    # info for conformal learning: margin + ip
    df_res['inv_m_err'] = ip_m_error_rate_arr
    df_res['inv_m_oneC'] = ip_m_oneC_arr
    df_res['inv_m_eff_oneC'] = ip_m_eff_oneC_arr
    df_res['inv_m_avgC'] = ip_m_avgC_arr

    # info for simulation of conformal learning
    df_res['Simul_err'] = svm_error_rate_arr
    df_res['Simul_oneC'] = svm_oneC_arr
    df_res['Simul_eff_oneC'] = svm_eff_oneC_arr
    df_res['Simul_avgC'] = svm_avgC_arr
    # reorder
    df_res.sort_values(by=['eps', 'fold'], inplace=True)
    # df_res.reset_index(inplace=True)

    return df_res


def run_conformal_inv_prob_margin(data, epsilon_arr, model_obj=SVC, params={'probability': True},
                                  err_func=MarginErrFunc, smoothing=False, seed=1, verbose=True, num_folds=10,
                                  min_max_mode=0):
    """
    min_max_mode = 0 -> min + max
                 = 1 -> min + 1 rand
                 = 2 -> min + 0 rand
                 = 3 -> min only if p_val >= (1- eps)
    """
    np.random.seed(seed=seed)

    data_size = len(data['target'])
    idx = np.random.permutation(data_size)
    train_len, cal_len, test_len = get_train_cal_test_len(data_size, 0.1, 0.2)

    if verbose:
        print('Info about dataset: ')
        print('classes: {}'.format(list(data['target_names'])))
        print('size of the dataset: {}'.format(len(data['target'])))
        print('train = {}, cal = {}, test = {}'.format(train_len, cal_len, test_len))
        pass

    # define arrays to store information in
    res_epsilon_arr = []
    fold_arr = []
    # error rate for normal SVM
    error_rate_trad_arr = []
    # info for conformal learning
    oneC_arr = []
    eff_oneC_arr = []
    avgC_arr = []
    error_rate_arr = []
    # info for simulation of conformal learning (exclude labels until you get 1-eps total probability)
    oneC_svm_arr = []
    eff_oneC_svm_arr = []
    avgC_svm_arr = []
    error_rate_svm_arr = []

    for fold in range(1, num_folds + 1):
        idx_train, idx_test, idx_cal = get_fold(fold, idx, test_len, cal_len)
        # print('effective training set')
        # print(idx_train[:10])

        # first run normal SVM
        # put train and calibration together
        idx_all_train = np.concatenate((idx_train, idx_cal))
        clf = model_obj(**params)
        clf.fit(data['data'][idx_all_train, :], data['target'][idx_all_train])
        prediction = clf.predict(data['data'][idx_test, :])
        error_rate_svm = (data['target'][idx_test] != prediction).sum() / len(prediction)

        # now for different values of eps do conformal prediction

        # build conformal model
        model = model_obj(**params)  # Create the underlying model

        nc_inv_prob = NcFactory.create_nc(model,
                                          err_func=InverseProbabilityErrFunc())  # specify non-conformity function
        icp_inv_prob = IcpClassifier(nc_inv_prob, smoothing=smoothing)
        # Fit the ICP using the proper training set
        icp_inv_prob.fit(data['data'][idx_train, :], data['target'][idx_train])
        # Calibrate the ICP using the calibration set
        icp_inv_prob.calibrate(data['data'][idx_cal, :], data['target'][idx_cal])
        p_vals_inv_prob = icp_inv_prob.predict(data['data'][idx_test, :])

        nc_margin = NcFactory.create_nc(model, err_func=MarginErrFunc())  # specify non-conformity function
        icp_margin = IcpClassifier(nc_margin, smoothing=smoothing)
        # Fit the ICP using the proper training set
        icp_margin.fit(data['data'][idx_train, :], data['target'][idx_train])
        # Calibrate the ICP using the calibration set
        icp_margin.calibrate(data['data'][idx_cal, :], data['target'][idx_cal])
        p_vals_margin = icp_margin.predict(data['data'][idx_test, :])

        for i in range(0, len(epsilon_arr)):
            epsilon = epsilon_arr[i]
            # save current info to arrays
            fold_arr.append(fold)
            res_epsilon_arr.append(epsilon)
            # save info about traditional SVM error rate
            error_rate_trad_arr.append(error_rate_svm)
            # run adjusted conformal prediction: (exclude labels until you get 1-eps total probability)
            # see function filter_class_res
            prediction_svm_prob = clf.predict_proba(data['data'][idx_test, :])
            prediction_svm = filter_class_res(prediction_svm_prob, epsilon)
            oneC, avgC = get_oneC_avgC(prediction_svm)
            real_class = data['target'][idx_test]
            accuracy, eff_oneC = get_accuracy(prediction_svm, real_class)
            eff_oneC_arr.append(eff_oneC)
            error_rate = 1 - accuracy
            oneC_svm_arr.append(oneC)
            avgC_svm_arr.append(avgC)
            error_rate_svm_arr.append(error_rate)
            # run conformal prediction itself
            # Produce predictions for the test set
            prediction_margin = p_vals_margin > epsilon / 2
            prediction_inv_prob = p_vals_inv_prob > epsilon
            prediction = []
            if min_max_mode == 0:
                num_pred_mar = prediction_margin.sum(axis=1)
                idx_one = np.where(num_pred_mar == 1)[0]
                idx_max = np.argsort(num_pred_mar)[-len(idx_one):]
                idx_margin = np.concatenate([idx_one, idx_max])
                for i in range(0, len(prediction_margin)):
                    if i in idx_margin:
                        prediction.append(prediction_margin[i])
                    else:
                        prediction.append(prediction_inv_prob[i])
                    pass
                pass
            elif min_max_mode == 1:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
                        prediction.append(prediction_margin[i])
                        if i + 1 < len(prediction_margin):
                            prediction.append(prediction_margin[i + 1])
                        i += 2
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass
            elif min_max_mode == 2:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
                        prediction.append(prediction_margin[i])
                        i += 1
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass
            elif min_max_mode == 3:
                i = 0
                while i < len(prediction_margin):
                    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1) and (
                            max(p_vals_margin[i]) >= 1 - epsilon):
                        prediction.append(prediction_margin[i])
                        i += 1
                    else:
                        prediction.append(prediction_inv_prob[i])
                        i += 1
                    pass
                pass

            # for i in range(0, len(prediction_margin)):
            #    if (sum(prediction_margin[i]) == 1) and (sum(prediction_inv_prob[i]) > 1):
            #        prediction.append(prediction_margin[i])
            #    else:
            #        prediction.append(prediction_inv_prob[i])
            oneC, avgC = get_oneC_avgC(prediction)
            accuracy, eff_oneC = get_accuracy(prediction, real_class)
            eff_oneC_svm_arr.append(eff_oneC)
            error_rate = 1 - accuracy
            oneC_arr.append(oneC)
            avgC_arr.append(avgC)
            error_rate_arr.append(error_rate)
            pass
        pass

    df_res = pd.DataFrame({})
    # save data into a data frame
    df_res['eps'] = res_epsilon_arr
    df_res['fold'] = fold_arr
    df_res['origin_err'] = error_rate_trad_arr
    # info for conformal learning
    df_res['Conf_err'] = error_rate_arr
    df_res['Conf_oneC'] = oneC_arr
    df_res['Conf_eff_oneC'] = eff_oneC_svm_arr
    df_res['Conf_avgC'] = avgC_arr
    # info for simulation of conformal learning
    df_res['Simul_err'] = error_rate_svm_arr
    df_res['Simul_oneC'] = oneC_svm_arr
    df_res['Simul_eff_oneC'] = eff_oneC_arr
    df_res['Simul_avgC'] = avgC_svm_arr
    # reorder
    df_res.sort_values(by=['eps', 'fold'], inplace=True)
    # df_res.reset_index(inplace=True)

    return df_res


def run_conformal(data, epsilon_arr, model_obj=SVC, params={'probability': True},
                  err_func=MarginErrFunc(), smoothing=False, seed=1, verbose=True, num_folds=10):
    np.random.seed(seed=seed)

    data_size = len(data['target'])
    idx = np.random.permutation(data_size)
    train_len, cal_len, test_len = get_train_cal_test_len(data_size, 0.1, 0.2)

    if verbose:
        print('Info about dataset: ')
        print('classes: {}'.format(list(data['target_names'])))
        print('size of the dataset: {}'.format(len(data['target'])))
        print('train = {}, cal = {}, test = {}'.format(train_len, cal_len, test_len))
        pass

    # define arrays to store information in
    res_epsilon_arr = []
    fold_arr = []
    # error rate for normal SVM
    error_rate_trad_arr = []
    # info for conformal learning
    oneC_arr = []
    eff_oneC_arr = []
    avgC_arr = []
    error_rate_arr = []
    # info for simulation of conformal learning (exclude labels until you get 1-eps total probability)
    oneC_svm_arr = []
    eff_oneC_svm_arr = []
    avgC_svm_arr = []
    error_rate_svm_arr = []

    for fold in range(1, num_folds + 1):
        idx_train, idx_test, idx_cal = get_fold(fold, idx, test_len, cal_len)
        # print('effective training set')
        # print(idx_train[:10])

        # first run normal SVM
        # put train and calibration together
        idx_all_train = np.concatenate((idx_train, idx_cal))
        clf = model_obj(**params)
        clf.fit(data['data'][idx_all_train, :], data['target'][idx_all_train])
        prediction = clf.predict(data['data'][idx_test, :])
        error_rate_svm = (data['target'][idx_test] != prediction).sum() / len(prediction)

        # now for different values of eps do conformal prediction

        # build conformal model
        model = model_obj(**params)  # Create the underlying model
        nc = NcFactory.create_nc(model, err_func=err_func)  # specify non-conformity function
        icp = IcpClassifier(nc, smoothing=smoothing)
        # Fit the ICP using the proper training set
        icp.fit(data['data'][idx_train, :], data['target'][idx_train])
        # Calibrate the ICP using the calibration set
        icp.calibrate(data['data'][idx_cal, :], data['target'][idx_cal])
        p_vals = icp.predict(data['data'][idx_test, :])

        for i in range(0, len(epsilon_arr)):
            epsilon = epsilon_arr[i]
            # save current info to arrays
            fold_arr.append(fold)
            res_epsilon_arr.append(epsilon)
            # save info about traditional SVM error rate
            error_rate_trad_arr.append(error_rate_svm)
            # run adjusted conformal prediction: (exclude labels until you get 1-eps total probability)
            # see function filter_class_res
            prediction_svm_prob = clf.predict_proba(data['data'][idx_test, :])
            prediction_svm = filter_class_res(prediction_svm_prob, epsilon)
            oneC, avgC = get_oneC_avgC(prediction_svm)
            real_class = data['target'][idx_test]
            accuracy, eff_oneC = get_accuracy(prediction_svm, real_class)
            eff_oneC_svm_arr.append(eff_oneC)
            error_rate = 1 - accuracy
            oneC_svm_arr.append(oneC)
            avgC_svm_arr.append(avgC)
            error_rate_svm_arr.append(error_rate)
            # run conformal prediction itself
            # Produce predictions for the test set
            prediction = p_vals > epsilon
            oneC, avgC = get_oneC_avgC(prediction)
            accuracy, eff_oneC = get_accuracy(prediction, real_class)
            eff_oneC_arr.append(eff_oneC)
            error_rate = 1 - accuracy
            oneC_arr.append(oneC)
            avgC_arr.append(avgC)
            error_rate_arr.append(error_rate)
            pass
        pass

    df_res = pd.DataFrame({})
    # save data into a data frame
    df_res['eps'] = res_epsilon_arr
    df_res['fold'] = fold_arr
    df_res['origin_err'] = error_rate_trad_arr
    # info for conformal learning
    df_res['Conf_err'] = error_rate_arr
    df_res['Conf_oneC'] = oneC_arr
    df_res['Conf_eff_oneC'] = eff_oneC_arr
    df_res['Conf_avgC'] = avgC_arr
    # info for simulation of conformal learning
    df_res['Simul_err'] = error_rate_svm_arr
    df_res['Simul_oneC'] = oneC_svm_arr
    df_res['Simul_eff_oneC'] = eff_oneC_svm_arr
    df_res['Simul_avgC'] = avgC_svm_arr
    # reorder
    df_res.sort_values(by=['eps', 'fold'], inplace=True)
    # df_res.reset_index(inplace=True)

    return df_res


"""
And analysis of conformal prediction with 2 error functions
"""


def analysis_with_3_func_in_1(data, model_obj, params, epsilon_arr, smoothing=True, seed=1, sub_verbose=True):
    combination_res_df = run_conformal_inv_prob_margin_3_in_1(data, model_obj=model_obj, params=params,
                                                              epsilon_arr=eps_err,
                                                              smoothing=False, seed=seed, verbose=False,
                                                              min_max_mode=2, )
    model_mean_df = combination_res_df.groupby('eps').mean()

    return model_mean_df


def analysis_with_3_func(data, model_obj, params, epsilon_arr, smoothing=True, seed=1, sub_verbose=True):
    margin_res_df = run_conformal(data=data, model_obj=model_obj, params=params,
                                  epsilon_arr=epsilon_arr, err_func=MarginErrFunc(),
                                  smoothing=smoothing, seed=seed, verbose=sub_verbose)
    model_mean_df = margin_res_df.groupby('eps').mean()
    # now inverse probability error func

    inverse_res_df = run_conformal(data=data, model_obj=model_obj, params=params,
                                   epsilon_arr=epsilon_arr, err_func=InverseProbabilityErrFunc(),
                                   smoothing=smoothing, seed=seed, verbose=False)
    tmp_df = inverse_res_df.groupby('eps').mean()
    # rename columns for margin function
    model_mean_df.columns = ['fold', 'origin_err', 'marg_err', 'marg_oneC', 'marg_eff_oneC',
                             'marg_avgC', 'Simul_err', 'Simul_oneC', 'Simul_eff_oneC', 'Simul_avgC']
    # these columns contain results for inverse probability non-conformity function
    model_mean_df['inv_err'] = tmp_df['Conf_err']
    model_mean_df['inv_oneC'] = tmp_df['Conf_oneC']
    model_mean_df['inv_eff_oneC'] = tmp_df['Conf_eff_oneC']
    model_mean_df['inv_avgC'] = tmp_df['Conf_avgC']

    combination_res_df = run_conformal_inv_prob_margin(data, model_obj=model_obj, params=params, epsilon_arr=eps_err,
                                                       err_func=None, smoothing=False, seed=seed, verbose=False,
                                                       min_max_mode=2, )
    tmp_df = combination_res_df.groupby('eps').mean()

    # these columns contain results for the combination of non-conformity functions
    model_mean_df['inv_m_err'] = tmp_df['Conf_err']
    model_mean_df['inv_m_oneC'] = tmp_df['Conf_oneC']
    model_mean_df['inv_m_eff_oneC'] = tmp_df['Conf_eff_oneC']
    model_mean_df['inv_m_avgC'] = tmp_df['Conf_avgC']
    # model_mean_df
    return model_mean_df


def analysis_with_2_func(data, model_obj, params, epsilon_arr, smoothing=True, seed=1, sub_verbose=True):
    margin_res_df = run_conformal(data=data, model_obj=model_obj, params=params,
                                  epsilon_arr=epsilon_arr, err_func=MarginErrFunc(),
                                  smoothing=smoothing, seed=seed, verbose=sub_verbose)
    model_mean_df = margin_res_df.groupby('eps').mean()
    # now inverse probability error func

    inverse_res_df = run_conformal(data=data, model_obj=model_obj, params=params,
                                   epsilon_arr=epsilon_arr, err_func=InverseProbabilityErrFunc(),
                                   smoothing=smoothing, seed=seed, verbose=False)
    tmp_df = inverse_res_df.groupby('eps').mean()
    # rename columns for margin function
    model_mean_df.columns = ['fold', 'origin_err', 'marg_err', 'marg_oneC', 'marg_eff_oneC',
                             'marg_avgC', 'Simul_err', 'Simul_oneC', 'Simul_eff_oneC', 'Simul_avgC']
    # these columns contain results for inverse probability non-conformity function
    model_mean_df['inv_err'] = tmp_df['Conf_err']
    model_mean_df['inv_oneC'] = tmp_df['Conf_oneC']
    model_mean_df['inv_eff_oneC'] = tmp_df['Conf_eff_oneC']
    model_mean_df['inv_avgC'] = tmp_df['Conf_avgC']
    # model_mean_df
    return model_mean_df


"""
Inputs for models
"""

# SVM
svm_model_obj = SVC
svm_params = {'probability': True}

# Decision tree
dt_model_obj = DecisionTreeClassifier
dt_min_sample_ratio = 0.05 * 0.9 * 0.8  # (5% of the effective learning set)
dt_params = {}

# KNN
k_kneighbors_model_obj = KNeighborsClassifier
k_kneighbors_params = {"n_neighbors": 5}

# Ada Boost
ada_boost_model_obj = AdaBoostClassifier
ada_boost_params = {}

# Gaussian NB
gaussian_nb_model_obj = GaussianNB
gaussian_nb_params = {}

# MPL 
mlp_classifier_model_obj = MLPClassifier
mlp_classifier_params = {"alpha": 1, "max_iter": 1000}

# Random Forest
random_forest_model_obj = RandomForestClassifier
MIN_SAMPLES_KEY = 'min_samples_split'
random_forest_params = {MIN_SAMPLES_KEY: 0, 'n_estimators': 10}

# Gaussian process
gaussian_process_obj = GaussianProcessClassifier
gaussian_process_params = {}

# QuadraticDiscriminantAnalysis
quadratic_discriminant_model_obj = QuadraticDiscriminantAnalysis
quadratic_discriminant_params = {}

"""
Plotting
"""
# constants
data_names = ['balance', 'cars', 'ecoli', 'glass', 'iris', 'user', 'wave', 'wine',
              'wine_Red', 'wine_White', 'yeast', 'zoo']

classifiers_names = ['SVM', 'DT', 'KNN', 'Ada', 'GNB', 'MPR', 'RF', 'GPr.', 'QDA']
file_name_template = 'analysis-results/{}_{}.csv'
colours = ['magenta', 'orange', 'brown', 'green', 'tomato', 'black', 'blue', 'grey', 'red']


# loading dataframes with results
def load_results(pos):
    dataset_name = data_names[pos]
    all_results = []
    for class_name in classifiers_names:
        file_name = file_name_template.format(dataset_name, class_name)
        # load dataframe from csv
        tmp = pd.read_csv(file_name)
        tmp.set_index('eps', inplace=True)
        # try:
        #    tmp.set_index('eps', inplace=True)
        # except:
        #    pass
        tmp.name = class_name
        all_results.append(tmp)
    return all_results


def plot_all_errors(all_results, dataset):
    all_errors = []
    for i in range(0, len(all_results)):
        all_errors.append(all_results[i]['origin_err'].min())
        pass
    x = np.arange(len(all_errors))
    plt.bar(x, all_errors, color=colours)
    plt.xticks(x, classifiers_names)
    plt.grid(True)
    plt.title('Baseline error')
    plt.savefig('figures/{}-all-errors.pdf'.format(dataset), pad_inches=0, bbox_inches='tight')
    pass


def plot_results(all_results, col, dataset):
    legend_arr = []

    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    min_val = 10 ** 6
    max_val = 0
    for i in range(0, 9):
        df = all_results[i]
        ax_pos = int(i / 3)
        axs[ax_pos].plot(df['marg_{}'.format(col)], c=colours[i])
        legend_arr.append(df.name)
        axs[ax_pos].plot(df['inv_{}'.format(col)], '--', c=colours[i])
        legend_arr.append('_nolegend_')

        max_tmp = df['marg_{}'.format(col)].max()
        min_tmp = df['marg_{}'.format(col)].min()
        if max_tmp > max_val:
            max_val = max_tmp
        if min_tmp < min_val:
            min_val = min_tmp
        max_tmp = df['inv_{}'.format(col)].max()
        min_tmp = df['inv_{}'.format(col)].min()
        if max_tmp > max_val:
            max_val = max_tmp
        if min_tmp < min_val:
            min_val = min_tmp
        pass
    for i in range(0, 3):
        plt.sca(axs[i])
        plt.xticks(eps_err)
        if col == 'err':
            plt.yticks(eps_err)
        plt.grid(True)
        plt.xlabel('error rate, $\epsilon$')
        plt.ylabel('${}$'.format(col))
        plt.legend(legend_arr[i * 6: i * 6 + 6])
        plt.ylim([min_val, max_val])
        pass

    fig.suptitle('${}$, solid - margin, dashed - inv.prob'.format(col))
    plt.savefig('figures/{}-{}.pdf'.format(dataset, col), pad_inches=0, bbox_inches='tight')
    pass


def plot_results_vertical(all_results, col, dataset):
    legend_arr = []

    fig, axs = plt.subplots(3, 1, figsize=(5, 12))

    min_val = 10 ** 6
    max_val = 0
    for i in range(0, 9):
        df = all_results[i]
        ax_pos = int(i / 3)
        axs[ax_pos].plot(df['marg_{}'.format(col)], c=colours[i])
        legend_arr.append(df.name)
        axs[ax_pos].plot(df['inv_{}'.format(col)], '--', c=colours[i])
        legend_arr.append('_nolegend_')

        max_tmp = df['marg_{}'.format(col)].max()
        min_tmp = df['marg_{}'.format(col)].min()
        if max_tmp > max_val:
            max_val = max_tmp
        if min_tmp < min_val:
            min_val = min_tmp
        max_tmp = df['inv_{}'.format(col)].max()
        min_tmp = df['inv_{}'.format(col)].min()
        if max_tmp > max_val:
            max_val = max_tmp
        if min_tmp < min_val:
            min_val = min_tmp
        pass
    for i in range(0, 3):
        plt.sca(axs[i])
        plt.xticks(eps_err)
        if col == 'err':
            plt.yticks(eps_err)
        plt.grid(True)
        plt.xlabel('error rate, $\epsilon$')
        plt.ylabel('${}$'.format(col))
        plt.legend(legend_arr[i * 6: i * 6 + 6])
        plt.ylim([min_val, max_val])
        pass

    # fig.suptitle('${}$, solid - margin, dashed - inv.prob'.format(col))
    plt.savefig('figures/{}-{}.pdf'.format(dataset, col), pad_inches=0, bbox_inches='tight')
    pass


datasets_load = {'balance': load_balance, 'cars': load_cars, 'ecoli': load_ecoli, 'glass': load_glass,
                 'iris': load_iris, 'user': load_user, 'wave': load_wave, 'wine': load_wine,
                 'wine_Red': load_wine_red, 'wine_White': load_wine_white, 'yeast': load_yeast, 'zoo': load_zoo
                 }
