import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import shapiro


def get_baseline_err(algo_arr, results, eps_err):
    baseline_err_arr = []
    for algo in algo_arr:
        df = results[algo]
        baseline_err_arr.append(df[df['eps'] == eps_err[0]]['origin_err'].mean())


# functions
def plot_baseline_err(algo_arr, results, eps_err):
    baseline_err_arr = []
    for algo in algo_arr:
        try:
            df = results[algo]
            baseline_err_arr.append(df[df['eps'] == eps_err[0]]['origin_err'].mean())
        except:
            baseline_err_arr.append(np.nan)

    print('Mean = {}'.format(np.array(baseline_err_arr).mean()))
    # print('Median = {}'.format(np.array(baseline_err_arr).median()))
    plt.bar(algo_arr, baseline_err_arr, color=colours)
    plt.grid(True)
    plt.title('Baseline error')

    return pd.DataFrame({'algo': algo_arr, 'b_err': baseline_err_arr})


def plot_metric(results, col_str, nc_func_arr, algo_arr, eps_err, colours, ls_arr, lw_arr):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    index_dic = {
        0: [0, 0],
        2: [0, 1],
        4: [1, 0],
        6: [1, 1],
    }

    min_val = np.inf
    max_val = 0
    for algo in algo_arr:
        try:
            new_min = np.min(np.min(results[algo][['eps', 'marg{}'.format(col_str),
                                                   'inv{}'.format(col_str),
                                                   'inv_m{}'.format(col_str)]].groupby('eps').mean()))
            new_max = np.max(np.max(results[algo][['eps', 'marg{}'.format(col_str),
                                                   'inv{}'.format(col_str),
                                                   'inv_m{}'.format(col_str)]].groupby('eps').mean()))
            if new_min < min_val:
                min_val = new_min
            if new_max > max_val:
                max_val = new_max
        except:
            pass

    for k in [0, 2, 4, 6]:
        idx0, idx1 = index_dic[k]
        for i in range(k, k + 2):
            algo = algo_arr[i]
            try:
                metric_df = results[algo][['eps', 'marg{}'.format(col_str),
                                           'inv{}'.format(col_str),
                                           'inv_m{}'.format(col_str)]].groupby('eps').mean()

                for nc_idx in range(0, len(nc_func_arr)):
                    # print(nc_idx)
                    if nc_idx == 2:
                        label_str = algo
                    else:
                        label_str = '_nolegend_'
                    ax[idx0, idx1].plot(eps_err, metric_df[nc_func_arr[nc_idx] + col_str],
                                        c=colours[i], ls=ls_arr[nc_idx], lw=lw_arr[nc_idx],
                                        label=label_str
                                        )
            except:
                pass
            pass
        ax[idx0, idx1].set_xticks(eps_err)
        ax[idx0, idx1].set_ylim((min_val * 0.9, max_val * 1.1))
        ax[idx0, idx1].grid(True)
        ax[idx0, idx1].legend()

        pass
    plt.suptitle(col_str[1:])
    pass


def get_stat_significance(eps_err, inv_prob_df, margin_df, inv_prob_margin_df, metric, p_val=0.05):
    # print(metric)

    larger_arr = []
    smaller_arr = []

    order_list = [
        ('inv_prob', 'inv_prob_margin'),
        ('inv_prob', 'margin'),
        ('inv_prob_margin', 'margin')
    ]

    eps_list = []
    key_arr = []

    for eps in eps_err:
        inv_prob_res = inv_prob_df.loc[inv_prob_df['eps'] == eps][metric]
        margin_res = margin_df.loc[inv_prob_df['eps'] == eps][metric]
        inv_prob_margin_res = inv_prob_margin_df.loc[inv_prob_df['eps'] == eps][metric]
        all_res = {'inv_prob': inv_prob_res,
                   'inv_prob_margin': inv_prob_margin_res,
                   'margin': margin_res,
                   }
        for order in order_list:
            key1 = order[0]
            key2 = order[1]
            eps_list.append(eps)
            key_arr.append('{}-{}'.format(key1, key2))
            res = ttest_ind(all_res[key1], all_res[key2], alternative='larger', usevar='unequal')
            larger_arr.append(res[1])
            res = ttest_ind(all_res[key1], all_res[key2], alternative='smaller', usevar='unequal')
            smaller_arr.append(res[1])
            pass
        pass

    larger_res_df = pd.DataFrame({})
    larger_res_df['eps'] = eps_list
    larger_res_df['key'] = key_arr
    larger_res_df['larger'] = (np.array(larger_arr) < p_val)

    smaller_res_df = pd.DataFrame({})
    smaller_res_df['eps'] = eps_list
    smaller_res_df['key'] = key_arr
    smaller_res_df['smaller'] = (np.array(smaller_arr) < p_val)

    res_dic = {}
    for eps in eps_err:
        ip_col = [np.nan for i in range(0, 3)]
        ip_m_col = [np.nan for i in range(0, 3)]
        m_col = [np.nan for i in range(0, 3)]

        key = 'inv_prob-inv_prob_margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[1] = 1
            ip_m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[1] = -1
            ip_m_col[0] = 1

        key = 'inv_prob-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[2] = 1
            m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[2] = -1
            m_col[0] = 1

        key = 'inv_prob_margin-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_m_col[2] = 1
            m_col[1] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_m_col[2] = -1
            m_col[1] = 1

        # if eps == 0.2:
        #    print(ip_col)
        #    print(ip_m_col),
        #    print(m_col)

        ttest_df = pd.DataFrame()
        ttest_df['IP'] = ip_col
        ttest_df['IP_M'] = ip_m_col
        ttest_df['M'] = m_col
        ttest_df['n-func'] = ['IP', 'IP_M', 'M']
        ttest_df.set_index('n-func', inplace=True)
        if metric == 'avgC':
            # print('avgC')
            res_dic[str(eps)] = ttest_df * -1
        else:
            res_dic[str(eps)] = ttest_df
        pass
    return res_dic


def get_diff_mean(eps_err, inv_prob_df, margin_df, inv_prob_margin_df, metric):
    # print(metric)

    larger_arr = []
    smaller_arr = []

    order_list = [
        ('inv_prob', 'inv_prob_margin'),
        ('inv_prob', 'margin'),
        ('inv_prob_margin', 'margin')
    ]

    eps_list = []
    key_arr = []

    for eps in eps_err:
        inv_prob_res = inv_prob_df.loc[inv_prob_df['eps'] == eps][metric]
        margin_res = margin_df.loc[inv_prob_df['eps'] == eps][metric]
        inv_prob_margin_res = inv_prob_margin_df.loc[inv_prob_df['eps'] == eps][metric]
        all_res = {'inv_prob': inv_prob_res,
                   'inv_prob_margin': inv_prob_margin_res,
                   'margin': margin_res,
                   }
        for order in order_list:
            key1 = order[0]
            key2 = order[1]
            eps_list.append(eps)
            key_arr.append('{}-{}'.format(key1, key2))
            res = all_res[key1].mean() > all_res[key2].mean()
            larger_arr.append(res)
            res = all_res[key1].mean() < all_res[key2].mean()
            smaller_arr.append(res)
            pass
        pass

    larger_res_df = pd.DataFrame({})
    larger_res_df['eps'] = eps_list
    larger_res_df['key'] = key_arr
    larger_res_df['larger'] = np.array(larger_arr)

    smaller_res_df = pd.DataFrame({})
    smaller_res_df['eps'] = eps_list
    smaller_res_df['key'] = key_arr
    smaller_res_df['smaller'] = np.array(smaller_arr)

    res_dic = {}
    for eps in eps_err:
        ip_col = [np.nan for i in range(0, 3)]
        ip_m_col = [np.nan for i in range(0, 3)]
        m_col = [np.nan for i in range(0, 3)]

        key = 'inv_prob-inv_prob_margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[1] = 1
            ip_m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[1] = -1
            ip_m_col[0] = 1

        key = 'inv_prob-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[2] = 1
            m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[2] = -1
            m_col[0] = 1

        key = 'inv_prob_margin-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_m_col[2] = 1
            m_col[1] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_m_col[2] = -1
            m_col[1] = 1

        # if eps == 0.2:
        #    print(ip_col)
        #    print(ip_m_col),
        #    print(m_col)

        ttest_df = pd.DataFrame()
        ttest_df['IP'] = ip_col
        ttest_df['IP_M'] = ip_m_col
        ttest_df['M'] = m_col
        ttest_df['n-func'] = ['IP', 'IP_M', 'M']
        ttest_df.set_index('n-func', inplace=True)
        if metric == 'avgC':
            print('avgC')
            res_dic[str(eps)] = ttest_df * -1
        else:
            res_dic[str(eps)] = ttest_df
        pass
    return res_dic


def get_diff_mean_threshold(eps_err, inv_prob_df, margin_df, inv_prob_margin_df, metric, n_classes, threshold=0.025):
    # print(metric)

    larger_arr = []
    smaller_arr = []

    order_list = [
        ('inv_prob', 'inv_prob_margin'),
        ('inv_prob', 'margin'),
        ('inv_prob_margin', 'margin')
    ]

    eps_list = []
    key_arr = []

    for eps in eps_err:
        inv_prob_res = inv_prob_df.loc[inv_prob_df['eps'] == eps][metric].copy()
        margin_res = margin_df.loc[inv_prob_df['eps'] == eps][metric].copy()
        inv_prob_margin_res = inv_prob_margin_df.loc[inv_prob_df['eps'] == eps][metric].copy()
        if metric == 'avgC':
            inv_prob_res = inv_prob_res / n_classes
            margin_res = margin_res / n_classes
            inv_prob_margin_res = inv_prob_margin_res / n_classes
        all_res = {'inv_prob': inv_prob_res,
                   'inv_prob_margin': inv_prob_margin_res,
                   'margin': margin_res,
                   }
        for order in order_list:
            key1 = order[0]
            key2 = order[1]
            eps_list.append(eps)
            key_arr.append('{}-{}'.format(key1, key2))
            res = all_res[key1].mean() > (all_res[key2].mean() + threshold)
            larger_arr.append(res)
            res = all_res[key1].mean() < (all_res[key2].mean() - threshold)
            smaller_arr.append(res)
            pass
        pass

    larger_res_df = pd.DataFrame({})
    larger_res_df['eps'] = eps_list
    larger_res_df['key'] = key_arr
    larger_res_df['larger'] = np.array(larger_arr)

    smaller_res_df = pd.DataFrame({})
    smaller_res_df['eps'] = eps_list
    smaller_res_df['key'] = key_arr
    smaller_res_df['smaller'] = np.array(smaller_arr)

    res_dic = {}
    for eps in eps_err:
        ip_col = [np.nan for i in range(0, 3)]
        ip_m_col = [np.nan for i in range(0, 3)]
        m_col = [np.nan for i in range(0, 3)]

        key = 'inv_prob-inv_prob_margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[1] = 1
            ip_m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[1] = -1
            ip_m_col[0] = 1

        key = 'inv_prob-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_col[2] = 1
            m_col[0] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_col[2] = -1
            m_col[0] = 1

        key = 'inv_prob_margin-margin'
        val = smaller_res_df[(smaller_res_df['eps'] == eps) & (smaller_res_df['key'] == key)]['smaller'].values[0]
        if val:
            ip_m_col[2] = 1
            m_col[1] = -1
        val = larger_res_df[(larger_res_df['eps'] == eps) & (larger_res_df['key'] == key)]['larger'].values[0]
        if val:
            ip_m_col[2] = -1
            m_col[1] = 1

        # if eps == 0.2:
        #    print(ip_col)
        #    print(ip_m_col),
        #    print(m_col)

        ttest_df = pd.DataFrame()
        ttest_df['IP'] = ip_col
        ttest_df['IP_M'] = ip_m_col
        ttest_df['M'] = m_col
        ttest_df['n-func'] = ['IP', 'IP_M', 'M']
        ttest_df.set_index('n-func', inplace=True)
        if metric == 'avgC':
            print('avgC')
            res_dic[str(eps)] = ttest_df * -1
        else:
            res_dic[str(eps)] = ttest_df
        pass
    return res_dic


def plot_stats(res, algo, col, message='Statistical difference for {}, {}. 1 - better, -1 - worse'):
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_figwidth(15)
    fig.set_figheight(7)
    fig.suptitle(message.format(col, algo))

    eps = 0.01
    sns.heatmap(res[str(eps)], annot=True, ax=axes[0, 0])
    axes[0, 0].set_title('$\epsilon={}$'.format(eps))

    eps = 0.03
    sns.heatmap(res[str(eps)], annot=True, ax=axes[0, 1])
    axes[0, 1].set_title('$\epsilon={}$'.format(eps))

    eps = 0.05
    sns.heatmap(res[str(eps)], annot=True, ax=axes[0, 2])
    axes[0, 2].set_title('$\epsilon={}$'.format(eps))

    eps = 0.1
    sns.heatmap(res[str(eps)], annot=True, ax=axes[1, 0])
    axes[1, 0].set_title('$\epsilon={}$'.format(eps))

    eps = 0.15
    sns.heatmap(res[str(eps)], annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('$\epsilon={}$'.format(eps))

    eps = 0.2
    sns.heatmap(res[str(eps)], annot=True, ax=axes[1, 2])
    axes[1, 2].set_title('$\epsilon={}$'.format(eps))
    pass


def stats_test_for_algo_2(algo, results, col, p_val=0.05):
    return


def stats_test_for_algo(algo, results, col, p_val=0.05):
    margin_df = results[algo][['eps', 'marg_oneC', 'marg_eff_oneC', 'marg_avgC']]
    margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_df = results[algo][['eps', 'inv_oneC', 'inv_eff_oneC', 'inv_avgC']]
    inv_prob_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_margin_df = results[algo][['eps', 'inv_m_oneC', 'inv_m_eff_oneC', 'inv_m_avgC']]
    inv_prob_margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    res = get_stat_significance(eps_err, inv_prob_df, margin_df, inv_prob_margin_df,
                                col, p_val=p_val)
    plot_stats(res, algo, col)
    return res


def mean_test_for_algo_2(algo, results, col):
    return


def mean_test_for_algo(algo, results, col):
    margin_df = results[algo][['eps', 'marg_oneC', 'marg_eff_oneC', 'marg_avgC']]
    margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_df = results[algo][['eps', 'inv_oneC', 'inv_eff_oneC', 'inv_avgC']]
    inv_prob_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_margin_df = results[algo][['eps', 'inv_m_oneC', 'inv_m_eff_oneC', 'inv_m_avgC']]
    inv_prob_margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    res = get_diff_mean(eps_err, inv_prob_df, margin_df, inv_prob_margin_df, col)
    plot_stats(res, algo, col, message='Difference in mean for {}, {}. 1 - better, -1 - worse')
    return res


def mean_threshold_test_for_algo(algo, results, col, n_classes, threshold):
    margin_df = results[algo][['eps', 'marg_oneC', 'marg_eff_oneC', 'marg_avgC']]
    margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_df = results[algo][['eps', 'inv_oneC', 'inv_eff_oneC', 'inv_avgC']]
    inv_prob_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    inv_prob_margin_df = results[algo][['eps', 'inv_m_oneC', 'inv_m_eff_oneC', 'inv_m_avgC']]
    inv_prob_margin_df.columns = ['eps', 'oneC', 'eff_oneC', 'avgC']

    res = get_diff_mean_threshold(eps_err, inv_prob_df, margin_df, inv_prob_margin_df, col, n_classes, threshold)
    plot_stats(res, algo, col, message='Difference with threshold in mean for {}, {}. 1 - better, -1 - worse')
    return res


# constants
path = 'analysis-results/'
data_order = ['balance',
              'cars',
              'ecoli',
              'glass',
              'iris',
              'user',
              'wave',
              'wine',
              'wine_Red',
              'wine_White',
              'yeast',
              'gen_nor_0.2',
              'gen_nor_0.4',
              'gen_nor_0.6',
              'gen_nor_0.8',
              'gen_nor_1',
              ]

algo_arr = [
    'SVM',
    'DT',
    'KNN',
    'Ada',
    'GNB',
    'MPR',
    'RF',
    'QDA',
]
colours = ['magenta', 'orange', 'brown', 'green', 'tomato', 'black', 'blue', 'grey']
eps_err = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
nc_func_arr = [
    'marg',
    'inv',
    'inv_m']
ls_arr = [
    '--',
    '-.',
    '-',
]
lw_arr = [
    3,
    3,
    1,
]
p_val = 0.05


def print_tab_2(mean_res, stat_res, metric='oneC'):
    return


def print_tab(mean_res, stat_res, metric='oneC'):
    for line in ['IP', 'IP_M', 'M']:
        str_tmp = ''
        if line == 'IP':
            if metric == 'oneC':
                str_tmp += '\multirow{3}{*}{\\rotatebox[origin=c]{90}{$oneC$}}&ip           & '
            else:
                str_tmp += '\multirow{3}{*}{\\rotatebox[origin=c]{90}{$avgC$}}&ip           & '
        elif line == 'IP_M':
            str_tmp += '&ip\_m        & '
        else:
            str_tmp += '&m            & '

        for eps in eps_err:
            mean_arr = mean_res[str(eps)].T[line].values
            stat_arr = stat_res[str(eps)].T[line].values

            if eps == 0.03:
                continue
            for i in range(0, len(mean_arr)):
                if mean_arr[i] > 0:
                    str_tmp += '$+$'
                elif mean_arr[i] < 0:
                    str_tmp += '$-$'
                else:
                    str_tmp += '   '
                if ~np.isnan(stat_arr[i]):
                    str_tmp += '*       & '
                else:
                    str_tmp += '        & '
            pass
        str_tmp = str_tmp[:-9]
        str_tmp += '        \\\\'
        print(str_tmp)
    pass


def print_tab_threshold(mean_res, mean_res_threshold, stat_res, metric='oneC'):
    res_str = ''
    for line in ['IP', 'IP_M', 'M']:
        str_tmp = ''
        if line == 'IP':
            if metric == 'oneC':
                str_tmp += '\multirow{3}{*}{\\rotatebox[origin=c]{90}{$oneC$}}&ip           & '
            else:
                str_tmp += '\multirow{3}{*}{\\rotatebox[origin=c]{90}{$avgC$}}&ip           & '
        elif line == 'IP_M':
            str_tmp += '&ip\_m        & '
        else:
            str_tmp += '&m            & '

        for eps in eps_err:
            mean_arr = mean_res[str(eps)].T[line].values
            mean_threshold_arr = mean_res_threshold[str(eps)].T[line].values
            stat_arr = stat_res[str(eps)].T[line].values

            if eps == 0.03:
                continue
            for i in range(0, len(mean_arr)):
                if (mean_threshold_arr[i] > 0) or (mean_arr[i] > 0 and ~np.isnan(stat_arr[i])):
                    str_tmp += '$+$'
                elif (mean_threshold_arr[i] < 0) or (mean_arr[i] < 0 and ~np.isnan(stat_arr[i])):
                    str_tmp += '$-$'
                else:
                    str_tmp += '   '
                if ~np.isnan(stat_arr[i]):
                    str_tmp += '*       & '
                else:
                    str_tmp += '        & '
            pass
        str_tmp = str_tmp[:-9]
        str_tmp += '        \\\\'
        res_str += str_tmp + '\n'
    return res_str


data_order_table = {
    'balance': (1, 'balance'),
    'cars': (2, 'cars'),
    'glass': (3, 'glass'),
    'iris': (4, 'iris'),
    'user': (5, 'user'),
    'wave': (6, 'wave'),
    'wine_Red': (7, 'wineR'),
    'wine_White': (8, 'wineW'),
    'yeast': (9, 'yeast'),
    'gen_0.5': (10, 'gen_0.5'),
    'gen_0.75': (11, 'gen_0.75'),
    'gen_1': (12, 'gen_1'),
    'gen_2': (13, 'gen_2'),
    'gen_5': (14, 'gen_5'),
    'gen_nor_0.2': (15, 'gen_nor_0.2'),
    'gen_nor_0.4': (16, 'gen_nor_0.4'),
    'gen_nor_0.6': (17, 'gen_nor_0.6'),
    'gen_nor_0.8': (18, 'gen_nor_0.8'),
    'gen_nor_1': (19, 'gen_nor_1'),
    'heat': (20, 'heat'),
    'cool': (21, 'cool'),
}


def prep_table(stat_res_dic, dataset_name, algo_arr, ):
    # preambula
    res_str = ""
    with open('for-paper/table_preambula') as f:
        for line in f:
            res_str += line
    res_str = res_str.replace('XXX', data_order_table[dataset_name][1])

    algo_str = ""
    with open('for-paper/table_algo') as f:
        for line in f:
            algo_str += line

    mean_res_val = 0
    mean_res_th_val = 1
    stat_res_val = 2

    for algo in algo_arr:
        # add 3 times /hline
        res_str += '\n\hline\n\hline\n\hline'
        # get oneC string
        metric = 'oneC'
        mean_res = stat_res_dic[algo][metric][mean_res_val]
        mean_res_threshold = stat_res_dic[algo][metric][mean_res_th_val]
        stat_res = stat_res_dic[algo][metric][stat_res_val]

        res_str += '\n' + print_tab_threshold(mean_res, mean_res_threshold, stat_res, metric=metric)
        res_str += '\n\hline'

        # algo description
        res_str += '\n' + algo_str.replace('XXX', algo)
        res_str += '\n\hline'

        # avgC
        metric = 'avgC'
        mean_res = stat_res_dic[algo][metric][mean_res_val]
        mean_res_threshold = stat_res_dic[algo][metric][mean_res_th_val]
        stat_res = stat_res_dic[algo][metric][stat_res_val]

        res_str += '\n' + print_tab_threshold(mean_res, mean_res_threshold, stat_res, metric=metric)
        # print(algo)
        pass
    # add ending
    res_str += '\n'
    with open('for-paper/table_ending') as f:
        for line in f:
            res_str += line

    return res_str
