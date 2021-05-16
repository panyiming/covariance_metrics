import json
import mgarch
import os
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt import solvers
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import OLSInfluence


solvers.options['show_progress'] = False


def load_json(path):
    with open(path, 'rb') as f:
        json_dict = json.load(f)
    return json_dict


def find_first_800(sp_data_matrix, num):
    posts = []
    trading_days = sp_data_matrix.shape[0]
    for i in range(trading_days):
        num_eff = np.sum(sp_data_matrix[i, 2:] > -5)
        if num_eff == num:
            posts.append(i)
    return posts[0]


def get_sub_mat(df_return, st_idx, code_shares, sp_data, sam_size):
    indx_length = sam_size
    num_lim = indx_length + 21
    ed_idx = st_idx + num_lim
    code_ls = list(df_return.columns)[1:]
    code_used = []
    res_mat = []
    res_dict = {}
    res_pre_dict = {}
    market_cap_dict = {}
    for code_i in code_ls:
        res = df_return[code_i][st_idx:ed_idx]
        eff_num = np.sum(res > -5)
        if eff_num == num_lim:
            cap_i = code_shares[code_i] * sp_data[code_i][st_idx + indx_length]
            res_ls = list(res)
            res_mat.append(res_ls[0:indx_length])
            code_used.append(code_i)
            res_dict[code_i] = res_ls[0:indx_length]
            res_pre_dict[code_i] = res_ls[indx_length:indx_length + 21]
            market_cap_dict[code_i] = cap_i
    return res_mat, code_used, res_dict, res_pre_dict, market_cap_dict


def get_N(res_mat, code_used, res_dict, res_pre_dict, market_cap_dict, N):
    pos = np.where(np.corrcoef(np.array(res_mat)) > 0.95)
    if len(pos) > 1:
        line_num, row_num = pos
        line_num = line_num.tolist()
        row_num = row_num.tolist()
        code_used_ori = copy.deepcopy(code_used)
        code_used = set(code_used)
        for i in range(len(line_num)):
            for j in range(len(row_num)):
                if i > j:
                    stock_i = code_used_ori[line_num[i]]
                    stock_j = code_used_ori[row_num[j]]
                    cap_i = market_cap_dict[stock_i]
                    cap_j = market_cap_dict[stock_j]

                    if cap_i >= cap_j and cap_i in code_used:
                        code_used.remove(stock_j)
                    elif cap_i < cap_j and cap_i in code_used:
                        code_used.remove(stock_i)
    code_cap = {}
    for code_i in code_used:
        code_cap[code_i] = market_cap_dict[code_i]
    code_cap_sorted = sorted(code_cap.items(), key=lambda x: x[1], reverse=True)
    code_N = [i[0] for i in code_cap_sorted[0:N]]

    res_mat_new = []
    res_pre_new = []
    for code_i in code_N:
        res_mat_new.append(res_dict[code_i])
        res_pre_new.append(res_pre_dict[code_i])
    return res_mat_new, res_pre_new


def get_rolls(code_shares, sp_data, df_return, last_num, N, num, sam_size):
    st_id = find_first_800(df_return.to_numpy(), num)
    roll_len = sam_size + 21
    j = 0
    all_rolls_res = 1
    while st_id + sam_size + 21 <= last_num:
        res_mat, code_used, res_dict, res_pre_dict, market_cap_dict = get_sub_mat(df_return,
                                                      st_id, code_shares, sp_data, sam_size)

        res_mat_new, res_pre_new = get_N(res_mat, code_used,
                                         res_dict, res_pre_dict, market_cap_dict, N)
        res_mat_new = np.array(res_mat_new).transpose()
        res_pre_new = np.array(res_pre_new).transpose()
        res_all = np.vstack((res_mat_new, res_pre_new))
        st_id += 21
        if all_rolls_res is 1:
            all_rolls_res = res_all
        else:
            all_rolls_res = np.vstack((all_rolls_res, res_all))

    return all_rolls_res


def get_return(weights, res_pre):
    weights = np.matrix(weights)
    res_pre = np.matrix(res_pre)
    return (weights * res_pre).tolist()[0]


def average_weights(n):
    return np.matrix([1 / n for i in range(n)])


def get_ws(cov, no_short=True):
    if no_short:
        N = cov.shape[0]
        P = matrix(cov)
        q = matrix(np.zeros((N, 1)))
        G = matrix(-np.identity(N))
        h = matrix(np.zeros((N, 1)))
        A = matrix(1.0, (1, N))
        b = matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)
        ws = np.matrix(sol['x']).T
    else:
        cov_mat = np.matrix(cov)
        N = cov_mat.shape[0]
        I = np.matrix(np.ones(N)).T
        ws = (cov_mat.I * I) / (I.T * cov_mat.I * I)
        ws = ws.T
    return ws


def get_sam_res(code_shares, sp_data, df_return, last_num, N, num, sam_size):
    st_id = find_first_800(df_return.to_numpy(), num)
    res_ew = []
    res_sample_cov = []
    while st_id + sam_size + 21 <= last_num:
        res_mat, code_used, res_dict, res_pre_dict, market_cap_dict = get_sub_mat(df_return,
                                                     st_id, code_shares, sp_data, sam_size)
        res_mat_new, res_pre_new = get_N(res_mat, code_used,
                                         res_dict, res_pre_dict, market_cap_dict, N)
        st_id += 21
        avg_ws = average_weights(N)
        avg_res_i = get_return(avg_ws, res_pre_new)
        res_ew += avg_res_i
        sam_cov = np.cov(np.array(res_mat_new))
        samp_ws = get_ws_cov(sam_cov, no_short=False)
        sam_res_i = get_return(samp_ws, res_pre_new)
        res_sample_cov += sam_res_i
    return res_ew, res_sample_cov


def indicators(res):
    AV = round(np.mean(res) * 252 * 100, 5)
    SD = round(np.cov(res) ** 0.5 * 252 ** 0.5 * 100, 5)
    IR = round(AV / SD, 5)
    return AV, SD, IR


def read_dcc_res(path):
    res_dccnl = pd.read_csv(path)
    res_dccnl = res_dccnl['x']
    res_dccnl = list(res_dccnl)
    return res_dccnl


def res_line(res):
    portfolio_res = [1]
    a = 1
    for i in res:
        a = a * (1 + i)
        portfolio_res.append(a)
    return portfolio_res


def get_dcc_res(root_dir, rolls, N, no_short):
    file_ls = os.listdir(root_dir)
    length = len(file_ls)
    res_dccnl = []
    for i in range(length):
        file_path = os.path.join(root_dir, '{}.csv'.format(i + 1))
        cov_mat = pd.read_csv(file_path)
        cov_mat = np.matrix(cov_mat.to_numpy())
        # st_idx = i*1281
        # ed_idx = i*1281+1260
        # res_mat = rolls[st_idx:ed_idx, 1:N+1]
        # cov_mat =  np.cov(np.array(res_mat).transpose())
        ws_i = get_ws_cov(cov_mat, no_short=no_short)
        st_idx = (i + 1) * 1281 - 21
        ed_idx = (i + 1) * 1281
        res_pre_i = rolls[st_idx:ed_idx, 1:N + 1]
        res_pre_i = np.matrix(res_pre_i).T
        res_i = get_return(ws_i, res_pre_i)
        res_dccnl += res_i
    return res_dccnl


def get_code(res_mat, code_used,  market_cap_dict, N):
    pos = np.where(np.corrcoef(np.array(res_mat)) > 0.95)
    if len(pos) > 1:
        line_num, row_num = pos
        line_num = line_num.tolist()
        row_num = row_num.tolist()
        code_used_ori = copy.deepcopy(code_used)
        code_used = set(code_used)
        for i in range(len(line_num)):
            for j in range(len(row_num)):
                if i > j:
                    stock_i = code_used_ori[line_num[i]]
                    stock_j = code_used_ori[row_num[j]]
                    cap_i = market_cap_dict[stock_i]
                    cap_j = market_cap_dict[stock_j]

                    if cap_i >= cap_j and cap_i in code_used:
                        code_used.remove(stock_j)
                    elif cap_i < cap_j and cap_i in code_used:
                        code_used.remove(stock_i)
    code_cap = {}
    for code_i in code_used:
        code_cap[code_i] = market_cap_dict[code_i]
    code_cap_sorted = sorted(code_cap.items(), key=lambda x: x[1], reverse=True)
    code_N = [i[0] for i in code_cap_sorted[0:N]]
    return code_N


def dcc_garch_pred(ffdata):
    vol = mgarch.mgarch()
    vol.fit(ffdata)
    cov_sum = np.zeros((5, 5))
    for i in range(21):
        cov_sum += vol.predict(i)['cov']
    cov = cov_sum / 21
    return cov


def regression(code_shares, sp_data, df_return,
               last_num, N, num, sam_size, ffdata, dir):
    """
    :param code_shares: the dict record the shares number
    :param sp_data: trading data from 19750101
    :param df_return: return rates from 19750102
    :param last_num: 11602
    :param N: number of assets in the portfolio
    :param num: the least number of stocks have data
    :param sam_size: trading dates for the covariance estimation
    :param ffdata: dataframe of five factors model
    :param dir: save parameters in the file dir
    :return: write the factor loading, residuals, factor covariance matrix
    """
    st_id = find_first_800(df_return.to_numpy(), num)
    residual_mat1 = np.zeros((500*190, 1260))
    residual_mat5 = np.zeros((500*190, 1260))
    params_mat1 = np.zeros((500*190, 1))
    params_mat5 = np.zeros((500*190, 5))
    ff_vol1 = np.zeros((190, 1))
    ff_vol5 = np.zeros((190*5, 5))
    #ff_vol_dcc1 = np.zeros((190, 1))
    #ff_vol_dcc5 = np.zeros((190*5, 5))
    num_i = 0
    while st_id + sam_size + 21 <= last_num:
        time_st = time.time()
        # get trading date index
        # get the names of the stock used at every trading date formed by list.
        # for every asset, run the regression, get the loading factors and the residuals.
        res_mat, code_used, res_dict, res_pre_dict, market_cap_dict = \
                 get_sub_mat(df_return, st_id, code_shares, sp_data, sam_size)
        code_N = get_code(res_mat, code_used, market_cap_dict, N)
        trading_date_i = st_id + sam_size
        ff_ls = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        # sample covariance matrix of factors
        ff_vol1[num_i, :] = np.cov(np.array(ffdata[ff_ls[0]][st_id:trading_date_i]).transpose())
        ff_vol5[num_i*5:num_i*5+5, :] = np.cov(np.array(ffdata[ff_ls][st_id:trading_date_i]).transpose())

        # DCC for estimation of factor covariance matrix:
        #cov_avg = dcc_garch_pred(ffdata[ff_ls][:trading_date_i])
        #ff_vol_dcc1[num_i, :] = cov_avg[1, 1]
        #ff_vol_dcc5[num_i*5:num_i+5, :] = cov_avg

        #num_j = 0
        #for code_i in code_N:
        #    st_date =np.where(df_return[code_i] > -5)[0][0]
        #    # five factors regression
        #    Y5 = list(df_return[code_i][st_id:trading_date_i] - ffdata['RF'][st_id:trading_date_i])
        #    X5 = ffdata[ff_ls][st_id:trading_date_i]
        #    X5 = sm.add_constant(X5)
        #    OLS_model5 = sm.OLS(Y5, X5).fit()
        #    l = len(OLS_model5.resid)
        #    residual_mat5[num_i*500+num_j, :] = list(OLS_model5.resid)[l-1260:l]
        #    params_mat5[num_i*500+num_j, :] = list(OLS_model5.params)[1:]

            # one factor regression
        #   Y1 = list(df_return[code_i][st_id:trading_date_i] - ffdata['RF'][st_id:trading_date_i])
        #    X1 = ffdata['Mkt-RF'][st_id:trading_date_i]
        #    X1 = sm.add_constant(X1)
        #    OLS_model1 = sm.OLS(Y1, X1).fit()
        #    residual_mat1[num_i*500+num_j, :] = list(OLS_model1.resid)[l-1260:l]
        #    params_mat1[num_i*500+num_j, :] = list(OLS_model1.params)[1:]
        #    num_j += 1

        st_id += 21
        num_i += 1
        time_ed = time.time()
        print("the {}th step costs {}".format(num_i, time_ed-time_st))

    #path_reid5 = os.path.join(dir, 'residual_mat5.npy')
    #np.save(path_reid5, residual_mat5)
    #path_reid1 = os.path.join(dir, 'residual_mat1.npy')
    #np.save(path_reid1, residual_mat1)
    #path_pram1 = os.path.join(dir, 'params_mat1.npy')
    #np.save(path_pram1, params_mat1)
    #path_pram5 = os.path.join(dir, 'params_mat5.npy')
    #np.save(path_pram5, params_mat5)
    path_ffvol5 = os.path.join(dir, 'ff5_vol.npy')
    np.save(path_ffvol5, ff_vol5)
    path_ffvol1 = os.path.join(dir, 'ff1_vol.npy')
    np.save(path_ffvol1, ff_vol1)
    # path_ffvol1_dcc = os.path.join(dir, 'ff_vol_dcc1.npy')
    # np.save(path_ffvol1_dcc, ff_vol_dcc1)
    # path_ffvol5_dcc = os.path.join(dir, 'ff_vol_dcc5.npy')
    # np.save(path_ffvol5_dcc, ff_vol_dcc5)


if __name__ == '__main__':
    print('loading data')
    code_shares = load_json('./code_shares.json')
    sp_data = pd.read_csv('./SP1273_19750103_20201231.csv')
    df_return = pd.read_csv('./returns1271.csv')
    ffdata = pd.read_csv('./ffdata_11062.csv')
    print('start regression')
    dir = 'ff_model_data_1260'
    N = 500
    num = 800
    last_num = 11602
    sam_size = 1260
    regression(code_shares, sp_data, df_return, last_num, N, num, sam_size, ffdata, dir)