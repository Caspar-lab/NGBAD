import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import io
import time
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.utils import check_consistent_length, column_or_1d
import GB
import nof

warnings.filterwarnings('ignore')


# load the data
dir_path = os.path.split(os.path.realpath(__file__))[0]
dataname_path = os.path.join(dir_path, 'datasets\\all_datalists_outlier.mat')  # Categorical Mixed Numerical
datalists = io.loadmat(dataname_path)['datalists']

no_data_ID = [8, 9, 26, 32, 34, 38, 39, 47, 48, 51] + list(range(62, 68))

algorithm_name = 'NGBAD'
for data_i in range(0, len(datalists)):
    # for data_i in range(33, 34):
    print('data_i=', data_i)
    data_name = datalists[data_i][0][0]
    print('dataset:', data_name)
    if data_i in no_data_ID:
        print('Dataset:' + data_name + ' 运行不出来！！！')
        continue
    dataset_path = r"D:\Microsoft\documents\博士课题\异常检测\实验\datasets"
    data_path = os.path.join(dataset_path, data_name + '.mat')
    trandata = io.loadmat(data_path)['trandata']

    add_folder = os.path.join(
        os.path.join(dir_path, 'Experimental_results\\' + algorithm_name + '_results_32\\' + data_name))
    # if os.path.exists(add_folder):
    #     print(data_name + " 已经有实验结果！！！")
    #     continue
    # os.mkdir(add_folder)

    oridata = trandata.copy()
    trandata = trandata.astype(float)
    # 标准化原始数据
    ID = (trandata >= 1).all(axis=0) & (trandata.max(axis=0) != trandata.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        trandata[:, ID] = scaler.fit_transform(trandata[:, ID])

    X = trandata[:, 0:-1]  # X是去除标签之后的数据
    labels = trandata[:, -1]

    opt_k = 1
    opt_AUC = 0
    opt_T = 0

    centers, gb_list, gb_weight = GB.getGranularBall(X)  # 得到粒球
    index = []
    for gb in gb_list:
        index.append(gb[:, -1])  # 获取在原始数据中的index

    start = time.time()

    out_scores = nof.NOF(centers, X, index)
    # out_scores = nof.NOF(X)

    end = time.time()
    T = end - start
    labels = column_or_1d(labels)  # 将一个列向量或一维numpy数组转换成一维数组
    out_scores = column_or_1d(out_scores)
    check_consistent_length(labels, out_scores)
    results_name1 = data_name + '_' + algorithm_name + '.mat'
    save_path = os.path.join(add_folder, results_name1)
    # io.savemat(save_path, {'out_scores': out_scores.reshape(-1, 1)})
    AUC = roc_auc_score(labels, out_scores)
    # print('ROC=',AUC)
    if AUC > opt_AUC:
        opt_AUC = AUC
        opt_out_scores = out_scores
        opt_T = T
    print('opt_AUC=', opt_AUC)
    T_temp = np.zeros((len(opt_out_scores), 1))
    T_temp[0] = opt_k
    T_temp[1] = opt_AUC
    T_temp[2] = opt_T
    opt_out_scores = opt_out_scores.reshape(-1, 1)
    opt_out_scores = np.column_stack((opt_out_scores, T_temp))
    results_name2 = data_name + '_' + algorithm_name + '.mat'
    save_path = os.path.join(add_folder, results_name2)
    # io.savemat(save_path, {'opt_out_scores': opt_out_scores})