import numpy as np
from scipy.spatial.distance import cdist
from NaNsearch import NaNSearch

# 假设已经实现了 NaNSearch，返回：sup_k, Rnb, NN_r, RNN_r
# 例如可以直接引用前面的 NaNSearch 函数
# from nansearch import NaNSearch

def NOF(data, trandata, index):
    data = np.asarray(data)
    n, _ = data.shape

    # 调用 NaNSearch 得到 Rnb，其他返回值未使用
    _, Rnb, _, _ = NaNSearch(data)
    # k为所有样本在 NaNSearch 中出现次数的最大值
    k = int(np.max(Rnb))
    if k > n-1:
        return np.zeros(trandata.shape[0])

    # 计算欧式距离矩阵
    dist = cdist(data, data)  # shape: (n, n)

    # 对每一行排序，distance为排序后的距离，num为对应的索引
    sorted_dists = np.sort(dist, axis=1)
    num = np.argsort(dist, axis=1)
    # 注意：由于每行第一个元素为0（自身距离），MATLAB中取第k+1个，Python中取第k个
    kdistance = sorted_dists[:, k]

    # 构造对称邻域关系矩阵：
    # 对于每个点i，若 dist(i,j) <= kdistance[i]，则该条件为True
    count_temp1 = (dist <= kdistance[:, np.newaxis]).astype(float)
    # 对称化：若任一方向满足条件，则设置为1
    count_temp = (count_temp1 + count_temp1.T) / 2
    count_temp[count_temp == 0.5] = 1

    # 计算每个点的邻域内对象个数（排除自身）
    count = np.sum(count_temp, axis=1) - 1

    # 计算两两之间的 reachable-distance 矩阵
    reachdist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            # reachable distance 定义为两个点间距离与邻居的 k-distance 中的较大者
            reachdist[i, j] = max(dist[i, j], kdistance[j])
            reachdist[j, i] = reachdist[i, j]

    # 计算每个点的局部可达密度 (lrd)
    lrd = np.zeros(n)
    for i in range(n):
        # 取该点的 count(i) 个邻居（按距离升序，跳过第一个自身）
        cnt = int(count[i])
        if cnt == 0:
            lrd[i] = 0
        else:
            # neighbors 为索引数组，注意跳过第一个自身（索引位置0）
            neighbors = num[i, 1:cnt+1]
            sum_reachdist = np.sum(reachdist[i, neighbors])
            # 防止除零
            lrd[i] = count[i] / sum_reachdist if sum_reachdist != 0 else 0

    NOF_score = np.zeros(n)
    # MATLAB 中 num 是二维，但索引时使用单一索引，MATLAB 按列优先展开；在 Python 中模拟这种行为：
    num_flat = num.flatten(order='F')  # Fortran 顺序（列优先）
    for i in range(n):
        sumlrd = 0
        cnt = int(count[i])
        for j in range(cnt):
            # 直接使用全局展平后的 num，索引 j+1，对应 MATLAB 中 lrd(num(j+1))
            sumlrd += lrd[num_flat[j+1]] / lrd[i]
        NOF_score[i] = sumlrd / count[i]

    origin_results = np.zeros(trandata.shape[0])

    for cluster_idx, cluster_indices in enumerate(index):
        for j in cluster_indices:
            origin_results[int(j)] = NOF_score[cluster_idx]

    return origin_results

# 示例调用
if __name__ == '__main__':
    # 构造示例数据：10个二维点，归一化到[0,1]
    data = np.random.rand(10, 2)
    nof_scores = NOF(data)
    print("NOF scores:", nof_scores)
