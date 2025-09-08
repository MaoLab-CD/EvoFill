import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def classical_mds_eigs(D):
    """
    经典 MDS：返回所有正特征值（按降序排列）和对应的累计解释百分比。
    D : 对称的 n×n 距离矩阵
    """
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    # 由于 B 是对称实矩阵，eigh 保证数值稳定
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = eigvals > 1e-12           # 去掉数值噪声
    eigvals = eigvals[idx][::-1]    # 降序
    explained = eigvals / eigvals.sum() * 100
    return eigvals, np.cumsum(explained)


def scree_plot(eigvals, cum_explained, auto_k=True):
    """
    画对数坐标的 scree plot，marker 更小，便于观察拐点。
    eigvals      : 降序排列的正特征值
    cum_explained: 累计解释百分比（与画图无关，仍可顺带显示）
    """
    k_elbow = np.argmax(np.diff(eigvals, 2)) + 2   # 肘部法则
    k_elbow = 5

    fig, ax1 = plt.subplots(figsize=(6, 4))
    kvals = np.arange(1, len(eigvals) + 1)

    # 左侧 y 轴：特征值（对数）
    line1, = ax1.plot(kvals[:100], eigvals[:100], 'o-',  color="tab:red", markersize=3,label="Eigenvalue")
    ax1.set_yscale("log")
    ax1.set_ylabel("Eigenvalue", color="tab:red")
    ax1.tick_params(axis='y', labelcolor="tab:red")
    ax1.set_xlabel("k")
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # # Kaiser 水平线（λ=1）
    # ax1.axhline(1, color="grey", linestyle="--", linewidth=1,
    #             label="Kaiser rule (λ=1)")

    # 肘部竖线
    ax1.axvline(k_elbow, color="tab:green", linestyle="--", linewidth=1,
                label=f"Elbow (k={k_elbow})")

    # 右侧 y 轴：累计解释百分比（线性即可）
    ax2 = ax1.twinx()
    line2, = ax2.plot(kvals[:100], cum_explained[:100], 's-', color="tab:blue", markersize=3, label="Cumulative explained (%)")
    ax2.set_ylabel("Cumulative explained (%)", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    fig.legend([line1, line2],
           [line1.get_label(), line2.get_label()],
           loc='upper center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=2)
    plt.title("Scree Plot")
    plt.tight_layout()
    plt.show()

    if auto_k:
        print(f"Suggested k (elbow rule) : {k_elbow}")
    return k_elbow

def mds_projection(D, k):
    """
    给定距离矩阵 D 和目标维数 k，返回 n×k 的投影坐标矩阵 X
    """
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx_pos = eigvals > 1e-12
    eigvals, eigvecs = eigvals[idx_pos][::-1], eigvecs[:, idx_pos][:, ::-1]

    if k > len(eigvals):
        raise ValueError(f"仅找到 {len(eigvals)} 个正特征值，无法投影到 {k} 维")

    Lambda_k = np.diag(np.sqrt(eigvals[:k]))
    V_k = eigvecs[:, :k]
    X = V_k @ Lambda_k          # n×k
    return X

# ===== 示例用法 =====
if __name__ == "__main__":
    path = '/home/qmtang/mnt_qmtang/EvoFill/data/hg19_chr22/chr22_train.p_dis.mat'
    D = pd.read_csv(path, sep='\t', skiprows=[0], header=None, index_col=0)
    D.index = D.columns = [s.strip() for s in D.index]

    eigvals, cum_explained = classical_mds_eigs(D)
    k_elbow = scree_plot(eigvals, cum_explained, auto_k=True)

    # 选一个 k（这里用肘部法则的结果）
    # k = k_elbow
    k = 4
    X_k = mds_projection(D, k)

    print(f"最终选定的维度 k = {k}")
    print("投影坐标矩阵形状:", X_k.shape)
    print("前 5 行坐标:\n", X_k[:5])