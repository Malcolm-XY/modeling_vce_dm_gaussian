# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:45:31 2025

@author: 18307
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

legend_name: dict[str, str] = {}
method_order_short: List[str] = []
color_map: dict[str, tuple] = {}

# -----------------------------
def selection_robust_auc(srs, accuracies):
    aucs = []
    n = len(srs) - 1
    for i in range(n):
        auc = (srs[i]-srs[i+1]) * (accuracies[i]+accuracies[i+1])/2
        aucs.append(auc)
        
    auc = np.sum(aucs) * 1/(srs[0]-srs[-1])
    
    return auc

def balanced_performance_efficiency_single_point(sr, accuracy, alpha=1, beta=1):
    bpe = alpha * (1-sr**2) * beta * accuracy
    return bpe

def balanced_performance_efficiency_single_points(srs, accuracies, alpha=1, beta=1):
    bpes = []
    for i, sr in enumerate(srs):
        bpe = alpha * (1-sr**2) * beta * accuracies[i]
        bpes.append(bpe)
        
    return bpes

def balanced_performance_efficiency_multiple_points(srs, accuracies, alpha=1, beta=1):
    bpe_term = []
    normalization_term = []
    n = len(srs) - 1
    for i in range(n):
         bpe_area = (srs[i] - srs[i+1]) * (accuracies[i] * (1-srs[i]**2) + accuracies[i+1] * (1-srs[i+1]**2)) * 1/2 * alpha
         bpe_term.append(bpe_area)
         
         normalization_area = (srs[i] - srs[i+1]) * ((1-srs[i]**2) + (1-srs[i+1]**2)) * 1/2 * beta
         normalization_term.append(normalization_area)
         
    bpe = np.sum(bpe_term)
    bpe_normalized = bpe/np.sum(normalization_term)
    
    return bpe_normalized

def test():
    # 1
    srs = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    accuracies = [93.01581329,91.55604576,89.67219151,84.23354008,75.6307667,58.99205587,47.50004181]
    
    auc = selection_robust_auc(srs, accuracies)
    # print(auc)
    
    bpe = balanced_performance_efficiency_multiple_points(srs, accuracies)
    print(bpe)
    
    # 2
    srs = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    accuracies = [93.14073651,91.29927595,89.60072607,84.14374403,73.70817798,59.63064271,49.01079594]
    
    auc = selection_robust_auc(srs, accuracies)
    # print(auc)
    
    bpe = balanced_performance_efficiency_multiple_points(srs, accuracies)
    print(bpe)

# -----------------------------
def _apply_sr_ticks_and_vlines(ax: plt.Axes, sr_values, vline_kwargs: dict | None = None, tick_labels: List[str] | None = None):
    """
    - 将 x 轴刻度设为给定 sr 集合（去重后按降序）。
    - 在每个 sr 位置画竖直虚线（贯穿当前 y 轴范围）。
    """
    sr_unique = np.array(sorted(np.unique(sr_values), reverse=True), dtype=float)
    # 设刻度
    ax.set_xticks(sr_unique)
    if tick_labels is None:
        ax.set_xticklabels([str(s) for s in sr_unique], fontsize=14)
    else:
        ax.set_xticklabels(tick_labels, fontsize=14)

    # 先拿到绘完图后的 y 轴范围，再画竖线以贯穿全高
    y0, y1 = ax.get_ylim()
    kw = dict(color="gray", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)
    if vline_kwargs:
        kw.update(vline_kwargs)
    for x in sr_unique:
        ax.vlines(x, y0, y1, **kw)
    # 不改变 y 轴范围
    ax.set_ylim(y0, y1)

def build_dataframe(data: dict, identifier: List[str]) -> pd.DataFrame:
    """
    根据给定的数据与方法名列表构建绘图所需的 DataFrame。
    
    参数
    ----
    data: dict
        形如 {"data": [...], "sr": [...], "std": [...]}，三者等长（总行数）。
    identifier: List[str]
        方法名列表（按期望在图例与排序中的顺序给出）。
    
    返回
    ----
    pd.DataFrame:
        包含列 ["Method", "sr", "data", "std"]，按 Method（给定顺序）和 sr（降序）排序。
    """
    global legend_name, method_order_short, color_map

    # 基础长度校验
    acc, sr, sd = data["data"], data["sr"], data["std"]
    if not (len(acc) == len(sr) == len(sd)):
        raise ValueError(f"Length mismatch: data={len(acc)}, sr={len(sr)}, std={len(sd)}")

    methods = [str(m).strip() for m in identifier]
    if any(m == "" for m in methods):
        raise ValueError("Empty method name detected in identifier.")

    n_methods = len(methods)
    total = len(acc)
    if total % n_methods != 0:
        raise ValueError(f"Total rows ({total}) not divisible by number of methods ({n_methods}).")
    per_method = total // n_methods

    # 构造 Method 列（按块拼接：每个方法连续 per_method 行）
    method_col = [m for m in methods for _ in range(per_method)]

    df = pd.DataFrame({
        "Method": method_col,
        "sr": sr,
        "data": acc,
        "std": sd,
    })

    # 一致性校验：每种方法的 SR 序列应一致
    sr_sets = df.groupby("Method")["sr"].apply(lambda s: tuple(s.to_list()))
    if len(set(sr_sets)) != 1:
        raise ValueError(f"Inconsistent SR sequences across methods: {sr_sets.to_dict()}")

    # 固定方法顺序；sr 从大到小
    method_order_short = methods[:]  # 按输入顺序
    df["Method"] = pd.Categorical(df["Method"], categories=method_order_short, ordered=True)
    df = df.sort_values(["Method", "sr"], ascending=[True, False]).reset_index(drop=True)

    # legend_name 直接同名映射；color_map 自动分配
    legend_name = {m: m for m in method_order_short}
    cmap = plt.cm.tab10.colors
    if len(method_order_short) > len(cmap):
        palette = [cmap[i % len(cmap)] for i in range(len(method_order_short))]
    else:
        palette = cmap[:len(method_order_short)]
    color_map = dict(zip(method_order_short, palette))

    # 终检：总行数 = 方法数 × SR 数
    n_sr = df["sr"].nunique()
    if len(df) != n_methods * n_sr:
        raise ValueError(f"Row count {len(df)} != methods({n_methods}) × SR({n_sr}).")

    return df

def plot_data_with_band(
    df: pd.DataFrame,
    data: str = 'data',
    std: str = 'std',
    ylabel: str = 'Accuracy',
    mode: str = "sem",     # "ci" | "sem" | "sd"
    level: float = 0.95,   # 置信水平（用于 mode="ci"）
    n: int | None = None,  # 每个(mean,std)对应的样本量
    fontsize: int = 16,     # 统一字号控制
    cmap = plt.colormaps['viridis'],
) -> None:
    import math

    # 尝试使用 t 分布；失败则退化为正态近似
    z = None
    t_ppf = None
    if mode == "ci":
        try:
            import scipy.stats as st
            t_ppf = lambda dof: st.t.ppf(0.5 + level/2, df=dof)
        except Exception:
            z = 1.96 if abs(level - 0.95) < 1e-6 else None
            if z is None:
                a = 0.147
                p_one_side = 0.5 + level/2
                s = 2*p_one_side - 1
                ln = math.log(1 - s*s)
                z = math.copysign(
                    math.sqrt(math.sqrt((2/(math.pi*a) + ln/2)**2 - ln/a) - (2/(math.pi*a) + ln/2)),
                    s
                )
    
    # plot
    cmap = cmap
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (method, sub) in enumerate(df.groupby("identifier")):
        sub = sub.sort_values("sr", ascending=False)
        x = sub["sr"].to_numpy()
        m = sub[data].to_numpy()
        s = sub[std].to_numpy()
        
        # Error Bar
        if mode == "sd":
            low, high = m - s, m + s
            band_note = "±SD"
        elif mode == "sem":
            if n is None:
                print("[plot_accuracy_with_band] n 未提供，SEM 无法计算，退化为 SD 阴影带（仅作展示）。")
                low, high = m - s, m + s
                band_note = "±SD (fallback)"
            else:
                sem = s / np.sqrt(n)
                low, high = m - sem, m + sem
                band_note = f"±SEM (n={n})"
        elif mode == "ci":
            if n is None:
                print("[plot_accuracy_with_band] n 未提供，CI 无法计算，退化为 SD 阴影带（仅作展示）。")
                low, high = m - s, m + s
                band_note = "±SD (fallback)"
            else:
                sem = s / np.sqrt(n)
                if t_ppf is not None:
                    tval = t_ppf(n - 1)
                    delta = tval * sem
                else:
                    delta = z * sem  # 正态近似
                low, high = m - delta, m + delta
                band_note = f"±{int(level*100)}% CI (n={n})"
        else:
            raise ValueError("mode must be one of {'ci','sem','sd'}")
        # Error Bar End
        
        # Plot Lines + Error Bars
        ax.plot(x, m, marker="o", linewidth=2.0, label=method, zorder=3, color=cmap(i/len(df.groupby("identifier"))))
        ax.fill_between(x, low, high, alpha=0.15, zorder=2, color=cmap(i/len(df.groupby("identifier"))))

    ax.set_xlabel("Selection Rate (for extraction of subnetworks)", fontsize=fontsize)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=fontsize)
    ax.invert_xaxis()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 只按 SR 标定刻度，并在刻度处加竖线
    _apply_sr_ticks_and_vlines(ax, df["sr"])

    ax.tick_params(axis="x", labelsize=fontsize*0.9)
    ax.tick_params(axis="y", labelsize=fontsize*0.9)

    ax.legend(fontsize=fontsize*0.9, title="Bands: " + (band_note if 'band_note' in locals() else ""),
              title_fontsize=fontsize)

    fig.tight_layout()
    plt.show()

def plot_bar_by_method(
    df: pd.DataFrame,
    mode: str = "sem",         # "sem" | "sd"
    n: int | None = None,      # 样本量（用于 sem）
    figsize = (10,10),
    ylabel: str = 'BPE(Balanced Performance Efficiency)',
    xlabel: str = 'FN Recovery Methods',
    fontsize: int = 16,
    bar_width: float = 0.6,    # 柱宽
    annotate: bool = True,
    annotate_fmt: str = "{m:.2f} ± {e:.2f}",
    xtick_rotation: float = 30,   # ✅ X 轴角度控制
    wrap_width: int | None = None # ✅ 自动换行：超过多少字符自动分行
) -> None:
    import textwrap

    # 若 df 含重复 Method，则聚合
    grouped = (
        df.groupby("identifier", sort=False)
        .agg({"data": "mean", "std": "mean"})
        .reset_index()
    )

    # 提取方法与统计值
    methods = grouped["identifier"].astype(str).tolist()

    # ✅ 自动换行
    if wrap_width is not None:
        methods_wrapped = [textwrap.fill(m, wrap_width) for m in methods]
    else:
        methods_wrapped = methods

    means = grouped["data"].to_numpy()
    stds = grouped["std"].to_numpy()

    # 误差计算
    if mode == "sd":
        errs = stds
        err_note = "±SD"
    elif mode == "sem":
        if n is None:
            print("[plot_bar_by_method] n 未提供，SEM 退化为 SD。")
            errs = stds
            err_note = "±SD (fallback)"
        else:
            errs = stds / np.sqrt(n)
            err_note = f"±SEM (n={n})"
    else:
        raise ValueError("mode must be one of {'sd','sem'}")

    x = np.arange(len(methods))

    # 绘制
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        x, means, width=bar_width,
        yerr=errs, capsize=5, zorder=3,
        color='skyblue', edgecolor='black'
    )

    # 注释数值
    if annotate:
        for xx, m, e in zip(x, means, errs):
            ax.text(xx, m + e + 0.3, annotate_fmt.format(m=m, e=e),
                    ha="center", va="bottom", fontsize=fontsize * 0.8)

    # 坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(methods_wrapped, fontsize=fontsize * 0.9,
                       rotation=xtick_rotation, ha="right" if xtick_rotation != 0 else "center")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=fontsize * 0.9)

    # 图例
    ax.legend([bars], [f"Errors: {err_note}"],
              fontsize=fontsize * 0.8, title_fontsize=fontsize)

    fig.tight_layout()
    plt.show()

def sbpe_portion():
    # Original Accuracy
    from line_chart_data import portion_data_pcc
    identifier_po = portion_data_pcc.identifier
    portion_pcc_accuracy_dic = portion_data_pcc.accuracy
    
    df = build_dataframe(portion_pcc_accuracy_dic, identifier_po)
    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))
    plot_data_with_band(df, mode="sem", level=0.95, n=30)
    
    # SBPE
    srs = portion_pcc_accuracy_dic['sr']
    accuracies = portion_pcc_accuracy_dic['data']
    stds = portion_pcc_accuracy_dic['std']
    portion_pcc_bpes = balanced_performance_efficiency_single_points(srs, accuracies)
    portion_pcc_bpes_std = balanced_performance_efficiency_single_points(srs, stds)
    
    portion_pcc_bpes_dic = portion_pcc_accuracy_dic.copy()
    portion_pcc_bpes_dic['data'] = portion_pcc_bpes
    portion_pcc_bpes_dic['std'] = portion_pcc_bpes_std
    
    df = build_dataframe(portion_pcc_bpes_dic, identifier_po)
    df = pd.DataFrame()
    
    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))
    plot_data_with_band(df, mode="sem", level=0.95, n=30, ylabel='BPE(Balanced Performance Efficiency)')

def sbpe_portion():
    from line_chart_data import portion_data_pcc
    portion_pcc_accuracy_dic = portion_data_pcc.accuracy
    
    df = pd.DataFrame(portion_pcc_accuracy_dic)
    
    sbpes, sbpe_stds = [], []
    for method, sub in df.groupby("identifier", sort=False):
        sub = sub.sort_values("sr", ascending=False)
        srs = sub["sr"].to_numpy()
        accuracies = sub["data"].to_numpy()
        stds = sub["std"].to_numpy()
        
        sbpes_ = balanced_performance_efficiency_single_points(srs, accuracies)
        sbpe_stds_ = balanced_performance_efficiency_single_points(srs, stds)
        
        # print(srs)
        # print(accuracies)
        print(f"Methods: {method}", f"SBPEs: {sbpes_}")
        
        sbpes.extend(sbpes_)
        sbpe_stds.extend(sbpe_stds_)
        
    sbpes_dic = {"SBPEs": sbpes, "SBPE_stds": sbpe_stds}
    df = pd.concat([df, pd.DataFrame(sbpes_dic)], axis=1)
    
    plot_data_with_band(df, data='SBPEs', std='SBPE_stds', ylabel='BPE(Balanced Performance Efficiency)', mode="sem", n=30)
    
    return df

def mbpe_portion():
    from line_chart_data import portion_data_pcc
    portion_pcc_accuracy_dic = portion_data_pcc.accuracy
    
    df = pd.DataFrame(portion_pcc_accuracy_dic)
    
    mbpe, mbpe_std = [], []
    for method, sub in df.groupby("identifier", sort=False):
        sub = sub.sort_values("sr", ascending=False)
        srs = sub["sr"].to_numpy()
        accuracies = sub["data"].to_numpy()
        stds = sub["std"].to_numpy()
        
        mbpe_ = balanced_performance_efficiency_multiple_points(srs, accuracies)
        mbpe_std_ = balanced_performance_efficiency_multiple_points(srs, stds)
        
        # print(srs)
        # print(accuracies)
        print(f"Methods: {method}", f"MBPE: {mbpe_}")
        
        mbpe_ = [mbpe_] * len(accuracies)
        mbpe_std_ = [mbpe_std_] * len(accuracies)
        
        mbpe.extend(mbpe_)
        mbpe_std.extend(mbpe_std_)
    
    mbpe_dic = {"MBPE": mbpe, "MBPE_std": mbpe_std}
    df = pd.concat([df, pd.DataFrame(mbpe_dic)], axis=1)
    
    plot_bar_by_method(df, mode="sem", n=30, xtick_rotation=45, wrap_width=30, figsize = (10,15))
    
    return df
    
# %% main
if __name__ == "__main__":
    df = sbpe_portion()
    df = mbpe_portion()
    