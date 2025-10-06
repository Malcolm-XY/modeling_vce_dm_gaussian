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
    srs = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    accuracies = [93.10136478, 91.32666373, 89.72256104, 83.85720465, 76.24294905, 60.40277742, 47.7409003]
    
    auc = selection_robust_auc(srs, accuracies)
    print(auc)
    
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

def _parse_identifiers(ident_list: List[str]) -> tuple[list[str], list[str]]:
    """
    将 "short, long" 形式的条目解析成短/长标签。
    缺逗号时，短=全文，长=全文；多逗号时，仅按第一个逗号切分。
    """
    shorts, longs = [], []
    for raw in ident_list:
        parts = [p.strip() for p in str(raw).split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s, l = parts[0], parts[1]
        else:
            s = l = str(raw).strip()
        shorts.append(s)
        longs.append(l)
    return shorts, longs

def build_dataframe(data: dict, identifier) -> pd.DataFrame:
    global legend_name, method_order_short, color_map

    # 基础长度校验
    data, sr, sd = data["data"], data["sr"], data["std"]
    if not (len(data) == len(sr) == len(sd)):
        raise ValueError(f"Length mismatch: accuracy={len(data)}, sr={len(sr)}, std={len(sd)}")

    # 解析 Identifier
    methods_short, methods_full = _parse_identifiers(identifier)
    n_methods = len(methods_short)
    total = len(data)
    if total % n_methods != 0:
        raise ValueError(f"Total rows ({total}) not divisible by number of methods ({n_methods}).")
    per_method = total // n_methods

    # 构造 Method / MethodFull 列（按块拼接：每个方法连续 per_method 行）
    method_col = [ms for ms in methods_short for _ in range(per_method)]
    method_full_col = [mf for mf in methods_full for _ in range(per_method)]

    df = pd.DataFrame({
        "Method": method_col,
        "MethodFull": method_full_col,
        "sr": sr,
        "data": data,
        "std": sd,
    })

    # 一致性校验：每种方法的 SR 集应相同
    sr_sets = df.groupby("Method")["sr"].apply(lambda s: tuple(s.to_list()))
    if len(set(sr_sets)) != 1:
        raise ValueError(f"Inconsistent SR sequences across methods: {sr_sets.to_dict()}")

    # 固定短标签顺序（按 Identifier 顺序）；SR 从大到小
    method_order_short = methods_short[:]  # 按输入顺序
    df["Method"] = pd.Categorical(df["Method"], categories=method_order_short, ordered=True)
    df = df.sort_values(["Method", "sr"], ascending=[True, False]).reset_index(drop=True)

    # legend_name & color_map 自动推断
    legend_name = {s: f for s, f in zip(methods_short, methods_full)}
    cmap = plt.cm.tab10.colors
    if len(method_order_short) > len(cmap):
        palette = [cmap[i % len(cmap)] for i in range(len(method_order_short))]
    else:
        palette = cmap[:len(method_order_short)]
    color_map = dict(zip(method_order_short, palette))

    # 终检：总行数 = 方法数 × SR数
    n_sr = df["sr"].nunique()
    if len(df) != n_methods * n_sr:
        raise ValueError(f"Row count {len(df)} != methods({n_methods}) × SR({n_sr}).")

    return df

def plot_data_with_band(
    df: pd.DataFrame,
    mode: str = "sem",     # "ci" | "sem" | "sd"
    level: float = 0.95,   # 置信水平（用于 mode="ci"）
    n: int | None = None,  # 每个(mean,std)对应的样本量
    ylabel: str = 'Accuracy',
    fontsize: int = 16     # 统一字号控制
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

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, sub in df.groupby("Method"):
        sub = sub.sort_values("sr", ascending=False)
        x = sub["sr"].to_numpy()
        m = sub["data"].to_numpy()
        s = sub["std"].to_numpy()

        # 计算上下界
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

        # 画线 + 阴影
        ax.plot(x, m, marker="o", linewidth=2.0,
                label=legend_name.get(method, str(method)),
                color=color_map[str(method)], zorder=3)
        ax.fill_between(x, low, high, alpha=0.15,
                        color=color_map[str(method)], zorder=2)

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
    """
    绘制以方法为横轴的柱状图（支持误差棒、文字注释、可旋转标签、自动换行）

    参数：
    - df: DataFrame，需包含 ["Method", "data", "std"]
    - mode: 误差类型，"sd" 或 "sem"
    - n: 样本数，用于 SEM
    - wrap_width: 超过此长度自动换行（如 10 -> 超过10字符自动分行）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import textwrap

    # 若 df 含重复 Method，则聚合
    grouped = (
        df.groupby("Method")
        .agg({"data": "mean", "std": "mean"})
        .reset_index()
    )

    # 提取方法与统计值
    methods = grouped["Method"].astype(str).tolist()

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
    from line_chart_data import portion_data_pcc
    identifier_po = portion_data_pcc.identifier
    portion_pcc_accuracy_dic = portion_data_pcc.accuracy
    
    df = build_dataframe(portion_pcc_accuracy_dic, identifier_po)
    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))
    plot_data_with_band(df, mode="sem", level=0.95, n=30)
    
    srs = portion_pcc_accuracy_dic['sr']
    accuracies = portion_pcc_accuracy_dic['data']
    stds = portion_pcc_accuracy_dic['std']
    portion_pcc_bpes = balanced_performance_efficiency_single_points(srs, accuracies)
    portion_pcc_bpes_std = balanced_performance_efficiency_single_points(srs, stds)
    
    portion_pcc_bpes_dic = portion_pcc_accuracy_dic.copy()
    portion_pcc_bpes_dic['data'] = portion_pcc_bpes
    portion_pcc_bpes_dic['std'] = portion_pcc_bpes_std
    
    df = build_dataframe(portion_pcc_bpes_dic, identifier_po)
    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))
    plot_data_with_band(df, mode="sem", level=0.95, n=30, ylabel='BPE(Balanced Performance Efficiency)')
    
def mbpe_portion():
    from line_chart_data import portion_data_pcc
    identifier_po = portion_data_pcc.identifier
    portion_pcc_accuracy_dic = portion_data_pcc.accuracy
    
    srs = portion_pcc_accuracy_dic['sr']
    accuracies = portion_pcc_accuracy_dic['data']
    stds = portion_pcc_accuracy_dic['std']
    
    mbpes_1 = balanced_performance_efficiency_multiple_points(srs[0:7], accuracies[0:7])
    stds_1 = balanced_performance_efficiency_multiple_points(srs[0:7], stds[0:7])
    mbpes_2 = balanced_performance_efficiency_multiple_points(srs[7:14], accuracies[7:14])
    stds_2 = balanced_performance_efficiency_multiple_points(srs[7:14], stds[7:14])
    
    portion_pcc_mbpes_dic = portion_pcc_accuracy_dic.copy()
    portion_pcc_mbpes_dic['data'] = [mbpes_1, mbpes_2]
    portion_pcc_mbpes_dic['std'] = [stds_1, stds_2]
    portion_pcc_mbpes_dic['sr'] = [1, 1]
    
    df = build_dataframe(portion_pcc_mbpes_dic, identifier_po)
    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))
    plot_bar_by_method(df, mode="sem", n=30, xtick_rotation=90, wrap_width=15)
    
# %% main
if __name__ == "__main__":
    sbpe_portion()
    mbpe_portion()