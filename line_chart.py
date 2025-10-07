# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 21:41:09 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Refactored on Thu Aug 14 2025

- 单一数据定义入口
- 自动补齐 Identifier 并派生 Method（短标签）与 MethodFull（长标题）
- 通用绘图函数（折线 + 误差棒、SR 分面柱状图、std 折线）
- 数据一致性校验与友好报错
"""
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) 原始数据（单一入口）
# -----------------------------
from line_chart_data import summary_data_pcc
identifier_sm = summary_data_pcc.identifier
summary_pcc_accuracy = summary_data_pcc.accuracy
summary_pcc_f1score = summary_data_pcc.f1score

from line_chart_data import summary_data_plv
identifier_sm = summary_data_plv.identifier
summary_plv_accuracy = summary_data_plv.accuracy
summary_plv_f1score = summary_data_plv.f1score

# 这些会在 build_dataframe 后被自动设置
legend_name: dict[str, str] = {}
method_order_short: List[str] = []
color_map: dict[str, tuple] = {}


# -----------------------------
# 2) 数据整理与验证
# -----------------------------
def _parse_identifiers(ident_list: List[str]) -> tuple[list[str], list[str]]:
    """
    将 "short, long" 形式的条目解析成短/长标签。
    缺逗号时，短=全文，长=全文；多逗号时，仅按第一个逗号切分。
    """
    shorts, longs = [], []
    for raw in ident_list:
        parts = [p.strip() for p in str(raw).split(":", 1)]
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

# -----------------------------
# 3) 公用：按 SR 设刻度并加竖直虚线
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

# -----------------------------
# 4) 绘图工具函数
# -----------------------------
def plot_accuracy_lines(df: pd.DataFrame, ylabel='Accuracy') -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, sub in df.groupby("Method"):
        sub = sub.sort_values("sr", ascending=False)
        ax.errorbar(
            sub["sr"], sub["data"], yerr=sub["std"],
            marker="o", linewidth=2.0, capsize=4,
            label=legend_name.get(method, str(method)),
            color=color_map[str(method)], zorder=3
        )

    ax.set_xlabel("Selection Rate (for extraction of subnetworks)", fontsize=16)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=16)
    ax.invert_xaxis()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)

    # 只按 SR 标定刻度，并在刻度处加竖线
    _apply_sr_ticks_and_vlines(ax, df["sr"])

    fig.tight_layout()
    plt.show()

def plot_bar_by_sr(df: pd.DataFrame, ylabel='magnitude') -> None:
    selection_rates = sorted(df["sr"].unique(), reverse=True)
    methods = list(df["Method"].cat.categories)

    fig, axes = plt.subplots(1, len(selection_rates), figsize=(20, 5), sharey=True)
    if len(selection_rates) == 1:
        axes = [axes]

    for i, sr in enumerate(selection_rates):
        ax = axes[i]
        sub = df[df["sr"] == sr].set_index("Method").reindex(methods)
        heights = sub["data"].to_numpy()
        yerr = sub["std"].to_numpy()
        colors = [color_map[m] for m in methods]

        ax.bar(range(len(methods)), heights, yerr=yerr, capsize=4, color=colors)
        ax.set_title(f"SR = {sr}", fontsize=16)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([legend_name.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=10)
        if i == 0:
            ax.set_ylabel(f"{ylabel} (%)", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.show()

def plot_std_lines(
    df: pd.DataFrame, 
    ylabel: str = 'Standard Deviation',
    figsize=(10, 6),
    fontsize: int = 16   # 统一字号控制
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for method, sub in df.groupby("Method"):
        sub = sub.sort_values("sr", ascending=False)
        ax.plot(
            sub["sr"], sub["std"], marker="o", linewidth=2.0,
            label=legend_name.get(method, str(method)),
            color=color_map[str(method)], zorder=3
        )

    ax.set_xlabel("Selection Rate (for extraction of subnetworks)", fontsize=fontsize)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=fontsize)
    ax.invert_xaxis()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.tick_params(axis="x", labelsize=fontsize*0.9)
    ax.tick_params(axis="y", labelsize=fontsize*0.9)

    ax.legend(fontsize=fontsize*0.9, title_fontsize=fontsize)

    # 只按 SR 标定刻度，并在刻度处加竖线
    _apply_sr_ticks_and_vlines(ax, df["sr"])

    fig.tight_layout()
    plt.show()

def plot_accuracy_with_band(
    df: pd.DataFrame,
    mode: str = "sem",     # "ci" | "sem" | "sd"
    level: float = 0.95,   # 置信水平（用于 mode="ci"）
    n: int | None = None,  # 每个(mean,std)对应的样本量
    ylabel: str = 'Accuracy',
    figsize=(10, 6),
    fontsize: int = 16     # 统一字号控制
) -> None:
    """
    以“曲线 + 半透明误差带”的方式展示 accuracy。
    - mode="ci"  : 均值 ± t*z * (std/sqrt(n))，优先用 t 分布；无 scipy 时用正态近似。
    - mode="sem" : 均值 ± (std/sqrt(n))。若 n=None，退化为 SD 带并提示。
    - mode="sd"  : 均值 ± std（仅可视化，不建议用于论文正文）。
    """
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

    fig, ax = plt.subplots(figsize=figsize)
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

# -----------------------------
# 5) 主流程
# -----------------------------
def main_portion():
    # -----------------------------
    # accuracy; pcc
    # -----------------------------
    from line_chart_data import portion_data_pcc
    identifier_po = portion_data_pcc.identifier
    portion_pcc_accuracy = portion_data_pcc.accuracy
    portion_pcc_f1score = portion_data_pcc.f1score
    
    df = build_dataframe(portion_pcc_accuracy, identifier_po)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="sem", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # f1 score; pcc
    df = build_dataframe(portion_pcc_f1score, identifier_po)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="sem", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # # -----------------------------
    # # accuracy; plv
    # # -----------------------------
    # from line_chart_data import portion_data_plv
    # identifier_po = portion_data_plv.identifier
    # portion_plv_accuracy = portion_data_plv.accuracy
    # portion_plv_f1score = portion_data_plv.f1score
    
    # df = build_dataframe(portion_plv_accuracy, identifier_po)

    # print("Methods (short order):", list(df["Method"].cat.categories))
    # print("SRs:", sorted(df["sr"].unique(), reverse=True))
    # print(df.head(10))

    # # 图 1：误差带（把 n 改成你的真实重复次数）
    # plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # # 图 2：Std 折线
    # plot_std_lines(df)

    # # f1 score; plv
    # df = build_dataframe(portion_plv_f1score, identifier_po)

    # print("Methods (short order):", list(df["Method"].cat.categories))
    # print("SRs:", sorted(df["sr"].unique(), reverse=True))
    # print(df.head(10))

    # # 图 1：误差带（把 n 改成你的真实重复次数）
    # plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # # 图 2：Std 折线
    # plot_std_lines(df)

def main_summary():
    # -----------------------------
    # accuracy; pcc
    # -----------------------------
    df = build_dataframe(summary_pcc_accuracy, identifier_sm)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="sem", n=30, figsize=(11,6))

    # 图 2：Std 折线
    plot_std_lines(df, figsize=(11,6))

    # f1 score; pcc
    df = build_dataframe(summary_pcc_f1score, identifier_sm)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="sem", n=30, ylabel='F1 score', figsize=(11,6))

    # 图 2：Std 折线
    plot_std_lines(df, figsize=(11,6))
    
    # # -----------------------------
    # # accuracy; plv
    # # -----------------------------
    # df = build_dataframe(summary_plv_accuracy, identifier_sm)

    # print("Methods (short order):", list(df["Method"].cat.categories))
    # print("SRs:", sorted(df["sr"].unique(), reverse=True))
    # print(df.head(10))

    # # 图 1：误差带（把 n 改成你的真实重复次数）
    # plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # # 图 2：Std 折线
    # plot_std_lines(df)

    # # f1 score; plv
    # df = build_dataframe(summary_plv_f1score, identifier_sm)

    # print("Methods (short order):", list(df["Method"].cat.categories))
    # print("SRs:", sorted(df["sr"].unique(), reverse=True))
    # print(df.head(10))

    # # 图 1：误差带（把 n 改成你的真实重复次数）
    # plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # # 图 2：Std 折线
    # plot_std_lines(df)

def main_appendix():
    # %% graph laplacian
    from line_chart_data import appendix_data_pcc_gl
    identifier_gl = appendix_data_pcc_gl.identifier
    gl_pcc_accuracy = appendix_data_pcc_gl.accuracy
    gl_pcc_f1score = appendix_data_pcc_gl.f1score

    from line_chart_data import appendix_data_plv_gl
    identifier_gl = appendix_data_plv_gl.identifier
    gl_plv_accuracy = appendix_data_plv_gl.accuracy
    gl_plv_f1score = appendix_data_plv_gl.f1score
    # -----------------------------
    # accuracy; pcc; gl
    # -----------------------------
    df = build_dataframe(gl_pcc_accuracy, identifier_gl)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; pcc; gl
    # -----------------------------
    df = build_dataframe(gl_pcc_f1score, identifier_gl)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # -----------------------------
    # accuracy; plv; gl
    # -----------------------------
    df = build_dataframe(gl_plv_accuracy, identifier_gl)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; plv; gl
    # -----------------------------
    df = build_dataframe(gl_plv_f1score, identifier_gl)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # %% graph laplacian denoising
    from line_chart_data import appendix_data_pcc_ld
    identifier_ld = appendix_data_pcc_ld.identifier
    ld_pcc_accuracy = appendix_data_pcc_ld.accuracy
    ld_pcc_f1score = appendix_data_pcc_ld.f1score

    from line_chart_data import appendix_data_plv_ld
    identifier_ld = appendix_data_plv_ld.identifier
    ld_plv_accuracy = appendix_data_plv_ld.accuracy
    ld_plv_f1score = appendix_data_plv_ld.f1score
    # -----------------------------
    # accuracy; pcc; ld
    # -----------------------------
    df = build_dataframe(ld_pcc_accuracy, identifier_ld)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; pcc; ld
    # -----------------------------
    df = build_dataframe(ld_pcc_f1score, identifier_ld)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # -----------------------------
    # accuracy; plv; ld
    # -----------------------------
    df = build_dataframe(ld_plv_accuracy, identifier_ld)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; plv; ld
    # -----------------------------
    df = build_dataframe(ld_plv_f1score, identifier_ld)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # %% spectral graph filtering
    from line_chart_data import appendix_data_pcc_sgf
    identifier_sgf = appendix_data_pcc_sgf.identifier
    sgf_pcc_accuracy = appendix_data_pcc_sgf.accuracy
    sgf_pcc_f1score = appendix_data_pcc_sgf.f1score

    from line_chart_data import appendix_data_plv_sgf
    identifier_sgf = appendix_data_plv_sgf.identifier
    sgf_plv_accuracy = appendix_data_plv_sgf.accuracy
    sgf_plv_f1score = appendix_data_plv_sgf.f1score
    # -----------------------------
    # accuracy; pcc; sgf
    # -----------------------------
    df = build_dataframe(sgf_pcc_accuracy, identifier_sgf)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; pcc; ld
    # -----------------------------
    df = build_dataframe(sgf_pcc_f1score, identifier_sgf)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)
    
    # -----------------------------
    # accuracy; plv; ld
    # -----------------------------
    df = build_dataframe(sgf_plv_accuracy, identifier_sgf)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30)

    # 图 2：Std 折线
    plot_std_lines(df)

    # -----------------------------
    # f1 score; plv; ld
    # -----------------------------
    df = build_dataframe(sgf_plv_f1score, identifier_sgf)

    print("Methods (short order):", list(df["Method"].cat.categories))
    print("SRs:", sorted(df["sr"].unique(), reverse=True))
    print(df.head(10))

    # 图 1：误差带（把 n 改成你的真实重复次数）
    plot_accuracy_with_band(df, mode="ci", level=0.95, n=30, ylabel='F1 score')

    # 图 2：Std 折线
    plot_std_lines(df)

# %% main
if __name__ == "__main__":
    main_portion()
    main_summary()
    # main_appendix()