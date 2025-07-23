# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 02:22:05 2025

@author: 18307
"""

# Re-import necessary libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt

# Re-define the data
data = {
    "Identifier": [
        "FN", "", "", "", "",
        "Recovered FN_lamda_0.1", "", "", "", "",
        "Recovered FN_lamda_0.25", "", "", "", "",
        "Recovered FN_lamda_0.5", "", "", "", ""
    ],
    "accuracy": [
        88.33055073, 82.25789727, 74.14004735, 60.48042515, 51.78654948,
        86.69695528, 84.12478193, 78.87308137, 71.45834595, 65.58994455,
        87.14258197, 81.84891449, 74.03558278, 65.28877124, 56.63437112,
        88.34461659, 81.81684963, 74.31009495, 67.6140105, 54.63921545
    ],
    "sr": [0.5, 0.3, 0.2, 0.1, 0.07] * 4,
    "std": [
        5.394329607, 7.322753465, 9.1606785, 12.26651809, 13.07545155,
        6.420139222, 7.345841251, 9.370771512, 9.280781633, 9.982370807,
        6.453998906, 7.432132521, 8.951404401, 10.23497552, 13.29572735,
        5.868706365, 8.051348588, 9.060998389, 9.704374871, 11.95983334
    ]
}

df = pd.DataFrame(data)
df['Identifier'] = df['Identifier'].replace('', pd.NA).ffill()

# Create the plot
plt.figure(figsize=(10, 6))

# Plot FN with thicker lines and error bars
fn_data = df[df["Identifier"] == "FN"]
plt.errorbar(
    fn_data["sr"], fn_data["accuracy"], yerr=fn_data["std"],
    marker='o', markersize=8, linewidth=3.5, capsize=10, label="FN", color="orange"
)

# Plot recovered methods with error bars
recovered_colors = ['teal', 'mediumturquoise', 'cadetblue']
for i, label in enumerate(["Recovered FN_lamda_0.1", "Recovered FN_lamda_0.25", "Recovered FN_lamda_0.5"]):
    sub_data = df[df["Identifier"] == label]
    plt.errorbar(
        sub_data["sr"], sub_data["accuracy"], yerr=sub_data["std"],
        marker='o', markersize=8, linewidth=3.5, capsize=10,
        label=label, color=recovered_colors[i]
    )

# Customize the plot
plt.xlabel("Selection Rate (for extraction of subnetworks)", fontsize=16, loc='center')
plt.ylabel("Accuracy (%)", fontsize=16, loc='center')
# plt.title("Accuracy vs Selection Rate", fontsize=16)
plt.gca().invert_xaxis()  # Reverse x-axis
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.show()

# 
df = pd.DataFrame(data)
df['Identifier'] = df['Identifier'].replace('', pd.NA).ffill()

# Create bar plots for each selection rate in a single row
selection_rates = sorted(df['sr'].unique(), reverse=True)
identifiers = df['Identifier'].unique()
colors = ['orange', 'teal', 'mediumturquoise', 'cadetblue']

# Create bar plots with legends for each method instead of xtick labels

fig, axes = plt.subplots(1, len(selection_rates), figsize=(20, 5), sharey=True)

for i, sr in enumerate(selection_rates):
    ax = axes[i]
    sub_df = df[df['sr'] == sr]
    
    for j, identifier in enumerate(identifiers):
        row = sub_df[sub_df['Identifier'] == identifier]
        if not row.empty:
            ax.bar(j, row['accuracy'].values[0], yerr=row['std'].values[0], 
                   capsize=5, color=colors[j], label=identifier if i == 0 else "")
    
    ax.set_title(f'SR = {sr}', fontsize=20)
    ax.tick_params(axis='x', labelsize=0)
    ax.tick_params(axis='y', labelsize=20)
    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=20)

# Add legend only once for clarity
axes[0].legend(fontsize=20, loc='lower center', bbox_to_anchor=(2.8, 1.15), ncol=4)

plt.tight_layout()
plt.show()

# 
df = pd.DataFrame(data)
df['Identifier'] = df['Identifier'].replace('', pd.NA).ffill()

# Create the plot for std only
plt.figure(figsize=(10, 6))

identifiers = df['Identifier'].unique()
colors = ['orange', 'teal', 'mediumturquoise', 'cadetblue']

for i, identifier in enumerate(identifiers):
    sub_df = df[df['Identifier'] == identifier]
    plt.plot(sub_df['sr'], sub_df['std'], marker='o', markersize=8, linewidth=3, label=identifier, color=colors[i])

# Customize the plot
plt.xlabel("Selection Rate (for extraction of subnetworks)", fontsize=16, loc='center')
plt.ylabel("Standard Deviation (%)", fontsize=16)
plt.gca().invert_xaxis()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.show()