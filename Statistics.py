import csv
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import yaml
from scipy.stats import f_oneway

all_data = pd.DataFrame(columns=["strain", "fluorescence ratio"])

with open('csvpaths.yaml') as f:
    measurements = yaml.safe_load(f)
    for strain, measurements_list in measurements.items():
        for measurement in measurements_list:
            with open (measurement) as csvfile:
                data=csvfile.read()
                data = data.split(';')
                for particle in data:
                    try:
                        particle = float(particle)
                        all_data.loc[len(all_data)] = {'strain': strain, 'fluorescence ratio':particle}
                    except:
                        pass

sns.catplot(data=all_data, x="strain", y="fluorescence ratio", hue="strain", kind="violin")
plt.show()

# One-way ANOVA
groups = [group["fluorescence ratio"].values for name, group in all_data.groupby("strain")]
f_stat, p_value = f_oneway(*groups)

print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")

# Pairwise p-value table
import numpy as np
from itertools import combinations

strains = all_data['strain'].unique()
pval_matrix = pd.DataFrame(np.ones((len(strains), len(strains))), index=strains, columns=strains)


# Use t-test for pairwise comparisons
from scipy.stats import ttest_ind
for strain1, strain2 in combinations(strains, 2):
    group1 = all_data[all_data['strain'] == strain1]['fluorescence ratio'].values
    group2 = all_data[all_data['strain'] == strain2]['fluorescence ratio'].values
    _, pval = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    pval_matrix.loc[strain1, strain2] = pval
    pval_matrix.loc[strain2, strain1] = pval

print("\nPairwise p-value table (t-test):")
print(pval_matrix.round(4))

# Plot the p-value table as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pval_matrix, annot=True, fmt=".4f", cmap="magma", cbar_kws={"label": "p-value"})
plt.title("Pairwise p-value table (one-way ANOVA)")
plt.ylabel("Strain")
plt.xlabel("Strain")
plt.tight_layout()
plt.show()

areas_data = pd.DataFrame(columns=["strain", "area"])
with open('areas.yaml') as f:
    areas = yaml.safe_load(f)
    for strain, areas_list in areas.items():
        for area in areas_list:
            with open (area) as csvfile:
                data=csvfile.read()
                data = data.split(';')
                for particle in data:
                    try:
                        particle = float(particle)
                        all_data.loc[len(all_data)] = {'strain': strain, 'area':particle}
                    except:
                        pass

sns.catplot(data=areas_data, x="strain", y="area", hue="strain", kind="violin")
plt.show()


