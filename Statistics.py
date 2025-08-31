import csv
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import yaml
from scipy.stats import f_oneway
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind

def plot_and_stats(yaml_file, measurement_name):
    all_data = pd.DataFrame(columns=["strain", measurement_name])

    with open(yaml_file) as f:
        measurements = yaml.safe_load(f)
        for strain, measurements_list in measurements.items():
            for measurement in measurements_list:
                with open (measurement) as csvfile:
                    data=csvfile.read()
                    data = data.split(';')
                    for particle in data:
                        try:
                            particle = float(particle)
                            all_data.loc[len(all_data)] = {'strain': strain, measurement_name:particle}
                        except:
                            pass

    sns.catplot(data=all_data, x="strain", y=measurement_name, hue="strain", kind="violin")
    plt.show()
    plt.savefig(f"{measurement_name}_violin_plot.png")

    # One-way ANOVA
    groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
    f_stat, p_value = f_oneway(*groups)

    print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")

    # Pairwise comparisons
    strains = all_data['strain'].unique()
    pval_matrix = pd.DataFrame(np.ones((len(strains), len(strains))), index=strains, columns=strains)


    # Use t-test for pairwise comparisons
    for strain1, strain2 in combinations(strains, 2):
        group1 = all_data[all_data['strain'] == strain1][measurement_name].values
        group2 = all_data[all_data['strain'] == strain2][measurement_name].values
        _, pval = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
        pval_matrix.loc[strain1, strain2] = pval
        pval_matrix.loc[strain2, strain1] = pval

    print("\nPairwise p-value table (t-test):")
    print(pval_matrix.round(4))

    # Plot the p-value table as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pval_matrix, annot=True, fmt=".4f", cmap="magma", cbar_kws={"label": "p-value"})
    plt.title("Pairwise p-value table (t-test)")
    plt.ylabel("Strain")
    plt.xlabel("Strain")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{measurement_name}_pvalue_heatmap.png")

def main():
    yaml_file = 'csvpaths.yaml'
    measurement_name = 'fluorescence ratio'
    plot_and_stats(yaml_file, measurement_name)
    yaml_file = 'areas.yaml'
    measurement_name = 'particle size'
    plot_and_stats(yaml_file, measurement_name)

if __name__ == "__main__":
    main()