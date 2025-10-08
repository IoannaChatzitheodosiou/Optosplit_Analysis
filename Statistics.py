import csv
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import yaml
from scipy.stats import f_oneway
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression


def plot_and_stats(yaml_file, measurement_name):
    all_data = pd.DataFrame(columns=["strain", measurement_name])
    particle_counts = {}

    # Load colours.yaml
    with open("colours.yaml") as colorfile:
        colour_map = yaml.safe_load(colorfile)

    

    with open(yaml_file) as f:
        measurements = yaml.safe_load(f)
        for strain, measurements_list in measurements.items():
            for measurement in measurements_list:
                i = 0
                with open (measurement) as csvfile:
                    data=csvfile.read()
                    data = data.split(';')
                    for particle in data:
                        try:
                            i += 1
                            particle = float(particle)
                            all_data.loc[len(all_data)] = {'strain': strain, measurement_name:particle}
                        except:
                            pass
                    particle_counts[strain] = i
        # Prepare palette using the colour_map for the strains present in the data
        strains_in_data = all_data['strain'].unique()
        strains_palette = {strain: colour_map.get(strain, "#333333") for strain in strains_in_data}
    
    # One-way ANOVA
    groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
    f_stat, p_value = f_oneway(*groups)

    print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")

    g = sns.catplot(data=all_data, x="strain", y=measurement_name, hue="strain", kind='box', palette=strains_palette, showcaps=True, boxprops={'facecolor':'none', "zorder": 1},whiskerprops={'zorder': 1},
        medianprops={'zorder': 2})
    g.fig.set_size_inches(8, 8)
    ax = g.ax
    sns.swarmplot(data=all_data, x="strain", y=measurement_name, hue="strain", dodge=False, palette=strains_palette, size=3, ax=ax, zorder=3)


    # Fit linear regression
    strain_to_num = {strain: i for i, strain in enumerate(all_data['strain'].unique())}
    all_data['strain_num'] = all_data['strain'].map(strain_to_num)
    X = all_data['strain_num'].values.reshape(-1, 1)
    y = all_data[measurement_name].values
    model = LinearRegression().fit(X, y)
    x_pred = np.arange(len(strain_to_num)).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    ax.plot(list(strain_to_num.values()), y_pred, color='black', linewidth=2, label='Linear regression')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the equation on the plot
    coef = model.coef_
    intercept = model.intercept_
    equation = f"y = {coef[0]:.3f}x + {intercept:.3f}"
    ax.text(0.15, 0.10, equation, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7))

    # Set the title using the FacetGrid's figure
    g.fig.suptitle(f"{measurement_name}\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g} (One-way ANOVA)", fontsize=12, y=0.95)
    g.fig.subplots_adjust(top=0.88)

    plt.tight_layout()
    plt.savefig(f"{measurement_name}_violin_plot.png")
    plt.show()
    plt.close()

    sns.barplot(x=list(particle_counts.keys()), y=list(particle_counts.values()), palette=strains_palette)
    plt.xticks(rotation=90)
    plt.ylabel('Number of particles analyzed')
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_particle_count.png")
    plt.show()
    plt.close()

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
    plt.title(f"Pairwise p-value table (t-test) {measurement_name}")
    plt.ylabel("Strain")
    plt.xlabel("Strain")
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_pvalue_heatmap.png")
    plt.show()
    plt.close()

def main():
    yaml_file = 'csvpaths.yaml'
    measurement_name = 'Fluorescence Ratio'
    plot_and_stats(yaml_file, measurement_name)
    yaml_file = 'areas.yaml'
    measurement_name = 'Particle Size'
    plot_and_stats(yaml_file, measurement_name)

if __name__ == "__main__":
    main()