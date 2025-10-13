import csv
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import yaml
import numpy as np
from itertools import combinations
from scipy.stats import kruskal, shapiro, f_oneway, mannwhitneyu
from scikit_posthocs import posthoc_dunn
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
    
    # Check normality for each group using Shapiro-Wilk test
    print("\nShapiro-Wilk normality test per strain:")
    shapiro_results = []
    for strain, group in all_data.groupby("strain"):
        stat, p = shapiro(group[measurement_name].values)
        print(f"{strain}: W={stat:.4f}, p-value={p:.4g}")
        shapiro_results.append(f"{strain}: W={stat:.4f}, p-value={p:.4g}")

    # Export Shapiro-Wilk results to txt
    with open(f"{measurement_name}_shapiro_wilk.txt", "w") as txtfile:
        txtfile.write("Shapiro-Wilk normality test per strain:\n")
        for line in shapiro_results:
            txtfile.write(line + "\n")

 

    # Kruskal-Wallis test (replaces ANOVA)
    groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
    h_stat, p_value = kruskal(*groups)

    print(f"Kruskal-Wallis test:\nH-statistic = {h_stat:.4f}, p-value = {p_value:.4g}")


    g = sns.catplot(data=all_data, x="strain", y=measurement_name, hue="strain", kind='swarm', palette=strains_palette)
    g.fig.set_size_inches(8, 8)
    ax = g.ax

    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
  
    # Show the median as a black line for each strain
    medians = all_data.groupby("strain")[measurement_name].median()
    for i, strain in enumerate(all_data['strain'].unique()):
        median = medians[strain]
        ax.hlines(median, i - 0.2, i + 0.2, color='black', linewidth=4, zorder=5, alpha=0.5)


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

    r_squared = model.score(X, y)
    ##For showing R^2 on the plot
    # ax.text(0.20, 0.10, f"$R^2$ = {r_squared:.3f}", transform=ax.transAxes, fontsize=10, 
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(facecolor='white', alpha=0.7))

    #For printing R^2 in the terminal
    print (f"For {measurement} Linear regression R^2 = {r_squared:.4f}")
     # Export R squared to txt
    with open(f"{measurement_name}_r_squared.txt", "w") as txtfile:
        txtfile.write(f"R^2 for {measurement_name}: {r_squared:.4f}\n")



    # Set the title using the FacetGrid's figure
    g.fig.suptitle(f"{measurement_name}\nH-statistic = {h_stat:.4f}, p-value = {p_value:.4g} (Kruskal-Wallis test)", fontsize=12, y=0.95)
    g.fig.subplots_adjust(top=0.88)

    plt.tight_layout()
    plt.savefig(f"{measurement_name}_scatterplot.pdf")
    plt.show()
    plt.close()

    sns.barplot(x=list(particle_counts.keys()), y=list(particle_counts.values()), palette=strains_palette)
    plt.xticks(rotation=90)
    plt.ylabel('Number of particles analyzed')
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_particle_count.pdf")
    plt.show()
    plt.close()

    # One-way ANOVA
    groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
    f_stat, p_value = f_oneway(*groups)

    print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")

    # Pairwise comparisons
    strains = all_data['strain'].unique()

    # # Use Mann-Whitney U test for pairwise comparisons
    # for strain1, strain2 in combinations(strains, 2):
    #     group1 = all_data[all_data['strain'] == strain1][measurement_name].values
    #     group2 = all_data[all_data['strain'] == strain2][measurement_name].values
    #     _, pval = mannwhitneyu(group1, group2, alternative='two-sided')
    #     pval_matrix.loc[strain1, strain2] = pval
    #     pval_matrix.loc[strain2, strain1] = pval

    dunn_results = posthoc_dunn(
        all_data, 
        val_col=measurement_name, 
        group_col='strain', 
        p_adjust='bonferroni'
    )
    ##reordering
    custom_order = ["RML", "Gen  1", "Gen  2", "Gen  3", "Gen  4", "Gen  5","Gen  6", "Gen  7", "Gen  8", "Gen  9", "Gen 10"]
    # Plot the p-value table as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(dunn_results.loc[custom_order, custom_order], annot=True, fmt=".4f", cmap="magma", cbar_kws={"label": "p-value"})
    plt.title(f"Post hoc comparisson (Dunn-Bonferroni) {measurement_name}")
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_pvalue_heatmap.pdf")
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