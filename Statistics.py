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


class PlotsAndStats:
    def __init__(self, data_paths, measurement_name, colours_file, settings_file):
        self.data_paths = data_paths
        self.measurement_name = measurement_name
        self.colours = self.load_colours(colours_file)
        self.settings = self.load_settings(settings_file)
        self.all_data = pd.DataFrame(columns=["strain", measurement_name])
        self.particle_counts = {}

    def remove_outliers(self, data, column, method='IQR', threshold=1.5):
        """
        Remove outliers from data using IQR method.
        
        Parameters:
        - data: DataFrame containing the data
        - column: Name of the column to check for outliers
        - method: Method to use ('IQR' or 'zscore')
        - threshold: Multiplier for IQR (default 1.5) or z-score threshold (default 3)
        
        Returns:
        - DataFrame with outliers removed
        """
        if method == 'IQR':
            cleaned_data = pd.DataFrame()
            for strain, group in data.groupby('strain'):
                Q1 = group[column].quantile(0.25)
                Q3 = group[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                filtered_group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
                outliers_removed = len(group) - len(filtered_group)
                print(f"{strain}: Removed {outliers_removed} outliers (lower bound: {lower_bound:.2f}, upper bound: {upper_bound:.2f})")
                
                cleaned_data = pd.concat([cleaned_data, filtered_group])
            return cleaned_data
        else:
            return data

    def load_colours(self, colours_file):
        with open(colours_file) as colorfile:
            return yaml.safe_load(colorfile)

    def load_settings(self, settings_file):
        with open(settings_file) as settingfile:
            return yaml.safe_load(settingfile)
        
    def load_data(self):
        with open(self.data_paths) as f:
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
                                self.all_data.loc[len(self.all_data)] = {'strain': strain, self.measurement_name:particle}
                            except:
                                pass
                        self.particle_counts[strain] = i
        
        # Remove outliers after loading all data
        print("\nRemoving outliers using IQR method (threshold=1.5):")
        self.all_data = self.remove_outliers(self.all_data, self.measurement_name)
        
    def generate_palette(self):
        strains_in_data = self.all_data['strain'].unique()
        return {strain: self.colours.get(strain, "#333333") for strain in strains_in_data}

    def normal_distribution_test(self):
        # Check normality for each group using Shapiro-Wilk test
        print("\nShapiro-Wilk normality test per strain:")
        shapiro_results = []
        non_normally_distributed = 0
        for strain, group in self.all_data.groupby("strain"):
            stat, p = shapiro(group[self.measurement_name].values)
            print(f"{strain}: W={stat:.4f}, p-value={p:.4g}")
            shapiro_results.append(f"{strain}: W={stat:.4f}, p-value={p:.4g}")
            if p < 0.05:
                non_normally_distributed += 1

        # Export Shapiro-Wilk results to txt
        with open(f"{self.measurement_name}_shapiro_wilk_2  .txt", "w") as txtfile:
            txtfile.write("Shapiro-Wilk normality test per strain:\n")
            for line in shapiro_results:
                txtfile.write(line + "\n")

        if non_normally_distributed == 0:
            # One-way ANOVA
            groups = [group[self.measurement_name].values for name, group in self.all_data.groupby("strain")]
            f_stat, p_value = f_oneway(*groups)
            print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")
        else:
            # Kruskal-Wallis test (replaces ANOVA)
            groups = [group[self.measurement_name].values for name, group in self.all_data.groupby("strain")]
            h_stat, p_value = kruskal(*groups)

            print(f"Kruskal-Wallis test:\nH-statistic = {h_stat:.4f}, p-value = {p_value:.4g}")

                    
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
        
        # Remove outliers using IQR method
        print("\nRemoving outliers using IQR method (threshold=1.5):")
        cleaned_data = pd.DataFrame()
        for strain, group in all_data.groupby('strain'):
            Q1 = group[measurement_name].quantile(0.25)
            Q3 = group[measurement_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            filtered_group = group[(group[measurement_name] >= lower_bound) & (group[measurement_name] <= upper_bound)]
            outliers_removed = len(group) - len(filtered_group)
            print(f"{strain}: Removed {outliers_removed} outliers (lower bound: {lower_bound:.2f}, upper bound: {upper_bound:.2f})")
            
            cleaned_data = pd.concat([cleaned_data, filtered_group])
        
        all_data = cleaned_data
        
        # Prepare palette using the colour_map for the strains present in the data
        strains_in_data = all_data['strain'].unique()
        strains_palette = {strain: colour_map.get(strain, "#333333") for strain in strains_in_data}
    
     ##settings
    with open ("settings.yaml") as settingfile:
        settings = yaml.safe_load(settingfile)
        custom_order = settings['custom_order']
        linear_regression = settings['linear_regression']
    
    # Check normality for each group using Shapiro-Wilk test
    print("\nShapiro-Wilk normality test per strain:")
    shapiro_results = []
    non_normally_distributed = 0
    for strain, group in all_data.groupby("strain"):
        stat, p = shapiro(group[measurement_name].values)
        print(f"{strain}: W={stat:.4f}, p-value={p:.4g}")
        shapiro_results.append(f"{strain}: W={stat:.4f}, p-value={p:.4g}")
        if p < 0.05:
            non_normally_distributed += 1

    # Export Shapiro-Wilk results to txt
    with open(f"{measurement_name}_shapiro_wilk.txt", "w") as txtfile:
        txtfile.write("Shapiro-Wilk normality test per strain:\n")
        for line in shapiro_results:
            txtfile.write(line + "\n")

    if non_normally_distributed == 0:
        # One-way ANOVA
        groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
        f_stat, p_value = f_oneway(*groups)
        print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")
    else:
        # Kruskal-Wallis test (replaces ANOVA)
        groups = [group[measurement_name].values for name, group in all_data.groupby("strain")]
        h_stat, p_value = kruskal(*groups)

        print(f"Kruskal-Wallis test:\nH-statistic = {h_stat:.4f}, p-value = {p_value:.4g}")


    # Plot up to 100 representative spots, preserving strain proportions.
    max_spots = 1000
    if len(all_data) > max_spots:
        strain_counts = all_data['strain'].value_counts()
        target_per_strain = (strain_counts / strain_counts.sum() * max_spots)
        base_counts = np.floor(target_per_strain).astype(int)
        remainders = target_per_strain - base_counts

        for strain in strain_counts.index:
            if base_counts[strain] == 0:
                base_counts[strain] = 1

        total_selected = int(base_counts.sum())

        if total_selected > max_spots:
            for strain in base_counts.sort_values(ascending=False).index:
                if total_selected <= max_spots:
                    break
                if base_counts[strain] > 1:
                    base_counts[strain] -= 1
                    total_selected -= 1
        elif total_selected < max_spots:
            for strain in remainders.sort_values(ascending=False).index:
                if total_selected >= max_spots:
                    break
                if base_counts[strain] < strain_counts[strain]:
                    base_counts[strain] += 1
                    total_selected += 1

        sampled_groups = []
        for strain, group in all_data.groupby('strain'):
            n_select = min(int(base_counts.get(strain, 0)), len(group))
            if n_select > 0:
                sampled_groups.append(group.sample(n=n_select, random_state=42))

        plot_data = pd.concat(sampled_groups, ignore_index=True)
    else:
        plot_data = all_data

    plt.figure(figsize=(8, 8))
    sns.catplot(data=plot_data, x="strain", y=measurement_name, hue="strain", palette=strains_palette, legend=False)
    ax = plt.gca()

    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
  
    # Show the median as a black line for each strain
    medians = all_data.groupby("strain")[measurement_name].median()
    for i, strain in enumerate(all_data['strain'].unique()):
        median = medians[strain]
        ax.hlines(median, i - 0.2, i + 0.2, color='red', linewidth=2, zorder=5, alpha=0.7, linestyles='--')

    if linear_regression==True:
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



    # Set the title
    plt.suptitle(f"{measurement_name}\nH-statistic = {h_stat:.4f}, p-value = {p_value:.4g} (Kruskal-Wallis test)", fontsize=12, y=0.95)
    plt.subplots_adjust(top=0.88)

    plt.tight_layout()
    plt.savefig(f"{measurement_name}_scatterplot.svg")
    plt.savefig(f"{measurement_name}_scatterplot.eps")

    
    plt.show()
    plt.close()

    sns.barplot(x=list(particle_counts.keys()), y=list(particle_counts.values()), palette=strains_palette)
    plt.xticks(rotation=90)
    plt.ylabel('Number of particles analyzed')
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_particle_count.svg")
    plt.savefig(f"{measurement_name}_particle_count.eps")
    plt.show()
    plt.close()



    # Pairwise comparisons
    strains = all_data['strain'].unique()

    dunn_results = posthoc_dunn(
        all_data, 
        val_col=measurement_name, 
        group_col='strain', 
        p_adjust='bonferroni'
    )

    # Plot the p-value table as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(dunn_results.loc[custom_order, custom_order], annot=True, fmt=".4f", cmap="magma", cbar_kws={"label": "p-value"})
    plt.title(f"Post hoc comparisson (Dunn-Bonferroni) {measurement_name}")
    plt.tight_layout()
    plt.savefig(f"{measurement_name}_pvalue_heatmap.svg")
    plt.savefig(f"{measurement_name}_pvalue_heatmap.eps")

    plt.show()
    plt.close()

def main():
    yaml_file = 'csvpaths.yaml'
    measurement_name = 'Fluorescence Ratio'
    plot_and_stats(yaml_file, measurement_name)
    yaml_file = 'areas.yaml'
    measurement_name = 'Particle Size'
    plot_and_stats(yaml_file, measurement_name)
    plots = PlotsAndStats('areas.yaml', 'Particle Size', 'colours.yaml', 'settings.yaml')
    plots.load_data()
    plots.normal_distribution_test()

if __name__ == "__main__":
    main()