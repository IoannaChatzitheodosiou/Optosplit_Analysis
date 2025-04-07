import csv
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import yaml
from scipy.stats import f_oneway

all_data = pd.DataFrame(columns=["strain", "measurement"])

with open('565Tirf/csvpaths.yaml') as f:
    measurements = yaml.safe_load(f)
    for strain, measurements_list in measurements.items():
        for measurement in measurements_list:
            with open (measurement) as csvfile:
                data=csvfile.read()
                data = data.split(';')
                for particle in data:
                    try:
                        particle = float(particle)
                        all_data.loc[len(all_data)] = {'strain': strain, 'measurement':particle}
                    except:
                        pass

sns.swarmplot(data=all_data, x="strain", y="measurement", hue="strain")
plt.show()

# One-way ANOVA
groups = [group["measurement"].values for name, group in all_data.groupby("strain")]
f_stat, p_value = f_oneway(*groups)

print(f"One-way ANOVA:\nF-statistic = {f_stat:.4f}, p-value = {p_value:.4g}")
            



