import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(os.path.dirname("heat_maps/"), exist_ok=True)

df = pd.read_csv("/Users/sharatbhat/Desktop/Tech/VLM_Maps/dataset/code/results/typed_questions_with_maptype3.csv")
df = df[["image_name", "map_type", "geographic_level"]]
df["map_type"] = df["map_type"].apply(lambda x:x[1:])
df["geographic_level"] = df["geographic_level"].apply(lambda x:x[1:])
df = df.drop_duplicates()

# --- Relative Heatmap 1: Normalized by Row (map_type) ---
# Shows the proportion of geographic levels WITHIN each map type
contingency_row_normalized = pd.crosstab(df['geographic_level'], df['map_type'], normalize='columns')

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_row_normalized, annot=True, fmt=".2f", cmap='YlGnBu', vmin=0, vmax=1, annot_kws={"size": 16})
# plt.title('Relative Heatmap (Normalized by Map Type)')
plt.xlabel('Geographical Granularity', fontsize=16)
plt.ylabel('Map Type', fontsize=16)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=16, rotation=45, ha='right') # Adjust fontsize for x-axis labels
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('heat_maps/heatmap_normalized_by_map_type.png')
plt.show()

# --- Relative Heatmap 2: Normalized by Column (geographic_level) ---
# Shows the proportion of map types WITHIN each geographic level
contingency_col_normalized = pd.crosstab(df['geographic_level'], df['map_type'], normalize='index')

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_col_normalized, annot=True, fmt=".2f", cmap='YlGnBu', vmin=0, vmax=1, annot_kws={"size": 16})
# plt.title('Relative Heatmap (Normalized by Geographic Level)')
plt.xlabel('Geographical Granularity', fontsize=16)
plt.ylabel('Map Type', fontsize=16)
plt.xticks(fontsize=16, rotation=45, ha='right') # Adjust fontsize for x-axis labels
plt.yticks(fontsize=16)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.savefig('heat_maps/heatmap_normalized_by_geographic_level.png')
plt.show()
