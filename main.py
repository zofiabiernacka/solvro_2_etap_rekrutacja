import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = os.path.join("data", "cocktail_dataset.json")
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

cocktails = []
for drink in data:
    for ingredient in drink["ingredients"]:
        cocktails.append({
            "cocktail_id": drink["id"],
            "cocktail_name": drink["name"],
            "category": drink["category"],
            "glass": drink["glass"],
            "alcoholic": drink["alcoholic"],
            "ingredient": ingredient["name"],
            "ingredient_type": ingredient.get("type", "Unknown"),
            "measure": ingredient.get("measure", "Unknown")
        })

df = pd.DataFrame(cocktails)

num_cocktails = df["cocktail_name"].nunique()
num_ingredients = df["ingredient"].nunique()
print(f"Liczba unikalnych koktajli: {num_cocktails}")
print(f"Liczba unikalnych składników: {num_ingredients}")

pivot_df = df.pivot_table(index="cocktail_name", columns="ingredient", aggfunc="size", fill_value=0)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
pivot_df["cluster"] = kmeans.fit_predict(pivot_df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(pivot_df.drop(columns=["cluster"]))

df_clusters = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
df_clusters["cluster"] = pivot_df["cluster"].values
df_clusters["cocktail_name"] = pivot_df.index

df = df.merge(df_clusters[["cocktail_name", "cluster"]], on="cocktail_name")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clusters, x="PC1", y="PC2", hue="cluster", palette="tab10", alpha=0.8)
plt.title("Klasteryzacja koktajli")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.savefig(os.path.join(OUTPUT_DIR, "clusters.png"))
plt.show()

with open(os.path.join(OUTPUT_DIR, "clusters_summary.txt"), "w") as f:
    for cluster in range(5):
        f.write(f"\no Klaster {cluster}:\n")
        f.write("Najczęściej używane składniki:\n")
        f.write(str(df[df["cluster"] == cluster]["ingredient"].value_counts().head(5)) + "\n")
        f.write("\nRodzaj szkła:\n")
        f.write(str(df[df["cluster"] == cluster]["glass"].value_counts().head(3)) + "\n")
        f.write("\nKategorie koktajli:\n")
        f.write(str(df[df["cluster"] == cluster]["category"].value_counts().head(3)) + "\n")
        f.write("\nPodział alkoholowy:\n")
        f.write(str(df[df["cluster"] == cluster]["alcoholic"].value_counts(normalize=True).map(lambda x: f"{x:.0%}")) + "\n")

print("Wyniki zapisano w katalogu 'output'.")
