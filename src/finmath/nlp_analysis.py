import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('src/finmath/Jan2May2025.csv', sep=';', decimal=',')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['booking_text'] = df['booking_text'].astype(str)

# Optional: filter unknown/miscategorized
df_unknown = df[df['category'].isin(['Unknown', '', None]) | df['category'].isnull()].copy()
if df_unknown.empty:
    df_unknown = df.copy()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(df_unknown['booking_text'])

# Clustering with KMeans
n_clusters = 11  # Tune based on dataset
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_unknown['cluster'] = kmeans.fit_predict(X)

# Show top terms per cluster
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
print("\n### Top terms per cluster:")
for i in range(n_clusters):
    top_words = [terms[ind] for ind in order_centroids[i, :5]]
    print(f"Cluster {i}: {', '.join(top_words)}")

# Show sample booking_text per cluster
print("\n### Sample booking_text per cluster:")
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print(df_unknown[df_unknown['cluster'] == i]['booking_text'].head(3).to_string(index=False))

# Optional: Suggest category labels manually
# e.g., map clusters to user-defined categories after inspection
cluster_to_category = {
    0: 'Living Expenses',
    1: 'Banking & Loans',
    2: 'Home',
    3: 'Online Shopping & Retail',
    4: 'Savings',
    5: 'Restaurants & Bars',
    6: 'Mobility & Car',
    7: 'Insurances',
    8: 'Maintenance',
    9: 'Leisure',
    10: 'Forex'
}
df_unknown['suggested_category'] = df_unknown['cluster'].map(cluster_to_category)

# Merge back with original
df_final = df.merge(df_unknown[['booking_text', 'suggested_category']], on='booking_text', how='left')
df_final['final_category'] = df_final['category'].combine_first(df_final['suggested_category'])

# Save result
# df_final.to_csv('categorized_transactions.csv', index=False)


import matplotlib.pyplot as plt

# Sum amounts per cluster
cluster_sums = df_unknown.groupby('cluster')['amount'].sum().abs()  # Use abs() for meaningful pie

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(cluster_sums, labels=[f'Cluster {i}' for i in cluster_sums.index],
        autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
plt.title('Share of Total Transaction Amount by Cluster')
plt.tight_layout()
plt.show()
                