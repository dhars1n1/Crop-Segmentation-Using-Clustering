# -*- coding: utf-8 -*-
"""B9-FAI.ipynb
Original file is located at
    https://colab.research.google.com/drive/1HV7g5NrBcCgbBcu3AaxdIk7DfY3l2dMQ

Dataset details (taken from Kaggle): [Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)

# Import necessary libraries
"""

import numpy as np  # For numerical computations
import pandas as pd  # For handling and manipulating data
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced and attractive data visualization
from sklearn.preprocessing import LabelEncoder  # For encoding categorical data into numerical format
from sklearn.preprocessing import StandardScaler  # For normalizing data to a standard scale

from sklearn.cluster import KMeans  # For K-Means clustering
from mpl_toolkits.mplot3d import Axes3D  # For creating 3D plots

from sklearn.metrics import silhouette_score  # For evaluating clustering performance using silhouette scores
from sklearn.metrics import davies_bouldin_score  # For evaluating clustering performance using Davies-Bouldin index
from sklearn.metrics import calinski_harabasz_score  # For evaluating clustering performance using Calinski-Harabasz index
from sklearn.decomposition import PCA  # For dimensionality reduction using Principal Component Analysis
from sklearn.cluster import DBSCAN  # For clustering using Density-Based Spatial Clustering of Applications with Noise
from sklearn.cluster import AgglomerativeClustering  # For hierarchical clustering
from sklearn.mixture import GaussianMixture  # For clustering using Gaussian Mixture Models

"""# Exploratory Data Analysis

**Data loading and initial exploration** - We begin by loading the dataset, checking its structure, and handling missing values.
"""

# Load dataset
df_cust = pd.read_csv('marketing_campaign.csv', sep=";")

# Display dataset structure and summary statistics
print("Dataset Information:")
df_cust.info()


print("\nDataset Summary Statistics:")
print(df_cust.describe())

print("Column names in the dataset:")
print(df_cust.columns)
print(df_cust.head())

"""Define a function to remove outliers"""

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

# Fill missing income values with the median
median_income = df_cust['Income'].median()
df_cust.loc[:, 'Income'] = df_cust['Income'].fillna(median_income)
print(f"\nFilled missing Income values with the median: {median_income}")

df_cust = remove_outliers(df_cust, 'Income')

"""**Age Distribution** -
We calculate the age of customers and visualize its distribution to understand the age range.
"""

# Calculate age from birth year
df_cust['Age'] = 2024 - df_cust['Year_Birth']

df_cust = remove_outliers(df_cust, 'Age')

# Visualize age distribution
print("\nVisualizing Age Distribution:")
plt.figure(figsize=(10, 6))

n, bins, patches = plt.hist(df_cust['Age'], bins=20, color='blue', alpha=0.7, edgecolor='black')
for i in range(len(patches)):
    patches[i].set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(n[i] / max(n)))
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], int(n[i]),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Age distribution of customers', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""**Filter Unusual Values** -
We remove extreme income and age values to focus on typical customers.

**Income Distribution Analysis** -
We analyze and visualize the income distribution to identify customer income levels.
"""

# Filter customers with income <= 300,000 and age <= 90
df_cust = df_cust[(df_cust['Income'] <= 300000) & (df_cust['Age'] <= 90)]
print("\nFiltered customers with income <= 300,000 and age <= 90.")

# Calculate percentage of high-income customers
total_customers = len(df_cust['Income'].dropna())
high_income_customers = len(df_cust[df_cust['Income'] > 100000])
percentage_high_income = (high_income_customers / total_customers) * 100
print(f"Percentage of customers with income > 100,000: {percentage_high_income:.2f}%")

# Filter data for visualization
df_cust_filtered = df_cust[df_cust['Income'] <= 100000]

# Visualize income distribution
print("\nVisualizing Income Distribution (Income <= 100,000):")
plt.figure(figsize=(10, 6))

n, bins, patches = plt.hist(df_cust_filtered['Income'].dropna(), bins=20, color='blue', alpha=0.7, edgecolor='black')
for i in range(len(patches)):
    patches[i].set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(n[i] / max(n)))
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], int(n[i]),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Income distribution of customers (income <= 100,000)', fontsize=16)
plt.xlabel('Income', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print inference
print("Inference: The majority of customers earn less than 100,000.")

"""**Spending Distribution Across Product Categories** -
We analyze how customers spend across various product categories.
"""

# Spending distribution across product categories
product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
total_spend = df_cust[product_columns].sum()

print("\nVisualizing Spending Distribution Across Product Categories:")
colors = sns.color_palette("rocket_r", len(product_columns))
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(total_spend, labels=product_columns, autopct='%1.1f%%',
                                   startangle=90, colors=colors, wedgeprops=dict(width=0.4), pctdistance=0.75)
for autotext in autotexts:
    autotext.set_color('white')  # white for better contrast
    autotext.set_fontsize(14)
    autotext.set_weight('bold')

plt.title('Distribution of spending across product categories', fontsize=16)
plt.show()

"""**Spending by Marital Status** -
We explore total spending across different marital statuses.
"""

# Add total spending column
df_cust.loc[:, 'Total_Spend'] = df_cust[product_columns].sum(axis=1)

# Group spending by marital status
spending_by_marital_status = df_cust.groupby('Marital_Status')['Total_Spend'].sum()

print("\nVisualizing Spending by Marital Status:")
plt.figure(figsize=(10, 6))
bars = plt.bar(spending_by_marital_status.index, spending_by_marital_status,
               color=sns.color_palette("rocket_r", len(spending_by_marital_status)))

for bar, spend in zip(bars, spending_by_marital_status):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(spend)}', ha='center', va='bottom', fontsize=12)

plt.title('Total spending by marital status', fontsize=16)
plt.xlabel('Marital status', fontsize=14)
plt.ylabel('Total spending', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""# Data preprocessing

Label Encoding Categorical Features - Converts categorical variables (Education and Marital_Status) into numerical values, as machine learning models generally work better with numerical data.
"""

# Label encoding for categorical columns
le = LabelEncoder()
df_cust.loc[:, 'Education'] = le.fit_transform(df_cust['Education'])
df_cust.loc[:, 'Marital_Status'] = le.fit_transform(df_cust['Marital_Status'])

"""Correlation Matrix for Numerical Features - To identify relationships between
numerical features. High correlations can indicate redundancy, while no correlation may suggest independent features.
"""

# Define numerical columns for correlation analysis
numerical_cols = [
    'Age', 'Education', 'Marital_Status', 'Income', 'MntWines',
    'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

# Compute and visualize the correlation matrix
corr_matrix = df_cust[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='rocket_r', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

"""Feature Engineering

Create Family Size - Kidhome and Teenhome represent the number of children in the household. Adding them together gives the total family size, which could influence spending behavior.
"""

df_cust.loc[:, 'Family_Size'] = df_cust['Kidhome'] + df_cust['Teenhome']

""" Average Spend Per Purchase - Helps analyze customer spending efficiency by calculating the average amount spent per purchase. Useful for segmenting high-value vs. low-value customers."""

df_cust.loc[:, 'Avg_Spend_Per_Purchase'] = df_cust['Total_Spend'] / (
    df_cust['NumDealsPurchases'] + df_cust['NumWebPurchases'] +
    df_cust['NumCatalogPurchases'] + df_cust['NumStorePurchases']
)

""" Spending Ratios - Ratios help understand how much a customer spends on specific product categories (e.g., wines, meats, fruits) compared to their total spending. Useful for targeted marketing."""

df_cust.loc[:, 'Wine_Ratio'] = df_cust['MntWines'] / df_cust['Total_Spend']
df_cust.loc[:, 'Meat_Ratio'] = df_cust['MntMeatProducts'] / df_cust['Total_Spend']
df_cust.loc[:, 'Fruit_Ratio'] = df_cust['MntFruits'] / df_cust['Total_Spend']

"""Total Accepted Campaigns - Aggregates the number of campaigns accepted by the customer. Helps identify how receptive a customer is to promotional efforts."""

df_cust.loc[:, 'Total_Accepted_Campaigns'] = (
    df_cust['AcceptedCmp1'] + df_cust['AcceptedCmp2'] +
    df_cust['AcceptedCmp3'] + df_cust['AcceptedCmp4'] + df_cust['AcceptedCmp5']
)

"""---

Pairplot for Insights
"""

# Visualize relationships between key numerical features
plt.figure(figsize=(8, 6))
sns.pairplot(df_cust[['Income', 'Total_Spend', 'Family_Size', 'Age']],
             plot_kws={'color': '#781C68'},
             diag_kws={'color': '#781C68', 'fill': True})
plt.show()

"""# Clustering"""

# --- Clustering Visualization ---
# Initial scatter plot of Income vs. Total Spend to observe potential clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Income', y='Total_Spend', data=df_cust, s=100, color='#781C68')
plt.title('Initial Observation: Income vs. Total Spending')
plt.show()

# --- Feature Selection for Clustering ---
# Selecting features relevant for clustering
features = ['Income', 'Total_Spend', 'Age', 'Family_Size']

# Scaling the features for better clustering performance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cust[features])

# Converting the scaled data back to a DataFrame for convenience (optional)
scaled_df = pd.DataFrame(scaled_features, columns=features)

# --- Elbow Method to Determine Optimal Clusters ---
# Calculating Sum of Squared Errors (SSE) for a range of cluster numbers
sse = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o', color='#781C68')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.show()

"""K-means clustering and GMM"""

# --- Data Preparation ---
features = ["Income", "Total_Spend", "Age", "Family_Size"]

# Scale data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cust[features])

# PCA Transformation
pca = PCA(n_components=3)  # 3 components for 3D visualization
pca_features = pca.fit_transform(scaled_features)

# --- Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
gmm = GaussianMixture(n_components=3, random_state=42)

# Clustering predictions
clusters_kmeans = kmeans.fit_predict(scaled_features)
clusters_kmeans_pca = kmeans.fit_predict(pca_features)
clusters_gmm = gmm.fit_predict(scaled_features)
clusters_gmm_pca = gmm.fit_predict(pca_features)

# Add cluster labels to the dataframe
df_cust["Cluster_kmeans"] = clusters_kmeans
df_cust["Cluster_KMeans_PCA"] = clusters_kmeans_pca
df_cust["Cluster_gmm"] = clusters_gmm
df_cust["Cluster_GMM_PCA"] = clusters_gmm_pca

# --- Plotting ---
fig = plt.figure(figsize=(20, 10))

# Row 1: 2D scatter plots
# Subplot 1: KMeans without PCA (2D)
ax1 = fig.add_subplot(241)
sns.scatterplot(x=df_cust["Income"], y=df_cust["Total_Spend"], hue=clusters_kmeans, palette="rocket", s=100, ax=ax1)
ax1.set_title("KMeans without PCA (2D)")
ax1.set_xlabel("Income")
ax1.set_ylabel("Total Spend")

# Subplot 2: KMeans with PCA (2D)
ax2 = fig.add_subplot(242)
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters_kmeans_pca, palette="rocket", s=100, ax=ax2)
ax2.set_title("KMeans with PCA (2D)")
ax2.set_xlabel("PCA Feature 1")
ax2.set_ylabel("PCA Feature 2")

# Subplot 3: GMM without PCA (2D)
ax3 = fig.add_subplot(243)
sns.scatterplot(x=df_cust["Income"], y=df_cust["Total_Spend"], hue=clusters_gmm, palette="rocket", s=100, ax=ax3)
ax3.set_title("GMM without PCA (2D)")
ax3.set_xlabel("Income")
ax3.set_ylabel("Total Spend")

# Subplot 4: GMM with PCA (2D)
ax4 = fig.add_subplot(244)
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters_gmm_pca, palette="rocket", s=100, ax=ax4)
ax4.set_title("GMM with PCA (2D)")
ax4.set_xlabel("PCA Feature 1")
ax4.set_ylabel("PCA Feature 2")

# Row 2: 3D scatter plots
# Subplot 5: KMeans without PCA (3D)
ax5 = fig.add_subplot(245, projection="3d")
scatter1 = ax5.scatter(df_cust["Income"], df_cust["Total_Spend"], df_cust["Age"], c=clusters_kmeans, cmap="rocket", s=100)
ax5.set_title("KMeans without PCA (3D)")
ax5.set_xlabel("Income")
ax5.set_ylabel("Total Spend")
ax5.set_zlabel("Age")

# Subplot 6: KMeans with PCA (3D)
ax6 = fig.add_subplot(246, projection="3d")
scatter2 = ax6.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=clusters_kmeans_pca, cmap="rocket", s=100)
ax6.set_title("KMeans with PCA (3D)")
ax6.set_xlabel("PCA Feature 1")
ax6.set_ylabel("PCA Feature 2")
ax6.set_zlabel("PCA Feature 3")

# Subplot 7: GMM without PCA (3D)
ax7 = fig.add_subplot(247, projection="3d")
scatter3 = ax7.scatter(df_cust["Income"], df_cust["Total_Spend"], df_cust["Age"], c=clusters_gmm, cmap="rocket", s=100)
ax7.set_title("GMM without PCA (3D)")
ax7.set_xlabel("Income")
ax7.set_ylabel("Total Spend")
ax7.set_zlabel("Age")

# Subplot 8: GMM with PCA (3D)
ax8 = fig.add_subplot(248, projection="3d")
scatter4 = ax8.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=clusters_gmm_pca, cmap="rocket", s=100)
ax8.set_title("GMM with PCA (3D)")
ax8.set_xlabel("PCA Feature 1")
ax8.set_ylabel("PCA Feature 2")
ax8.set_zlabel("PCA Feature 3")

# Adjust layout and display
plt.tight_layout()
plt.show()

"""---

DBSCAN clustering
"""

# Scaling the data
features = ['Income', 'Total_Spend', 'Age', 'Family_Size']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cust[features])

# Convert scaled data back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features)

dbscan = DBSCAN(eps=0.5, min_samples=5)
df_cust['Cluster_DBSCAN'] = dbscan.fit_predict(scaled_df)

# Metrics
if len(set(df_cust['Cluster_DBSCAN'])) > 1:
    silhouette_db = silhouette_score(scaled_df, df_cust['Cluster_DBSCAN'])
    db_score_db = davies_bouldin_score(scaled_df, df_cust['Cluster_DBSCAN'])
    ch_score_db = calinski_harabasz_score(scaled_df, df_cust['Cluster_DBSCAN'])
    print(f"Silhouette Score (DBSCAN): {silhouette_db}")
    print(f"Davies-Bouldin Index (DBSCAN): {db_score_db}")
    print(f"Calinski-Harabasz Score (DBSCAN): {ch_score_db}")

# Visualize Clusters (Income vs. Total Spending)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Income', y='Total_Spend', hue='Cluster_DBSCAN', data=df_cust, palette='rocket', s=100)
plt.title('DBSCAN Clusters (Income vs. Total Spending)')
plt.show()

"""Agglomerative Clustering"""

# Extract Income and Total Spending features
features = ['Income', 'Total_Spend']
data = df_cust[features]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)  # Set n_clusters based on your data exploration
df_cust['Cluster_Agglo'] = agglo.fit_predict(scaled_data)

# Metrics for Agglomerative Clustering
silhouette_agglo = silhouette_score(scaled_data, df_cust['Cluster_Agglo'])
db_score_agglo = davies_bouldin_score(scaled_data, df_cust['Cluster_Agglo'])
ch_score_agglo = calinski_harabasz_score(scaled_data, df_cust['Cluster_Agglo'])

# Visualization
plt.figure(figsize=(14, 6))


# Agglomerative Clustering
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='Income', y='Total_Spend', hue='Cluster_Agglo',
    data=df_cust, palette='rocket', s=100
)
plt.title('Agglomerative Clustering (Income vs. Total Spending)')
plt.xlabel('Income')
plt.ylabel('Total Spending')

plt.tight_layout()
plt.show()

"""---

# Interpretation

Cluster summary
"""

print(df_cust.columns)

# Compute key statistics for each cluster
cluster_summary = df_cust.groupby('Cluster_KMeans_PCA').agg({
    'Income': ['mean', 'median'],
    'Total_Spend': ['mean', 'median'],
    'MntWines': 'mean',
    'MntMeatProducts': 'mean',
    'Marital_Status': lambda x: x.mode()[0],  # Most common marital status
    'Age': ['mean', 'median']
})

# Clean up column names
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.reset_index()

# Display the summary
print(cluster_summary)

"""PCA
1 - high spenders

Boxplots for Cluster Visualization
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster_KMeans_PCA', y='Income', data=df_cust, hue='Cluster_KMeans_PCA', palette='rocket', dodge=False)
plt.legend([], [], frameon=False)  # Optional: Remove unnecessary legends
plt.title('Income Distribution Across Clusters')
plt.show()

"""Marketing Recommendations"""

# Calculate total campaigns accepted
df_cust.loc[:, 'Total_Campaigns_Accepted'] = (
    df_cust['AcceptedCmp1'] + df_cust['AcceptedCmp2'] +
    df_cust['AcceptedCmp3'] + df_cust['AcceptedCmp4'] +
    df_cust['AcceptedCmp5']
)

# Plot: Total Campaigns Accepted by Clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Total_Campaigns_Accepted', hue='Cluster_KMeans_PCA', data=df_cust, palette='rocket')
plt.title('Total Campaigns Accepted by Clusters')
plt.show()

"""Clustering evaluation"""

# Function to evaluate clustering methods
def evaluate_clustering_methods(X, labels):
    if len(set(labels)) > 1:  # Ensure there are multiple clusters
        print("Calinski-Harabasz Score:", calinski_harabasz_score(X, labels))
        print("Davies-Bouldin Score:", davies_bouldin_score(X, labels))
        print("Silhouette Score:", silhouette_score(X, labels))
    else:
        print("Only one cluster found; no valid scores available.")
    print("-" * 50)

# Evaluate all clustering methods
print("KMeans without PCA Clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_kmeans'])

print("KMeans with PCA Clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_KMeans_PCA'])

print("GMM without PCA Clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_gmm'])

print("GMM with PCA Clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_GMM_PCA'])


print("Agglomerative Clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_Agglo'])

print("DBSCAN clustering Scores:")
evaluate_clustering_methods(scaled_df, df_cust['Cluster_DBSCAN'])