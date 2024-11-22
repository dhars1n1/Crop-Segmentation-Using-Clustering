# Customer Segmentation using Clustering

This project involves customer segmentation based on a marketing dataset using various clustering techniques. The main goal is to analyze customer behavior, spending patterns, and demographic data to group customers into meaningful clusters for targeted marketing campaigns.

## Dataset Details

- **Source**: [Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)
- **Description**: The dataset includes information on customer demographics, spending habits, and responsiveness to campaigns.

---

## Steps Involved

### 1. Exploratory Data Analysis (EDA)
- **Data Loading and Initial Exploration**:
  - Loaded the dataset and checked its structure, including data types, missing values, and basic statistics.
  - Visualized the distribution of key features.

- **Outlier Removal**:
  - Used the IQR method to remove outliers in `Income` and `Age`.

- **Feature Engineering**:
  - Created new features such as `Family_Size`, `Avg_Spend_Per_Purchase`, and spending ratios for product categories.

- **Visualization**:
  - Distribution of income, age, and spending across product categories and marital statuses.

---

### 2. Data Preprocessing
- **Handling Missing Values**:
  - Filled missing values in the `Income` column with the median.

- **Label Encoding**:
  - Encoded categorical variables (`Education` and `Marital_Status`) to numerical values.

- **Correlation Analysis**:
  - Analyzed correlations between numerical features using a heatmap.

---

### 3. Clustering
- **Feature Selection**:
  - Selected features (`Income`, `Total_Spend`, `Age`, `Family_Size`) for clustering.

- **Scaling**:
  - Standardized the selected features for better clustering performance.

- **PCA Transformation**:
  - Reduced dimensions for better visualization in 3D space.

- **Clustering Techniques**:
  - Implemented **K-Means**, **Gaussian Mixture Models (GMM)**, **DBSCAN**, and **Hierarchical Clustering**.
  - Determined the optimal number of clusters using the Elbow Method, Silhouette Scores, and Davies-Bouldin Index.

---

### 4. Visualization
- **Cluster Analysis**:
  - Plotted clusters in 2D and 3D to visualize customer segments.
  - Compared results between different clustering techniques with and without PCA.

---

## Key Findings
- Most customers fall into low to medium income groups.
- High-income customers account for a small percentage, making them potential targets for premium products.
- Spending behavior varies significantly based on family size, marital status, and age.

---

## Tools and Libraries Used
- **Python Libraries**:
  - `numpy`, `pandas` for data handling and manipulation.
  - `matplotlib`, `seaborn` for visualization.
  - `sklearn` for preprocessing, clustering, and evaluation.
  - `PCA` for dimensionality reduction.

---

## Results
- Identified customer segments for targeted marketing.
- Implemented a robust preprocessing pipeline to handle missing data and outliers.
- Compared different clustering methods to find the best fit for this dataset.

---

## Future Work
- Implementing predictive modeling for campaign success based on identified clusters.
- Exploring deep learning techniques for advanced segmentation.
- Extending analysis to include time-series customer behavior data.
