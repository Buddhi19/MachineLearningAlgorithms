# %%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

import seaborn as sns


from word2number import w2n


import matplotlib.pyplot as plt
from scipy import stats

# %%
dataset = pd.read_csv('customer_dataset.csv')
dataset.head()

# %%
#print all non numeric columns
for i in dataset.columns:
    if dataset[i].dtype == 'object':
        print(i)

# %%
#handling non-numeric data converting them to numeric
def word_to_num(value):
    try:
        return w2n.word_to_num(value)
    except ValueError:
        if value.isdigit():
            return int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
            return float(value)
        return None
    
for column in dataset.columns:
    dataset[column] = dataset[column].apply(lambda x : word_to_num(str(x)))

# %%
#nan count
print(dataset.isna().sum())
#length of dataset
print(f"Length of dataset: {len(dataset)}")
#count all the rows with nan values
rows_with_nan = dataset.isna().any(axis=1).sum()
print(f"Rows with nan values: {rows_with_nan}")
#ratio of rows with nan values
missing_percentage = dataset.isnull().mean() * 100
print(f"Ratio of rows with nan values: {round(rows_with_nan/len(dataset),2)*100}%")

plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1)
plt.title('Missing Values')
sns.heatmap(dataset.isnull(), cbar=False)

plt.subplot(1, 2, 2)
missing_percentage.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.tight_layout()
plt.show()

# %%
#correlation matrix to see the missing values are correlated
plt.figure(figsize=(5, 5))
corr_matrix = dataset.isnull().corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Missing Values')
plt.show()

# %%
#extract nan rows from dataset
nan_rows = dataset[dataset.isnull().any(axis=1)]
#replace nan with 0
dataset.dropna(inplace=True)
# Handle duplicates
dataset.drop_duplicates(inplace=True)

#replace nan with 0
nan_rows.fillna(0, inplace=True)
nan_rows

# %%
# iso_forest = IsolationForest(contamination=0.05, random_state=42)
# features = ['average_monthly_spend','average_monthly_visit_frequency','average_monthly_basket_size']
# data = dataset.copy()
# # Fit the model
# iso_forest.fit(data[features])

# # Predict outliers
# outliers = iso_forest.predict(data[features])

# # Add the outlier predictions to the original data
# data['outlier'] = outliers

# # Display the rows identified as outliers
# outliers_data = data[data['outlier'] == -1]

# sns.pairplot(data, hue='outlier', palette={1: 'blue', -1: 'red'}, markers=["o", "s"])
# plt.suptitle('Isolation Forest Outlier Detection', y=1.02)
# plt.show()

# %%
data = dataset.copy()
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1

# Consider a data point an outlier if it is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR
outliers = ((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)
dataset = dataset[~outliers]

data.describe()

# %%
#scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.iloc[:, 1:])
scaled_data_nan = scaler.transform(nan_rows.iloc[:, 1:])

# %%
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)


# %%
dataset['Cluster'] = clusters
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(dataset['average_monthly_spend'], dataset['average_monthly_visit_frequency'], dataset['average_monthly_basket_size'], c=clusters, cmap='viridis')
ax.set_xlabel('average_monthly_spend')
ax.set_ylabel('average_monthly_visit_frequency')
ax.set_zlabel('average_monthly_basket_size')
plt.title('Clusters')
ax.scatter(nan_rows['average_monthly_spend'], nan_rows['average_monthly_visit_frequency'], nan_rows['average_monthly_basket_size'],
           c='black',s = 0.1)
ax.set_xlabel('average_monthly_spend')
ax.set_ylabel('average_monthly_visit_frequency')
ax.set_zlabel('average_monthly_basket_size')


plt.show()


