# # import pandas as pd
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.cluster import AgglomerativeClustering
# # from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
# # from sklearn.decomposition import PCA
# # import numpy as np
# #
# # # Load the data
# # file_path = 'C:\\Users\\rezaa\\OneDrive\\Desktop\\Personal\\Uni\\4022\\Machine Leaning Basics\\Project\\02\\leaves.csv'
# # data = pd.read_csv(file_path)
# #
# # # Extract ground truth labels and features
# # labels_true = data.iloc[:, 0]
# # features = data.iloc[:, 2:]
# #
# # # Preprocess the data
# # scaler = StandardScaler()
# # features_scaled = scaler.fit_transform(features)
# #
# # pca = PCA()
# # features_reduced = pca.fit_transform(features_scaled)
# #
# # # Apply Agglomerative clustering
# # agg_clustering = AgglomerativeClustering(n_clusters=36)
# # agg_labels = agg_clustering.fit_predict(features_reduced)
# #
# # # Compute silhouette score
# # silhouette = silhouette_score(features_reduced, agg_labels)
# # print(f"Silhouette Score: {silhouette:.4f}")
# #
# # # Compute adjusted Rand index (if ground truth labels are available)
# # if labels_true is not None:
# #     ARI = adjusted_rand_score(labels_true, agg_labels)
# #     print(f"Adjusted Rand Index: {ARI:.4f}")
# #
# # # Compute adjusted mutual information (if ground truth labels are available)
# # if labels_true is not None:
# #     AMI = adjusted_mutual_info_score(labels_true, agg_labels)
# #     print(f"Adjusted Mutual Information: {AMI:.4f}")
# #
# # # Define a function to compute cluster purity (if ground truth labels are available)
# # def cluster_purity(labels_true, labels_pred):
# #     cluster_purities = []
# #     for cluster in set(labels_pred):
# #         indices = labels_pred == cluster
# #         cluster_labels = labels_true[indices]
# #         if len(cluster_labels) > 0:
# #             most_frequent_label = np.argmax(np.bincount(cluster_labels))
# #             purity = np.sum(cluster_labels == most_frequent_label) / len(cluster_labels)
# #             cluster_purities.append(purity)
# #     return np.mean(cluster_purities)
# #
# # # Calculate cluster purity (if ground truth labels are available)
# # if labels_true is not None:
# #     purity = cluster_purity(labels_true, agg_labels)
# #     print(f"Cluster Purity: {purity:.4f}")
#
#
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
# from sklearn.feature_selection import SelectKBest, f_classif
# import numpy as np
#
# # Load the data
# file_path = 'C:\\Users\\rezaa\\OneDrive\\Desktop\\Personal\\Uni\\4022\\Machine Leaning Basics\\Project\\02\\leaves.csv'
# data = pd.read_csv(file_path)
#
# # Extract ground truth labels and features
# labels_true = data.iloc[:, 0]
# features = data.iloc[:, 2:]
#
# # Preprocess the data
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
#
# # Feature selection with SelectKBest
# selector = SelectKBest(score_func=f_classif, k=5)
# features_selected = selector.fit_transform(features_scaled, labels_true)
#
# # Apply Agglomerative clustering
# agg_clustering = AgglomerativeClustering(n_clusters=20, linkage='ward')
# agg_labels = agg_clustering.fit_predict(features_selected)
#
# # Compute and print evaluation metrics
# silhouette = silhouette_score(features_selected, agg_labels)
# print(f"Silhouette Score: {silhouette:.4f}")
#
# if labels_true is not None:
#     ARI = adjusted_rand_score(labels_true, agg_labels)
#     print(f"Adjusted Rand Index: {ARI:.4f}")
#
# if labels_true is not None:
#     AMI = adjusted_mutual_info_score(labels_true, agg_labels)
#     print(f"Adjusted Mutual Information: {AMI:.4f}")
#
# def cluster_purity(labels_true, labels_pred):
#     cluster_purities = []
#     for cluster in set(labels_pred):
#         indices = labels_pred == cluster
#         cluster_labels = labels_true[indices]
#         if len(cluster_labels) > 0:
#             most_frequent_label = np.argmax(np.bincount(cluster_labels))
#             purity = np.sum(cluster_labels == most_frequent_label) / len(cluster_labels)
#             cluster_purities.append(purity)
#     return np.mean(cluster_purities)
#
# if labels_true is not None:
#     purity = cluster_purity(labels_true, agg_labels)
#     print(f"Cluster Purity: {purity:.4f}")


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data
file_path = 'C:\\Users\\rezaa\\OneDrive\\Desktop\\Personal\\Uni\\4022\\Machine Leaning Basics\\Project\\02\\leaves.csv'
data = pd.read_csv(file_path)

# Extract ground truth labels and features
labels_true = data.iloc[:, 0]
features = data.iloc[:, 2:]

# Preprocess the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Feature selection with SelectKBest
selector = SelectKBest(score_func=f_classif, k=5)
features_selected = selector.fit_transform(features_scaled, labels_true)

# Apply Agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(features_selected)

# Compute and print evaluation metrics
silhouette = silhouette_score(features_selected, agg_labels)
print(f"Silhouette Score: {silhouette:.4f}")

if labels_true is not None:
    ARI = adjusted_rand_score(labels_true, agg_labels)
    print(f"Adjusted Rand Index: {ARI:.4f}")

if labels_true is not None:
    AMI = adjusted_mutual_info_score(labels_true, agg_labels)
    print(f"Adjusted Mutual Information: {AMI:.4f}")

def cluster_purity(labels_true, labels_pred):
    cluster_purities = []
    for cluster in set(labels_pred):
        indices = labels_pred == cluster
        cluster_labels = labels_true[indices]
        if len(cluster_labels) > 0:
            most_frequent_label = np.argmax(np.bincount(cluster_labels))
            purity = np.sum(cluster_labels == most_frequent_label) / len(cluster_labels)
            cluster_purities.append(purity)
    return np.mean(cluster_purities)

if labels_true is not None:
    purity = cluster_purity(labels_true, agg_labels)
    print(f"Cluster Purity: {purity:.4f}")

# Visualize dendrogram
plt.figure(figsize=(12, 6))
plt.title('Agglomerative Clustering Dendrogram')
linkage_matrix = linkage(features_selected, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=50)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
