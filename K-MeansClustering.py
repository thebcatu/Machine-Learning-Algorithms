from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

class KMeansClustering:
    def __init__(self, n_clusters=3):
        self.data = load_iris()
        self.X = self.data.data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def train(self):
        self.model.fit(self.X_scaled)
        self.labels = self.model.labels_
        self.centroids = self.model.cluster_centers_

    def predict(self, sample):
        sample_scaled = self.scaler.transform([sample])
        return self.model.predict(sample_scaled)

    def get_cluster_info(self):
        return self.labels, self.centroids

clustering_model = KMeansClustering(n_clusters=3)
clustering_model.train()
labels, centroids = clustering_model.get_cluster_info()
print("Cluster Labels:", labels)
print("Centroids:\n", centroids)

sample_data = clustering_model.X[0]
predicted_cluster = clustering_model.predict(sample_data)
print(f"Predicted cluster for sample data: {predicted_cluster[0]}")
