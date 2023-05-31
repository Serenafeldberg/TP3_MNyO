
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def data ():
    data = []
    with open('dataset_clusters.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            values = []
            for val in row:
                values.append(float(val))

            data.append(np.array(values))

    data = np.array(data)
    return data

def dec_svd (data, d):
    U, s, Vt = np.linalg.svd(data)

    U = U[:, :d]
    s = np.diag(s[:d])
    Vt = Vt[:d, :]

    reconstructed_data = U @ s @ Vt

    return reconstructed_data

def Kmeans (data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_

    return labels, centroids

def dbscan (data):

    debscan = DBSCAN ()

    cluster = debscan.fit(data)
    labels = cluster.labels_

    return labels

def plot_clusters (data, label, title, bool, cen):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    if bool:
        plt.scatter(cen[0], cen[1], c='red', marker='x')
    plt.show()

def centroid (data, labels):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:  # Ignorar puntos de ruido
            continue
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    return np.array(centroids)

def euclidean_distance(x1, x2, sigma=0.5):
    return np.linalg.norm(x1 - x2)
    return np.exp(-((np.linalg.norm(x1 - x2))**2)/(2 * (sigma**2)) )

def get_neighbors(data, point_idx, eps):
    neighbors = []
    for idx, point in enumerate(data):
        if euclidean_distance(data[point_idx], point) <= eps:
            neighbors.append(idx)
    return neighbors

def clusttering(data, eps, min_samples):
    labels = np.zeros(len(data))
    cluster_label = 0

    for idx, point in enumerate(data):
        if labels[idx] != 0:
            continue
        
        neighbors = get_neighbors(data, idx, eps)
        
        if len(neighbors) < min_samples:
            labels[idx] = -1  # Etiqueta como ruido
        else:
            cluster_label += 1
            labels[idx] = cluster_label
            
            # Expandir el cluster
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_label
                elif labels[neighbor_idx] == 0:
                    labels[neighbor_idx] = cluster_label
                    neighbor_neighbors = get_neighbors(data, neighbor_idx, eps)
                    if len(neighbor_neighbors) >= min_samples:
                        neighbors.extend(neighbor_neighbors)
    
    return labels

def find_centroids(data, eps, min_samples):
    centroids = []
    visited = np.zeros(len(data), dtype=bool)
    
    for idx, point in enumerate(data):
        if visited[idx]:
            continue
        
        neighbors = []
        for neighbor_idx, neighbor in enumerate(data):
            if not visited[neighbor_idx] and euclidean_distance(point, neighbor) <= eps:
                neighbors.append(neighbor_idx)
        
        if len(neighbors) >= min_samples:
            centroid = np.mean(data[neighbors], axis=0)
            centroids.append(centroid)
            visited[neighbors] = True
    
    return centroids


if __name__ == "__main__":
    data = data()
    reconstructed_data = dec_svd(data, 2)

    # label = dbscan(reconstructed_data)
    # centroids = centroid(reconstructed_data, label)
    # plot_clusters(reconstructed_data, label, 'DBSCAN', True, centroids)

    # k = len(set(label)) - (1 if -1 in label else 0)
    # label2, cen = Kmeans(reconstructed_data, 2)

    # plot_clusters(reconstructed_data, label2, 'K means', True, cen)

    label = clusttering(reconstructed_data, 0.5, 5)

    cen = find_centroids(reconstructed_data, 0.5, 5)

    plot_clusters(reconstructed_data, label, 'DBSCAN', True, cen)




