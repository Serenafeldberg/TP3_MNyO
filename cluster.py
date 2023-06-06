
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

<<<<<<< Updated upstream
    cluster = debscan.fit(data)
    labels = cluster.labels_

    return labels
=======

def distance(x1, x2):
    return np.linalg.norm(x1-x2)


def euclidean_matrix(data):
    matrix = np.zeros((1755, 1755))

    for i in range(1755):
        point_1 = data[i]
        for j in range(1755):
            point_2  = data[j]
            matrix[i][j] = euclidean_distance(point_1, point_2)
    
    return matrix


def get_neighbors(data, centroid, eps):

    neighbours = []

    for idx, point in enumerate(data):

        if distance(centroid, point) <= eps:
            neighbours.append(idx)

    return neighbours


def find_cluster(neighbours1, neighbours2, data):



    for idx, point in enumerate(data):

        minimum_distance_1 = distance(data[idx], data[neighbours1[0]])
        minimum_distance_2 = distance(data[idx], data[neighbours2[0]])

        for neighbours_index in neighbours1:
            compare = distance(data[idx], data[neighbours_index])
            if (compare < minimum_distance_1):
                minimum_distance_1 = compare

    
        for neighbours_index in neighbours2:
            compare = distance(data[idx], data[neighbours_index])
            if (compare < minimum_distance_2):
                minimum_distance_2 = compare
            


        if (idx not in neighbours1 and idx not in neighbours2):
            if minimum_distance_2 > minimum_distance_1:
                neighbours1.append(idx)
            else:
                neighbours2.append(idx)


def get_label(neighbours1, neighbours2, data, bool, cen):

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for idx in neighbours1:
        x1.append(data[idx][0])
        y1.append(data[idx][1])

    for idx in neighbours2:
        x2.append(data[idx][0])
        y2.append(data[idx][1])

    
    plt.figure()
    plt.scatter(x1, y1, color='red', label = "Primer cluster")
    plt.scatter(x2, y2, color='blue', label = "Segundo cluster")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Centroides y Clusters")

    if bool:
        plt.scatter(cen[:, 0], cen[:, 1], c='yellow', marker='x', label = "Centroides")

    print("longitud primer cluster:", len(x1))
    print("longitud segundo cluster:", len(x2))
    plt.legend()
    plt.show()


def check_repeated(neighbours1, neighbours2, centroid1, centroid2):

    for elem in neighbours1:

        if elem in neighbours2:

            if (distance(elem, centroid1) < distance(elem, centroid2)):
                neighbours2.remove(elem)
            else:
                neighbours1.remove(elem)


def count_label(label):

    count1 = 0
    count2 = 0

    for elem in label:
        if elem == 0:
            count1 += 1
        else:
            count2 +=1
    
    print("cluster1:", count1)
    print("cluster2:",count2)
>>>>>>> Stashed changes


def plot_clusters (data, label, title, bool, cen):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    if bool:
        plt.scatter(cen[0], cen[1], c='red', marker='x')
    plt.show()

<<<<<<< Updated upstream
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
=======

def find_better_centroid(data, neighbours1, neighbours2, centroid1, centroid2):

    min_distance_1 = find_distance(centroid1, data, neighbours1)
    min_distance_2 = find_distance(centroid2, data, neighbours2)

    for idx in neighbours1:
        new_min_distance_1 = find_distance(data[idx], data, neighbours1)

        if new_min_distance_1 < min_distance_1:
            min_distance_1 = new_min_distance_1
            centroid1 = data[idx]


    for idx in neighbours2:
        new_min_distance_2 = find_distance(data[idx], data, neighbours2)

        if new_min_distance_2 < min_distance_2:
            min_distance_2 = new_min_distance_2
            centroid2 = data[idx]

    
    return centroid1, centroid2


def find_distance(centroid, data, neighbours):

    centroid_distance = 0

    for idx in neighbours:
        centroid_distance += distance(centroid, data[idx])
    
    centroid_distance = centroid_distance/len(neighbours)

    return centroid_distance

        
def iterate(data, neighbours1, neighbours2, centroid1, centroid2, n):
    
    new_centroid_1, new_centroid_2 = find_better_centroid(data, neighbours1, neighbours2, centroid1, centroid2)
    
    neighbours1_1 = get_neighbors(data, new_centroid_1, n)
    neighbours2_1 = get_neighbors(data, new_centroid_2, n)
    check_neigbours(reconstructed_data, new_centroid_1, new_centroid_2, n, neighbours1_1, neighbours2_1)
    check_repeated(neighbours1_1, neighbours2_1, new_centroid_1, new_centroid_2)

    find_cluster(neighbours1_1, neighbours2_1, data)

    while np.array_equal(centroid1, new_centroid_1) == False or np.array_equal(centroid2, new_centroid_2) == False:

        n = 2
        new_centroid_1, new_centroid_2 = find_better_centroid(data, neighbours1_1, neighbours2_1, centroid1, centroid2)
        neighbours1_1 = get_neighbors(data, new_centroid_1, n)
        neighbours2_1 = get_neighbors(data, new_centroid_2, n)
        check_neigbours(reconstructed_data, new_centroid_1, new_centroid_2, n, neighbours1_1, neighbours2_1)
        check_repeated(neighbours1_1, neighbours2_1, new_centroid_1, new_centroid_2)
        find_cluster(neighbours1_1, neighbours2_1, data)
        centroid1, centroid2 = find_better_centroid(data, neighbours1_1, neighbours2_1, new_centroid_1, new_centroid_2)
        print("iteracion")

    print(len(neighbours1_1))
    print(len(neighbours2_1))

    return np.array([centroid1, centroid2]), neighbours1_1, neighbours2_1


def check_neigbours(data, centroid1, centroid2, n, neighbours1, neighbours2):

    while len(neighbours1) == 0 or len(neighbours2) == 0:
        n += 1
        neighbours1 = get_neighbors(data, centroid1, n)
        neighbours2 = get_neighbors(data, centroid2, n)
>>>>>>> Stashed changes


if __name__ == "__main__":
    data = data()
    reconstructed_data = dec_svd(data, 2)

<<<<<<< Updated upstream
    # label = dbscan(reconstructed_data)
    # centroids = centroid(reconstructed_data, label)
    # plot_clusters(reconstructed_data, label, 'DBSCAN', True, centroids)

    # k = len(set(label)) - (1 if -1 in label else 0)
    # label2, cen = Kmeans(reconstructed_data, 2)

    # plot_clusters(reconstructed_data, label2, 'K means', True, cen)
=======
    n = 2
    data_ = data()
    reconstructed_data = dec_svd(data_, 2)

    label2, cen2 = Kmeans(reconstructed_data, 2)
    count_label(label2)
>>>>>>> Stashed changes

    label = clusttering(reconstructed_data, 0.5, 5)

<<<<<<< Updated upstream
    cen = find_centroids(reconstructed_data, 0.5, 5)
=======
    centroid1 = reconstructed_data[0]
    centroid2 = reconstructed_data[-1]

    neighbours1 = get_neighbors(reconstructed_data, centroid1, n)
    neighbours2 = get_neighbors(reconstructed_data, centroid2, n)

    check_neigbours(reconstructed_data, centroid1, centroid2, n, neighbours1, neighbours2)
    check_repeated(neighbours1, neighbours2, centroid1, centroid2)
    find_cluster(neighbours1, neighbours2, reconstructed_data)

    n = 2 
    cen, cluster1, cluster2 = iterate(reconstructed_data, neighbours1, neighbours2, centroid1, centroid2, n)

    get_label(cluster1, cluster2, reconstructed_data, True, cen)
    plot_clusters(reconstructed_data, label2, "KMeans", True, cen2)

>>>>>>> Stashed changes

    plot_clusters(reconstructed_data, label, 'DBSCAN', True, cen)




