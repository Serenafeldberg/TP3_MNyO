
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids



def euclidean_distance(x1, x2, sigma=0.5):
    return np.linalg.norm(x1 - x2)
#   return np.exp(-((np.linalg.norm(x1 - x2))**2)/(2 * (sigma**2)) )

def get_neighbors(data, centroid, eps):

    neighbors = []

    for idx, point in enumerate(data):

        if euclidean_distance(centroid, point) <= eps:
            neighbors.append(idx)

    return neighbors


def find_cluster(neighbours1, neighbours2, data):

    for idx, point in enumerate(data):

        minimum_distance_1 = euclidean_distance(data[idx], data[neighbours1[0]])
        minimum_distance_2 = euclidean_distance(data[idx], data[neighbours2[0]])

        for neighbours_index in neighbours1:
            compare = euclidean_distance(data[idx], data[neighbours_index])
            if (compare < minimum_distance_1):
                minimum_distance_1 = compare

    
        for neighbours_index in neighbours2:
            compare = euclidean_distance(data[idx], data[neighbours_index])
            if (compare < minimum_distance_2):
                minimum_distance_2 = compare
            
        
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

    plt.legend()
    plt.show()



if __name__ == "__main__":

    data_ = data()
    reconstructed_data = dec_svd(data_, 2)

    label2, cen = Kmeans(reconstructed_data, 2)


    neighbours1 = get_neighbors(reconstructed_data, cen[0], 2)
    neighbours2 = get_neighbors(reconstructed_data, cen[1], 2)

    find_cluster(neighbours1, neighbours2, reconstructed_data)
    get_label(neighbours1, neighbours2, reconstructed_data, True, cen)







