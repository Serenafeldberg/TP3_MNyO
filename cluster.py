
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


def euclidean_distance(x1, x2, sigma=100):
    return np.linalg.norm(x1 - x2)**2/sigma
   # return np.exp(-((np.linalg.norm(x1 - x2))**2)/(2 * (sigma**2)) )

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

def plot_clusters (data, label, title, bool, cen):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    if bool:
        plt.scatter(cen[:,0], cen[:,1], c='red', marker='x')
    plt.show()

if __name__ == "__main__":

    n = 2
    data_ = data()
    reconstructed_data = dec_svd(data_, 20)

    label2, cen = Kmeans(reconstructed_data, 2)

    count_label(label2)

    matrix = euclidean_matrix(reconstructed_data)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    neighbours1 = get_neighbors(reconstructed_data, cen[0], n)
    neighbours2 = get_neighbors(reconstructed_data, cen[1], n)

    while len(neighbours1) == 0 or len(neighbours2) == 0:
        n += 1
        neighbours1 = get_neighbors(reconstructed_data, cen[0], n)
        neighbours2 = get_neighbors(reconstructed_data, cen[1], n)

    check_repeated(neighbours1, neighbours2, cen[0], cen[1])
    find_cluster(neighbours1, neighbours2, reconstructed_data)
    get_label(neighbours1, neighbours2, reconstructed_data, True, cen)
    plot_clusters(reconstructed_data, label2, "Kmeans", True, cen)





