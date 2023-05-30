
import numpy as np
import csv
import matplotlib.pyplot as plt

# Load the data
data = []
with open('dataset_clusters.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        values = []
        for val in row:
            values.append(float(val))

        data.append(np.array(values))

data = np.array(data)
print(data.shape)

# Perform singular value decomposition
U, s, Vt = np.linalg.svd(data)

U = U[:, :2]
s = np.diag(s[:2])
Vt = Vt[:2, :]

# Reconstruct the data using the first 2 components
reconstructed_data = U @ s @ Vt

# Plot the reconstructed data
plt.figure()
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], c='r')
plt.title('Reconstructed data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()




