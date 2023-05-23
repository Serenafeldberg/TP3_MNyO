import numpy as np
import matplotlib.pyplot as plt
import zipfile
from PIL import Image

# Extraction of the images from the zip file
with zipfile.ZipFile('dataset_imagenes.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the images into a data matrix
data = []
for i in range(0, 16):
    if (i < 10):
        img_path = f'img0{i}.jpeg'
    else:
        img_path = f'img{i}.jpeg'
    img = Image.open(img_path)
    data.append(np.array(img))

data = np.stack(data)
print("data shape", data.shape)

# Flatten the images into vectors
flattened_data = data.reshape(data.shape[0], -1)
print("flattened_data shape", flattened_data.shape)

# Perform singular value decomposition
U, s, Vt = np.linalg.svd(flattened_data)

print(U.shape)
print(s.shape)
print(Vt.shape)

# Visualize the first 10 dimensions (eigenvectors)
fig, axs = plt.subplots(2, 8, figsize=(10, 4))
fig.suptitle('16 images', fontsize=16)

for i, ax in enumerate(axs.flatten()):
    ax.imshow(data[i], cmap='gray')
    ax.axis('off')

#plt.show()

# Visualize the first 10 dimensions (eigenvectors)
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle('First 10 Dimensions (Eigenvectors)')

for i, ax in enumerate(axs.flatten()):
    print(U.shape)
    ax.imshow(U, cmap='gray')
    ax.axis('off')

plt.show()