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

fig, axs = plt.subplots(2, 8, figsize=(10, 4))
fig.suptitle('16 images', fontsize=16)

for i, ax in enumerate(axs.flatten()):
    ax.imshow(data[i], cmap='gray')
    ax.axis('off')

plt.show()

# Flatten the images into vectors
flattened_data = data.reshape(data.shape[0], -1)

# Perform singular value decomposition
U, s, Vt = np.linalg.svd(flattened_data)

first_10_U = U[:, :10]
first_10_s = np.diag(s[:10])
first_10_Vt = Vt[:10, :]

# Reconstruct the first image using the first 10 components
reconstructed_image = first_10_U @ first_10_s @ first_10_Vt

fig, axs = plt.subplots (2, 5, figsize=(10, 4))
fig.suptitle('First 10 dimensions', fontsize=16)

for i, ax in enumerate(axs.flatten()):
    ax.imshow(reconstructed_image[i].reshape(data.shape[1], -1), cmap='gray')
    ax.axis('off')
plt.show()

last_10_U = U[:, -10:]
last_10_s = np.diag(s[-10:])
last_10_Vt = Vt[-10:, :]

reconstructed_image = last_10_U @ last_10_s @ last_10_Vt

fig, axs = plt.subplots (2, 5, figsize=(10, 4))
fig.suptitle('Last 10 dimensions', fontsize=16)

for i, ax in enumerate(axs.flatten()):
    ax.imshow(reconstructed_image[i].reshape(data.shape[1], -1), cmap='gray')
    ax.axis('off')
plt.show()



image_to_compress = data[0]

U, s, Vt = np.linalg.svd(image_to_compress)

Sd = np.diag(s)
k  = 100
comprimida = U[:,:k] @ Sd[0:k,:k] @ Vt[:k,:]
plt.imshow(comprimida, cmap='gray')
plt.show()

# Definir el porcentaje máximo de error permitido (en este caso, 5%)
max_error_percentage = 0.05

# Inicializar el número mínimo de dimensiones requeridas
min_dimensions = 1

# Calcular el error inicial
reconstructed_image = U[:, :min_dimensions] @ np.diag(s[:min_dimensions]) @ Vt[:min_dimensions, :]
error = np.linalg.norm(image_to_compress - reconstructed_image, ord='fro') / np.linalg.norm(image_to_compress, ord='fro')

# Incrementar el número de dimensiones hasta que se cumpla el error máximo
while error > max_error_percentage:
    min_dimensions += 1
    reconstructed_image = U[:, :min_dimensions] @ np.diag(s[:min_dimensions]) @ Vt[:min_dimensions, :]
    error = np.linalg.norm(image_to_compress - reconstructed_image, ord='fro') / np.linalg.norm(image_to_compress, ord='fro')

print("Número mínimo de dimensiones requeridas:", min_dimensions)