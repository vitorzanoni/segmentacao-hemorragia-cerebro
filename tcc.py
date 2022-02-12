import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import morphology
from PIL import Image

image_original = pydicom.dcmread("CT000134.dcm")
slope = image_original.RescaleSlope
intercept = image_original.RescaleIntercept
image_original = image_original.pixel_array
image_original = image_original.astype(float)
image_original = image_original * slope + intercept

plt.imshow(image_original, 'gray')
plt.show()

final_image_original = Image.fromarray(image_original)
plt.imshow(final_image_original, 'gray')
plt.show()

#connected components opencv
#fazer erosao e depois dilatacao
image = pydicom.dcmread("CT000134.dcm")
slope = image.RescaleSlope
intercept = image.RescaleIntercept

image = image.pixel_array
image[image == -2000] = 0
image = image.astype(float)
image = image * slope + intercept
# multiplicar slope e somar intercept, segmentacao -> remover de -200 e +1000

# plt.hist(image.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

image[image <= 55] = 0
image[image >= 100] = 0
plt.hist(image.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

plt.imshow(image, 'gray')
plt.show()

# final_image = Image.fromarray(image)
# plt.imshow(final_image, 'gray')
# plt.show()

image_erosion = morphology.erosion(image, morphology.disk(5)).astype(np.uint8)
image_erosion_2 = morphology.erosion(image, morphology.disk(8)).astype(np.uint8)

plt.imshow(image_erosion, 'gray')
plt.show()

plt.imshow(image_erosion_2, 'gray')
plt.show()

image_dilation_1 = morphology.binary_dilation(image_erosion_2, morphology.square(8)).astype(np.uint8)
image_dilation_2 = morphology.binary_dilation(image_erosion_2, morphology.diamond(8)).astype(np.uint8)
image_dilation_3 = morphology.binary_dilation(image_erosion_2, morphology.disk(8)).astype(np.uint8)
image_dilation_4 = morphology.binary_dilation(image_erosion_2, morphology.star(8)).astype(np.uint8)

fig, ax = plt.subplots(2, 2, figsize=[12, 12])
ax[0, 0].set_title("square")
ax[0, 0].imshow(image_dilation_1, 'gray')
ax[0, 0].axis('off')
ax[0, 1].set_title("diamond")
ax[0, 1].imshow(image_dilation_2, 'gray')
ax[0, 1].axis('off')
ax[1, 0].set_title("disk")
ax[1, 0].imshow(image_dilation_3, 'gray')
ax[1, 0].axis('off')
ax[1, 1].set_title("star")
ax[1, 1].imshow(image_dilation_4, 'gray')
ax[1, 1].axis('off')
plt.show()

fig, ax = plt.subplots(2, 2, figsize=[12, 12])
ax[0, 0].set_title("square")
ax[0, 0].imshow(image_dilation_1*image_original, 'gray')
ax[0, 0].axis('off')
ax[0, 1].set_title("diamond")
ax[0, 1].imshow(image_dilation_2*image_original, 'gray')
ax[0, 1].axis('off')
ax[1, 0].set_title("disk")
ax[1, 0].imshow(image_dilation_3*image_original, 'gray')
ax[1, 0].axis('off')
ax[1, 1].set_title("star")
ax[1, 1].imshow(image_dilation_4*image_original, 'gray')
ax[1, 1].axis('off')
plt.show()