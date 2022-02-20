import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import morphology
from PIL import Image, ImageOps
import cv2

def getImagem(dicom):
    dicom = pydicom.dcmread(dicom)
    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    imagem = dicom.pixel_array
    imagem[imagem == -2000] = 0
    imagem = imagem.astype(float)
    imagem = imagem * slope + intercept
    return imagem

def mostraImagem(imagem):
    plt.imshow(imagem, 'gray', vmin = 0, vmax = 255)
    plt.show()

def converteEscalaCinza(imagem):
    imagem = Image.fromarray(imagem)
    # plt.imshow(imagem, 'gray')
    # plt.show()
    imagem_255 = ImageOps.grayscale(imagem)
    imagem_255.save("cinza.png")
    # plt.imshow(imagem_255, 'gray')
    # plt.show()
    return imagem_255

def mostraEscalaCinza(imagem):
    plt.ylim([0, 20000])
    plt.hist(imagem.getdata(), 255, color='c')
    plt.xlabel("Escala de Cinza")
    plt.ylabel("Frequência")
    plt.show()

def mostraHounsfield(imagem):
    plt.ylim([0, 2500])
    plt.hist(imagem.flatten(), bins=100, color='c')
    plt.xlabel("Unidades de Hounsfield (HU)")
    plt.ylabel("Frequência")
    plt.show()

def segmentacao(imagem):
    imagem[imagem <= 55] = 0
    imagem[imagem >= 100] = 0
    # plt.ylim([0, 2500])
    # plt.hist(imagem.flatten(), bins=100, color='c')
    # plt.xlabel("Unidades de Hounsfield (HU)")
    # plt.ylabel("Frequência")
    # plt.show()
    return imagem

def erosao(imagem):
    imagem = morphology.erosion(imagem, morphology.disk(8)).astype(np.uint8)
    plt.imshow(imagem, 'gray')
    plt.show()
    return imagem

def dilatacao(imagem):
    imagem = morphology.binary_dilation(imagem, morphology.disk(8)).astype(np.uint8)
    plt.imshow(imagem, 'gray')
    plt.show()
    return imagem

def aplicaMascara(imagem, mascara):
    plt.imshow(mascara*imagem, 'gray')
    plt.show()

def componentesConectados(imagem):
    imagem = cv2.imread(imagem, 0)
    imagem = cv2.threshold(imagem, 200, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(imagem)
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.show()

# def plotMultiplasImagens(imagem1, imagem2, imagem3, imagem4):
#     fig, ax = plt.subplots(2, 2, figsize=[12, 12])
#     ax[0, 0].set_title("1")
#     ax[0, 0].imshow(imagem1, 'gray')
#     ax[0, 0].axis('off')
#     ax[0, 1].set_title("2")
#     ax[0, 1].imshow(imagem2, 'gray')
#     ax[0, 1].axis('off')
#     ax[1, 0].set_title("3")
#     ax[1, 0].imshow(imagem3, 'gray')
#     ax[1, 0].axis('off')
#     ax[1, 1].set_title("4")
#     ax[1, 1].imshow(imagem4, 'gray')
#     ax[1, 1].axis('off')
#     plt.show()

# connected components opencv
# fazer erosao e depois dilatacao
# multiplicar slope e somar intercept, segmentacao -> remover de -200 e +1000
def main():
    imagem = getImagem("CT000134.dcm")
    mostraImagem(imagem)
    mostraHounsfield(imagem)
    imagem_255 = converteEscalaCinza(imagem)
    mostraEscalaCinza(imagem_255)
    imagem_seg = segmentacao(imagem)
    mostraImagem(imagem_seg)
    imagem_seg = erosao(imagem_seg)
    imagem_seg = dilatacao(imagem_seg)
    aplicaMascara(imagem, imagem_seg)
    componentesConectados("cinza.png")
main()