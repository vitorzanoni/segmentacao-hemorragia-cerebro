import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import morphology
from PIL import Image, ImageOps
import cv2
import glob
import os

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
    plt.xlabel("Unidades de Hounsfield")
    plt.ylabel("Frequência")
    plt.show()

def segmentacao(imagem):
    imagem[imagem <= 60] = 0
    imagem[imagem >= 100] = 0
    # plt.ylim([0, 2500])
    # plt.hist(imagem.flatten(), bins=100, color='c')
    # plt.xlabel("Unidades de Hounsfield")
    # plt.ylabel("Frequência")
    # plt.show()
    return imagem

def erosao(imagem):
    imagem = morphology.erosion(imagem, morphology.disk(8)).astype(np.uint8)
    # plt.imshow(imagem, 'gray')
    # plt.show()
    return imagem

def dilatacao(imagem):
    imagem = morphology.binary_dilation(imagem, morphology.disk(8)).astype(np.uint8)
    # plt.imshow(imagem, 'gray')
    # plt.show()
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

def findDiretorios():
    diretorios = []
    for name in glob.glob("E:/TCC/IMAGENS/CQ500/*", recursive = False):
        diretorios.append(name)
    return diretorios

def findImagens(diretorio):
    pastas = glob.glob(diretorio + "/*/*", recursive = False)

    menosImagens = pastas[0]
    count = len(glob.glob(pastas[0] + "/*", recursive = False))
    # print("1 -> " + str(count))
    pastas.pop(0)
    for pasta in pastas:
        tmpCount = len(glob.glob(pasta + "/*", recursive = False))
        if tmpCount < count:
            # print("2 -> " + str(tmpCount))
            menosImagens = pasta
            count = tmpCount
    # print("3 -> " + menosImagens)

    imagens = []
    for name in glob.glob(menosImagens + "/*", recursive = False):
        imagens.append(name)
    return imagens

def findHemorragia(imagens, diretorio):
    fw = open("tcc_res.csv", "a")
    for i in imagens:
        imagem = getImagem(i)
        # original = getImagem(i)
        # mostraImagem(imagem)
        # mostraHounsfield(imagem)
        # imagem_255 = converteEscalaCinza(imagem)
        # mostraEscalaCinza(imagem_255)
        # componentesConectados("cinza.png")
        imagem_seg = segmentacao(imagem)
        # mostraImagem(imagem_seg)
        imagem_seg = erosao(imagem_seg)
        imagem_seg = dilatacao(imagem_seg)
        if imagem_seg.argmax() > 0:
            print(i)
            # mostraImagem(original)
            # plt.imshow(imagem_seg, 'gray')
            # plt.show()
            # aplicaMascara(imagem, imagem_seg)
            fw.write(os.path.basename(diretorio) + ";1\n")
            return
    fw.write(os.path.basename(diretorio) + ";0\n")
    fw.close()

# connected components opencv
# fazer erosao e depois dilatacao
# multiplicar slope e somar intercept, segmentacao -> remover de -200 e +1000
def main():
    fw = open("tcc_res.csv", "w")
    fw.write("PASTA;RES\n")
    fw.close()
    diretorios = findDiretorios()
    for diretorio in diretorios:
        if(os.path.isdir(diretorio)):
            findHemorragia(findImagens(diretorio), diretorio)

main()