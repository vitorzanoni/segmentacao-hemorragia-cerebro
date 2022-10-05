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
    # imagem[imagem == -2000] = 0
    imagem = imagem * slope + intercept
    return imagem


def mostraImagem(imagem):
    plt.imshow(imagem, 'gray')
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


def segmentacao(imagem, window_center, window_width):
    # np.set_printoptions(threshold=sys.maxsize)
    # fw = open("imagem.txt", "w")
    # fw.write(str(imagem))
    # fw.close()

    img_min = window_center - window_width//2  # minimum HU level
    img_max = window_center + window_width//2  # maximum HU level
    min = (60 - img_min) / (img_max - img_min)*255.0
    max = (90 - img_min) / (img_max - img_min)*255.0

    imagem[imagem < min] = 0
    imagem[imagem > max] = 0
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
    imagem = morphology.binary_dilation(
        imagem, morphology.disk(8)).astype(np.uint8)
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
    labeled_img[label_hue == 0] = 0
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
    for name in glob.glob("C:/Users/vitor/Downloads/Vitor/tcc/PD1/TCC/CQ500/*", recursive=False):
        diretorios.append(name)
    return diretorios


def findImagens(diretorio):
    pastas = glob.glob(diretorio + "/*/*", recursive=False)

    maisImagens = pastas[0]
    count = len(glob.glob(pastas[0] + "/*", recursive=False))
    pastas.pop(0)
    for pasta in pastas:
        tmpCount = len(glob.glob(pasta + "/*", recursive=False))
        if tmpCount > count:
            maisImagens = pasta
            count = tmpCount

    imagens = []
    for name in glob.glob(maisImagens + "/*", recursive=False):
        imagens.append(name)
    return imagens


def window_image(img: np.ndarray,
                 window_center: int,
                 window_width: int,
                 rescale: bool = True) -> np.ndarray:

    img = img.astype(np.float32)
    # for translation adjustments given in the dicom file.
    img_min = window_center - window_width//2  # minimum HU level
    img_max = window_center + window_width//2  # maximum HU level
    # set img_min for all HU levels less than minimum HU level
    img[img < img_min] = img_min
    # set img_max for all HU levels higher than maximum HU level
    img[img > img_max] = img_max
    if rescale:
        img = (img - img_min) / (img_max - img_min)*255.0
    return img


def findHemorragia(imagens, diretorio):
    fw = open("tcc_res_255.csv", "a")
    for i in imagens:
        original = getImagem(i)
        # mostraImagem(original)
        imagem = window_image(original, 60, 120)
        # mostraImagem(imagem)
        # mostraHounsfield(imagem)
        # imagem_255 = converteEscalaCinza(imagem)
        # mostraEscalaCinza(imagem_255)
        # componentesConectados("cinza.png")
        imagem_seg = segmentacao(imagem, 60, 120)
        # mostraImagem(imagem_seg)
        imagem_seg = erosao(imagem_seg)
        imagem_seg = dilatacao(imagem_seg)
        if imagem_seg.argmax() > 0:
            print(i)
            # mostraImagem(original)
            # plt.imshow(imagem_seg, 'gray')
            # plt.show()
            # aplicaMascara(original, imagem_seg)
            fw.write(os.path.basename(diretorio) + ";1\n")
            return
    fw.write(os.path.basename(diretorio) + ";0\n")
    fw.close()


def main():
    fw = open("tcc_res_255.csv", "w")
    fw.write("PASTA;RES\n")
    fw.close()
    diretorios = findDiretorios()
    for diretorio in diretorios:
        if(os.path.isdir(diretorio)):
            findHemorragia(findImagens(diretorio), diretorio)


main()
