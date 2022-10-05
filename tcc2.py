import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import morphology
from PIL import Image, ImageOps
import cv2
import glob
import os
from sklearn.metrics import f1_score
from scipy.optimize import differential_evolution


def getImagem(dicom):
    dicom = pydicom.dcmread(dicom)
    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    imagem = dicom.pixel_array
    imagem[imagem == -2000] = 0
    # imagem[imagem >= 1000] = 0
    # print(imagem.max())
    # print(dicom.pixel_array.shape)
    imagem = imagem * slope + intercept
    # print(imagem.max())
    return imagem


def mostraImagem(imagem):
    plt.imshow(imagem, 'gray', vmin=0, vmax=255)
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
    imagem[imagem < 60] = 0


def erosao_dilatacao(imagem):
    return morphology.binary_opening(imagem, morphology.disk(3))


def erosao(imagem):
    return morphology.binary_erosion(imagem, morphology.disk(3))


def dilatacao(imagem):
    return morphology.binary_dilation(imagem, morphology.disk(3))


def aplicaMascara(imagem, mascara):
    plt.imshow(imagem*mascara, 'gray')
    plt.show()


def componentesConectados(imagem):
    # imagem = cv2.imread(imagem, 0)
    imagem = window_image(imagem, 60, 120).astype(np.uint8)
    # img_min = 60 - 120//2  # minimum HU level
    # img_max = 60 + 120//2  # maximum HU level
    # min = (60 - img_min) / (img_max - img_min)*255.0
    # max = (90 - img_min) / (img_max - img_min)*255.0
    # num_labels, labels = cv2.connectedComponents(imagem.astype(np.uint8))
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        imagem)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    # cv2.imshow("Biggest component", img2)
    # cv2.waitKey()

    # img2 = np.zeros(output.shape)
    # img2[output == max_label] = 255
    # label_hue = np.uint8(255*labels/np.max(labels))
    # blank_ch = 255*np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # labeled_img[label_hue == 0] = 0
    # plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    return img2


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
        if (name.find(".zip") == -1):
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


def teste(imagem):
    img = window_image(imagem, 60, 120).astype(np.uint8)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    cv2.imshow("Biggest component", img2)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", img)

    blur = cv2.medianBlur(img, 31)
    cv2.imshow("blur", blur)
    plt.imshow(blur, vmin=0, vmax=255)
    plt.show()

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)

    canny = cv2.Canny(img, 75, 200)
    cv2.imshow('canny', canny)

    borda = cv2.dilate(thresh, None) - thresh
    cv2.imshow('borda', borda)
    cv2.waitKey()

    # im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contour_list = []
    # for contour in contours:
    #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    #     area = cv2.contourArea(contour)
    #     if 5000 < area < 15000:
    #         contour_list.append(contour)

    # msg = "Total holes: {}".format(len(approx)//2)
    # cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # cv2.drawContours(img, contour_list, -1, (0, 255, 0), 2)
    # cv2.imshow('Objects Detected', img)

    # cv2.imwrite("detected_holes.png", img)

    # cv2.waitKey(0)


def findHemorragia(imagens, diretorio, seg_min, seg_max, disk_size):  # , area_size
    # if os.path.basename(diretorio) == "CQ500CT100 CQ500CT100":
    fw = open("tcc_teste.csv", "a")
    count = 0
    for i in imagens:
        imagem = getImagem(i)
        imagem[imagem > seg_max] = 0  # 75
        imagem = imagem * componentesConectados(imagem)
        imagem[imagem < seg_min] = 0  # 60
        imagem = morphology.binary_opening(
            imagem, morphology.disk(disk_size))  # 3
        # imagem = morphology.area_opening(imagem, area_size)  # 50
        if imagem.max() > 0:
            # original = getImagem(i)
            # imagem = original
            # imagem[imagem > 75] = 0
            # teste(imagem)
            # imagem = imagem * componentesConectados(imagem)

            # fig = plt.figure(figsize=(5, 5))
            # rows = 2
            # columns = 4
            # fig.add_subplot(rows, columns, 8)
            # plt.imshow(morphology.area_opening(
            #     imagem, 15625), vmin=0, vmax=255)
            # plt.axis('off')
            # plt.title("Mask")

            # imagem[imagem < 60] = 0

            # fig.add_subplot(rows, columns, 1)
            # plt.imshow(original, vmin=0, vmax=255)
            # plt.axis('off')
            # plt.title("Ori")

            # fig.add_subplot(rows, columns, 2)
            # plt.imshow(imagem)
            # plt.axis('off')
            # plt.title("Seg")

            # fig.add_subplot(rows, columns, 3)
            # plt.imshow(morphology.area_opening(imagem, 50))
            # plt.axis('off')
            # plt.title("Area Maior 50")

            # fig.add_subplot(rows, columns, 4)
            # plt.imshow(morphology.binary_erosion(
            #     imagem, morphology.disk(3)))
            # plt.axis('off')
            # plt.title("Ero")

            # fig.add_subplot(rows, columns, 5)
            # plt.imshow(morphology.area_opening(morphology.binary_erosion(
            #     imagem, morphology.disk(3)), 50))
            # plt.axis('off')
            # plt.title("Area Ero Maior 50")

            # fig.add_subplot(rows, columns, 6)
            # plt.imshow(morphology.binary_dilation(
            #     morphology.area_opening(morphology.binary_erosion(
            #         imagem, morphology.disk(3)), 50), morphology.disk(3)))
            # plt.axis('off')
            # plt.title("Dila")

            # fig.add_subplot(rows, columns, 7)
            # plt.imshow(morphology.binary_opening(
            #     imagem, morphology.disk(3)))
            # plt.axis('off')
            # plt.title("Binary Open 3")
            # plt.show()

            count += 1
            # print(i)
    if count >= 1:
        fw.write(os.path.basename(diretorio) + ";1;" + str(count) + "\n")
        fw.close()
        return 1
    fw.write(os.path.basename(diretorio) + ";0;" + str(count) + "\n")
    fw.close()
    return 0


RES = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
diretorios = findDiretorios()
# print(diretorios)


def performance(bounds):
    seg_min, seg_max, disk_size = bounds  # , area_size
    f1txt = open("resultados.txt", "a")
    f1txt.write("%d, %d, %d\n" % (seg_min, seg_max, disk_size))
    # print("%d, %d, %d" % (seg_min, seg_max, disk_size))
    fw = open("tcc_teste.csv", "w")
    fw.write("PASTA;RES\n")
    fw.close()
    diretorios_teste = []
    validacao = []
    res = []
    i = 0
    for diretorio in diretorios:
        if (os.path.isdir(diretorio)) and (len(diretorios_teste) < 20):
            diretorios_teste.append(diretorio)
            validacao.append(RES[i])
            res.append(findHemorragia(findImagens(diretorio),
                       diretorio, seg_min, seg_max, disk_size))
        i += 1
    # print(validacao)
    # print(res)
    score = f1_score(validacao, res)
    f1txt.write("F1-Score: %.2f\n" % (score))
    f1txt.close()
    # print("F1-Score:", score)
    return score


def main():
    bounds = [(60, 75), (75, 91), (2, 6)]  # , (0, 1000)
    result = differential_evolution(performance, bounds)
    print(result.x)
    print(result.fun)


main()
