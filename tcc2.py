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


def componentesConectados(imagem, seg_max, cc_erosion):
    original = window_image(imagem, 60, 120).astype(np.uint8)
    imagem[imagem > seg_max] = 0
    imagem = imagem * \
        morphology.binary_opening(imagem, morphology.disk(cc_erosion))
    imagem = window_image(imagem, 60, 120).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        imagem)
    sizes = stats[:, -1]
    if(len(sizes) > 1):
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 1
    else:
        img2 = imagem
    antes = img2
    img2 = morphology.binary_closing(img2, morphology.disk(cc_erosion))
    depois = img2
    original = original * depois
    antes = antes * 255
    depois = depois * 255
    teste = depois - antes
    antes = cv2.cvtColor(antes.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    depois = cv2.cvtColor(depois.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    teste = cv2.cvtColor(teste.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    seg1 = np.concatenate((antes, depois), axis=1)
    seg2 = np.concatenate((teste, original), axis=1)
    seg3 = np.concatenate((seg1, seg2), axis=0)
    cv2.imshow('Teste', seg3)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    for i in range(len(img2)):
        if(1 in img2[i]):
            img2[i:i + round(len(img2)*0.05), :] = 0
            break
    for i in range(len(img2) - 1, 0, -1):
        if(1 in img2[i]):
            img2[i - round(len(img2)*0.05):, :] = 0
            break
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


def load_scan_sorted(list_dicom_paths):
    slices = []
    for path in list_dicom_paths:
        slices.append([path, pydicom.dcmread(path)])
    slices = sorted(slices, key=lambda x: x[1].ImagePositionPatient[-1])
    imagens = []
    for slice in slices:
        imagens.append(slice[0])
    imagens = imagens[round(len(imagens)*0.4):]
    return imagens[:round(len(imagens)*0.8)]


def findHemorragia(imagens, diretorio, seg_min, seg_max, disk_size, cc_erosion):  # , area_size
    # if os.path.basename(diretorio) == "CQ500CT100 CQ500CT100":
    fw = open("tcc_teste.csv", "a")
    count = 0
    imagens = load_scan_sorted(imagens)
    print("cc_erosion -> %d" % cc_erosion)
    print(diretorio)
    for i in imagens:
        imagem = getImagem(i)
        imagem = imagem * componentesConectados(imagem, seg_max, cc_erosion)
        imagem[imagem > seg_max] = 0  # 75
        area = window_image(imagem, 60, 120).astype(np.uint8)
        imagem[imagem < seg_min] = 0  # 60
        antes = window_image(imagem, 60, 120).astype(np.uint8)
        imagem = morphology.binary_opening(
            imagem, morphology.disk(disk_size))  # 3
        # imagem = morphology.area_opening(imagem, area_size)  # 50
        if imagem.max() > 0:
            count += 1
            original = window_image(getImagem(i), 60, 120).astype(np.uint8)
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            imagem = window_image(imagem, 60, 120).astype(np.uint8)
            depois = imagem * 255
            borda = cv2.dilate(imagem, None) - imagem
            contours = cv2.findContours(
                borda, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(original, contours, -1, (0, 0, 255), 1)
            msg = str(count)
            cv2.putText(original, msg, (20, 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
            area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
            antes = cv2.cvtColor(antes, cv2.COLOR_GRAY2BGR)
            depois = cv2.cvtColor(depois, cv2.COLOR_GRAY2BGR)
            seg1 = np.concatenate((area, antes), axis=1)
            seg2 = np.concatenate((depois, original), axis=1)
            seg3 = np.concatenate((seg1, seg2), axis=0)
            cv2.imshow('Segmentacao', seg3)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

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

            # print(i)
    if count >= 1:
        fw.write(os.path.basename(diretorio) + ";1;" + str(count) + "\n")
        fw.close()
        return 1
    fw.write(os.path.basename(diretorio) + ";0;" + str(count) + "\n")
    fw.close()
    return 0


RES = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
diretorios = findDiretorios()
# print(diretorios)


def performance(bounds):
    seg_min, seg_max, disk_size, cc_erosion = bounds  # , area_size
    # print("%d, %d, %d" % (seg_min, seg_max, disk_size))
    fw = open("tcc_teste.csv", "w")
    fw.write("PASTA;RES\n")
    fw.close()
    diretorios_teste = []
    validacao = []
    res = []
    i = 0
    for diretorio in diretorios:
        if (os.path.isdir(diretorio)) and (len(diretorios_teste) < 40):
            diretorios_teste.append(diretorio)
            validacao.append(RES[i])
            res.append(findHemorragia(findImagens(diretorio),
                       diretorio, seg_min, seg_max, disk_size, cc_erosion))
        i += 1
    # print(validacao)
    # print(res)
    score = f1_score(validacao, res)
    f1txt = open("resultados.txt", "a")
    f1txt.write("%d, %d, %d, %d\n" % (seg_min, seg_max, disk_size, cc_erosion))
    f1txt.write("F1-Score: %.2f\n" % (score))
    f1txt.close()
    # print("F1-Score:", score)
    return score


def main():
    bounds = [(50, 75), (75, 101), (2, 6), (2, 11)]  # , (0, 1000)
    result = differential_evolution(performance, bounds)
    print(result)


main()
