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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.ndimage import gaussian_filter
from datetime import datetime
import seaborn as sns


def get_imagem(dicom):
    dicom = pydicom.dcmread(dicom)
    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    imagem = dicom.pixel_array
    imagem[imagem == -2000] = 0
    imagem = imagem * slope + intercept
    return imagem


def salva_cc(diretorio, count, nb_components, output):
    num_labels, labels = nb_components, output
    # Create random colors for each label
    colors = [tuple(np.random.randint(0, 255, 3).tolist())
              for _ in range(num_labels)]
    # Create an empty image to draw on
    imagem_cc = np.zeros(
        (labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    # Color each connected component with a unique color
    for label in range(1, num_labels):
        color = colors[label]
        mask = labels == label
        imagem_cc[mask] = color
    os.makedirs(os.path.basename(
        diretorio), exist_ok=True)
    cv2.imwrite("./" + os.path.basename(diretorio) +
                "/" + str(count) + "_COR.png", imagem_cc)

    return imagem_cc


def mostra_cc(original, imagem_cc, maior_area, count, diretorio):
    # antes = morphology.binary_closing(
    #     maior_area, morphology.disk(8))
    antes = maior_area
    depois = original * antes
    antes = antes * 255
    cor = cv2.cvtColor(imagem_cc.astype(np.uint8), cv2.COLOR_RGB2BGR)
    antes = cv2.cvtColor(antes.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    depois = cv2.cvtColor(depois.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    seg1 = np.concatenate((cor, antes), axis=1)
    seg2 = np.concatenate((original, depois), axis=1)
    seg3 = np.concatenate((seg1, seg2), axis=0)
    # cv2.imshow('Componentes Conectados', seg3)
    os.makedirs(os.path.basename(
        diretorio), exist_ok=True)
    cv2.imwrite("./" + os.path.basename(diretorio) +
                "/" + str(count) + "_OG.png", original)
    cv2.imwrite("./" + os.path.basename(diretorio) +
                "/" + str(count) + "_CC.png", depois)


def teste_mascara(cc_erosion, original, maior_area):
    antes = maior_area
    depois = morphology.binary_closing(
        maior_area, morphology.disk(cc_erosion))
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


def componentes_conectados(imagem, seg_max, diretorio, count):
    original = window_image(imagem, 60, 120).astype(np.uint8)
    imagem[imagem > seg_max] = 0

    seg1 = cv2.cvtColor(window_image(imagem, 60, 120).astype(
        np.uint8), cv2.COLOR_GRAY2BGR)
    os.makedirs(os.path.basename(diretorio), exist_ok=True)
    cv2.imwrite("./" + os.path.basename(diretorio) +
                "/" + str(count) + "_SG1.png", seg1)

    imagem = window_image(imagem, 60, 120).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        imagem)
    sizes = stats[:, -1]
    if (len(sizes) > 1):
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        maior_area = np.zeros(output.shape)
        maior_area[output == max_label] = 1
        imagem_cc = salva_cc(diretorio, count, nb_components, output)
        mostra_cc(original, imagem_cc, maior_area,
                  count, diretorio)
    else:
        maior_area = imagem

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    for i in range(len(maior_area)):
        if (1 in maior_area[i]):
            maior_area[i:i + round(len(maior_area)*0.05), :] = 0
            break
    for i in range(len(maior_area) - 1, 0, -1):
        if (1 in maior_area[i]):
            maior_area[i - round(len(maior_area)*0.05):, :] = 0
            break
    return maior_area


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


def find_diretorios():
    diretorios = []
    for name in glob.glob("C:/Users/vitor/Downloads/Vitor/tcc/PD1/TCC/CQ500/*", recursive=False):
        if (name.find(".zip") == -1):
            diretorios.append(name)
    return sorted(diretorios)


def find_imagens(diretorio):
    pastas = glob.glob(diretorio + "/*/*", recursive=False)

    mais_imagens = pastas[0]
    count = len(glob.glob(pastas[0] + "/*", recursive=False))
    pastas.pop(0)
    for pasta in pastas:
        tmp_count = len(glob.glob(pasta + "/*", recursive=False))
        if tmp_count > count:
            mais_imagens = pasta
            count = tmp_count

    imagens = []
    for name in glob.glob(mais_imagens + "/*", recursive=False):
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
    total = len(imagens)
    inicio = round(total*0.45)
    fim = round(total*0.85)
    return inicio, total, imagens[inicio:fim]


def apply_gaussian_filter(seg_min, seg_max, sigma_val, imagem):
    gaussian_image = gaussian_filter(imagem, sigma=sigma_val)
    hemorrhage = gaussian_image.copy()
    hemorrhage[hemorrhage > seg_max] = 0
    hemorrhage[hemorrhage < seg_min] = 0
    return hemorrhage, gaussian_image


def mostra_fg(i, cc_mask, gaussian_image):
    teste1 = window_image(get_imagem(i), 60, 120).astype(np.uint8)
    teste1 = cv2.cvtColor(teste1, cv2.COLOR_GRAY2BGR)
    borda = cv2.dilate(window_image(cc_mask, 60, 120).astype(
        np.uint8), None) - window_image(cc_mask, 60, 120).astype(np.uint8)
    contours = cv2.findContours(
        borda, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(teste1, contours, -1, (0, 0, 255), 1)
    teste2 = cv2.cvtColor(window_image(
        gaussian_image, 60, 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    seg4 = np.concatenate((teste1, teste2), axis=1)
    cv2.imshow('Filtro Gaussiano', seg4)


def mostra_seg(img_num, total, i, imagem, area, antes):
    original = window_image(get_imagem(i), 60, 120).astype(np.uint8)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    imagem = window_image(imagem, 60, 120).astype(np.uint8)
    depois = imagem * 255
    borda = cv2.dilate(imagem, None) - imagem
    contours = cv2.findContours(
        borda, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(original, contours, -1, (0, 0, 255), 1)
    msg = "Slice: " + str(img_num) + "/" + str(total)
    cv2.putText(original, msg, (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
    antes = cv2.cvtColor(antes, cv2.COLOR_GRAY2BGR)
    depois = cv2.cvtColor(depois, cv2.COLOR_GRAY2BGR)
    seg1 = np.concatenate((area, antes), axis=1)
    seg2 = np.concatenate((depois, original), axis=1)
    seg3 = np.concatenate((seg1, seg2), axis=0)
    # cv2.imshow('Segmentacao', seg3)
    return original


def find_hemorragia(imagens, diretorio, seg_min, seg_max, disk_size, sigma_val, res_atual):
    fw = open("tcc_teste.csv", "a")
    count = 0
    img_num, total, imagens = load_scan_sorted(imagens)
    exame_ini = datetime.now()
    for i in imagens:
        slice_ini = datetime.now()
        img_num += 1
        imagem = get_imagem(i)
        cc_mask = componentes_conectados(
            imagem, seg_max, diretorio, img_num)
        imagem = imagem * cc_mask

        seg2 = cv2.cvtColor(window_image(imagem, 60, 120).astype(
            np.uint8), cv2.COLOR_GRAY2BGR)
        os.makedirs(os.path.basename(diretorio), exist_ok=True)
        cv2.imwrite("./" + os.path.basename(diretorio) +
                    "/" + str(img_num) + "_SG2.png", seg2)

        imagem, gaussian_image = apply_gaussian_filter(
            seg_min, seg_max, sigma_val, imagem)
        # mostra_fg(i, cc_mask, gaussian_image)

        fg = cv2.cvtColor(window_image(
            gaussian_image, 60, 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        os.makedirs(os.path.basename(diretorio), exist_ok=True)
        cv2.imwrite("./" + os.path.basename(diretorio) +
                    "/" + str(img_num) + "_FG.png", fg)

        seg3 = cv2.cvtColor(window_image(
            imagem, 60, 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        os.makedirs(os.path.basename(diretorio), exist_ok=True)
        cv2.imwrite("./" + os.path.basename(diretorio) +
                    "/" + str(img_num) + "_SG3.png", seg3)

        area = window_image(gaussian_image, 60, 120).astype(np.uint8)
        antes = window_image(imagem, 60, 120).astype(np.uint8)
        # imagem = morphology.binary_opening(
        #     imagem, morphology.disk(disk_size))

        imagem = morphology.binary_erosion(
            imagem, morphology.disk(disk_size))
        ero = cv2.cvtColor((window_image(
            imagem, 60, 120).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite("./" + os.path.basename(diretorio) +
                    "/" + str(img_num) + "_ERO.png", ero)

        imagem = morphology.binary_dilation(
            imagem, morphology.disk(disk_size))
        dil = cv2.cvtColor((window_image(
            imagem, 60, 120).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite("./" + os.path.basename(diretorio) +
                    "/" + str(img_num) + "_DIL.png", dil)

        if imagem.max() > 0:
            count += 1
            seg = mostra_seg(img_num, total, i, imagem, area, antes)
            os.makedirs(os.path.basename(diretorio), exist_ok=True)
            cv2.imwrite("./" + os.path.basename(diretorio) +
                        "/" + str(img_num) + "_SEG" + str(count) + ".png", seg)
        else:
            seg = mostra_seg(img_num, total, i, imagem, area, antes)
            os.makedirs(os.path.basename(diretorio), exist_ok=True)
            cv2.imwrite("./" + os.path.basename(diretorio) +
                        "/" + str(img_num) + "_NOSEG.png", seg)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        slice_fim = datetime.now()
        slice_temp = slice_fim - slice_ini
        # print(
        #     f'Tempo Slice: {slice_temp.total_seconds()}')

    exame_fim = datetime.now()
    exame_temp = exame_fim - exame_ini
    # print(
    #     f'Tempo Exame: {exame_temp.total_seconds()}')
    if count >= 1:
        fw.write(os.path.basename(diretorio) + ";1;" + str(count) + "\n")
        fw.close()
        return 1
    fw.write(os.path.basename(diretorio) + ";0;" + str(count) + "\n")
    fw.close()
    return 0


RES = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
       0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
       0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
       1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
diretorios = find_diretorios()


def performance(bounds):
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'Inicio: {current_time}')
    f1txt = open("resultados.txt", "a")
    f1txt.write(f'Inicio: {current_time}\n')
    f1txt.close()

    seg_min, seg_max, disk_size, sigma_val = bounds
    print(bounds)
    fw = open("tcc_teste.csv", "w")
    fw.write("PASTA;RES\n")
    fw.close()
    diretorios_teste = []
    validacao = []
    res = []
    i = 0
    for diretorio in diretorios:
        if (os.path.isdir(diretorio)) and (len(diretorios_teste) < 40):
            fw = open("tcc_teste.csv", "a")
            fw.write(str(RES[i]) + ";")
            fw.close()
            diretorios_teste.append(diretorio)
            validacao.append(RES[i])
            res.append(find_hemorragia(find_imagens(diretorio),
                       diretorio, seg_min, seg_max, disk_size, sigma_val, RES[i]))
        i += 1
    score = f1_score(validacao, res)
    f1txt = open("resultados.txt", "a")
    f1txt.write("%.20f, %.20f, %.20f, %.20f\n" %
                (seg_min, seg_max, disk_size, sigma_val))
    f1txt.write("F1-Score: %.20f\n" % (score))
    print("F1-Score:", score)

    matriz = confusion_matrix(validacao, res)
    # cm_display = ConfusionMatrixDisplay(
    #     confusion_matrix=matriz, display_labels=["Saudável", "Não Saudável"])
    # cm_display.plot()
    # cm_display.ax_.set(xlabel='Classificação Predita',
    #                    ylabel='Classificação Real')

    sns.set_theme(font_scale=2, rc={'figure.figsize': (10, 8)})
    ax = sns.heatmap(matriz, annot=True, fmt='g',
                     cmap=sns.color_palette("flare", as_cmap=True))
    ax.set_xlabel("Classificação Predita")
    ax.set_ylabel("Classificação Real")
    ax.xaxis.set_ticklabels(["Saudável", "Não Saudável"])
    ax.yaxis.set_ticklabels(["Saudável", "Não Saudável"])

    now2 = datetime.now()
    now3 = now2 - now
    current_time = now2.strftime("%d/%m/%Y %H:%M:%S")
    print(f'Fim: {current_time}')
    print(
        f'Tempo: {int(now3.total_seconds() // 60)}:{now3.total_seconds() - ((now3.total_seconds() // 60) * 60)}')
    f1txt.write(f'Fim: {current_time}\n')
    f1txt.close()
    return score * -1


def main():
    # bounds = [(50, 75), (75, 101), (2, 6), (0, 6)]
    # result = differential_evolution(performance, bounds)
    # 50.08803987095695475773, 86.71610084319101474648, 4.84795220930039150176, 2.45324398424467071678 # usar esse
    # 50.60006964356053771326, 86.56139984090714278864, 4.67032122564753926497, 2.23910991042969964582
    # 50.31570196656763016563, 86.75417763293270922986, 4.72120205271766835153, 2.56849357717579485083
    # 51.17108169100011849650, 86.59956876783407153653, 3.76213373312758125877, 6.87147069150776701463, 3.33002622294939554237 # original
    # result = performance([51.17108169100011849650, 86.59956876783407153653,
    #                       3.76213373312758125877, 3.33002622294939554237])
    # result = performance([50.31570196656763016563, 86.75417763293270922986,
    #                     4.72120205271766835153, 2.56849357717579485083])
    # result = performance([50.60006964356053771326, 86.56139984090714278864,
    #                      4.67032122564753926497, 2.23910991042969964582])

    result = performance([50.08803987095695475773, 86.71610084319101474648,
                         4.84795220930039150176, 2.45324398424467071678])
    # result = performance([60, 90, 3, 1])
    plt.show()
    print(result)
    f1txt = open("resultados.txt", "a")
    f1txt.write(str(result) + "\n")
    f1txt.close()


main()
