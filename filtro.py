import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
from PIL import Image
from skimage import measure, morphology
from sklearn.cluster import KMeans


def leitorCSV():
    dados = pd.read_csv("3_Extrapolation_to_Selected_Series.csv")
    return dados

def findImagens():
    imagens = []
    for name in glob.glob("E:/TCC/IMAGENS/**/*.dcm", recursive = True):
        imagens.append(name)
    return imagens

def separaHemorragia(imagens, seriesInstanceUID):
    encontrados = []
    f = open("hemorragias.csv", "w")
    f.write("diretorio;seriesInstanceUID\n")
    for i in imagens:
        imagem = pydicom.dcmread(i)
        if (imagem.SeriesInstanceUID in seriesInstanceUID) and not(imagem.SeriesInstanceUID in encontrados):
            encontrados.append(imagem.SeriesInstanceUID)
            f.write("%s;%s\n" % (i, imagem.SeriesInstanceUID))
    f.close()

def separaSlice(imagens, sopInstanceUID, data, labelName):
    encontrados = []
    f = open("slices.csv", "w")
    f.write("diretorio;sopInstanceUID;seriesInstanceUID;studyInstanceUID;data;labelName;patientID\n")
    for i in imagens:
        imagem = pydicom.dcmread(i)
        for j in range(len(sopInstanceUID)):
            if (imagem.SOPInstanceUID == sopInstanceUID[j]) and not(imagem.SOPInstanceUID in encontrados):
                encontrados.append(imagem.SOPInstanceUID)
                f.write("%s;%s;%s;%s;%s;%s;%s\n" % (i, imagem.SOPInstanceUID, imagem.SeriesInstanceUID, imagem.StudyInstanceUID, data[j], labelName[j], imagem.PatientID))
    f.close()

def main():
    dados = leitorCSV()
    sopInstanceUID = dados.iloc[:, -6].values
    seriesInstanceUID = dados.iloc[:, -5].values
    studyInstanceUID = dados.iloc[:, -4].values
    data = dados.iloc[:, -3].values
    labelName = dados.iloc[:, -2].values
    imagens = findImagens()
    #separaHemorragia(imagens, seriesInstanceUID)
    separaSlice(imagens, sopInstanceUID, data, labelName)
main()
