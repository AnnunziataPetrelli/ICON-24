import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import os
import shutil
import numpy as np
import cv2
import imghdr

path_directory_train = "data/train/train/"
path_directory_test = "data/test/test/"
dsize_img = 128

def preProcessingCSV():
    df = pd.read_csv("Data/heart.csv")

    print("Inizio preProcessing dataset CSV \n")
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    print("PreProcessing dataset CSV salvato e completato con successo! \n")
    df.to_csv("Data/heart_preprocessed.csv", index=False)

def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype("uint8")

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype("uint8")

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype("uint8")
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    cv2.merge((equ_b, equ_g, equ_r))

    return img_out


def removeResizeImage(data_dir):
    print("Cerco file non compatibili..", data_dir)
    image_exts = ["jpeg", "jpg", "bmp", "png"]
    for image in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip is None:
                print(f"Estensione non riconosciuta: {image_path}")
                print(f"Il file è stato eliminato: {image_path}")
                os.remove(image_path)
            elif tip not in image_exts:
                print("Estensione del file non compatibile {}".format(image_path))
                print(f"Il file è stato eliminato: {image_path}")
                os.remove(image_path)

            resized_img = cv2.resize(img, (dsize_img, dsize_img))
            resized_img = histogram_equalization(resized_img)
            cv2.imwrite(image_path, resized_img)
        except Exception:
            print("Problema con la seguente immagine {}".format(image_path))
            print(f"Il file è stato eliminato: {image_path}")
            os.remove(image_path)
    return

def preProcessingImage():
    print("Inizio preProcessing dataset immagini \n")

    removeResizeImage(path_directory_train + "true")
    removeResizeImage(path_directory_train + "false")
    removeResizeImage(path_directory_test + "true")
    removeResizeImage(path_directory_test + "false")

    print("PreProcessing dataset immagini salvate e completato con successo! \n")