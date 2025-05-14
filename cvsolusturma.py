import os
import cv2
import numpy as np
import pandas as pd
from skimage import color, measure, io
from scipy.stats import skew
import mahotas as mh

# 1. Renk Özellikleri
def renk_ozellik(img):
    red, green, blue = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red_mean, green_mean, blue_mean = np.mean(red), np.mean(green), np.mean(blue)
    red_std, green_std, blue_std = np.std(red), np.std(green), np.std(blue)
    lab_image = color.rgb2lab(img)
    l_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    return [round(x, 3) for x in [red_mean, green_mean, blue_mean, red_std, green_std, blue_std, l_mean, a_mean, b_mean]]

def renk_moment_ozellik(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return [round(x, 3) for x in [
        np.mean(h), np.std(h), skew(h.flatten()),
        np.mean(s), np.std(s), skew(s.flatten()),
        np.mean(v), np.std(v), skew(v.flatten())
    ]]

# 2. Şekil Özellikleri
def boyut_ozellik(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    props = measure.regionprops(measure.label(binary))
    if not props: return [0] * 9
    p = props[0]
    return [round(x, 3) for x in [
        p.area, p.perimeter**2 / (4 * np.pi * p.area), p.convex_area,
        p.eccentricity, p.equivalent_diameter, p.extent,
        p.orientation, p.perimeter, p.solidity
    ]]

# 3. Doku Özellikleri
def haralick_ozellik(img):
    gray = color.rgb2gray(img)
    gray = (gray * 255).astype(np.uint8)
    return [round(x, 3) for x in mh.features.haralick(gray).mean(axis=0)]

# Veri Seti Oluşturma
def process_images(input_folder, output_csv):
    data = []
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(input_folder, filename)
            img = io.imread(path)
            if img is None:
                print("Hatalı görsel:", path)
                continue
            renk = renk_ozellik(img)
            renk_moment = renk_moment_ozellik(img)
            boyut = boyut_ozellik(img)
            doku = haralick_ozellik(img)
            label = 2 if i < 119 else 1  # nohut = 1, mercimek = 2
            data.append(renk + renk_moment + boyut + doku + [label])

    col_names = [
        'red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std',
        'l_mean', 'a_mean', 'b_mean',
        'h_mean', 'h_std', 'h_skew', 's_mean', 's_std', 's_skew', 'v_mean', 'v_std', 'v_skew',
        'area', 'circularity', 'convex_area', 'eccentricity', 'equivalent_diameter', 'extent',
        'orientation', 'perimeter', 'solidity'
    ] + [f'haralick_{i}' for i in range(13)] + ['label']

    df = pd.DataFrame(data, columns=col_names)
    df.to_csv(output_csv, index=False)
process_images("C:\c\karma", r"C:\Users\erenh\OneDrive\Desktop\soncsv\veri_nisan4390.csv")
