import cv2
import numpy as np
import os

# =====================================================
# CARPETAS
# =====================================================

INPUT_FOLDER = "cad_original"
OUTPUT_FOLDER = "dataset"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# LISTA DE IMAGENES
# =====================================================

imagenes = [
    "img_00_000.png",
    "img_01_010.png",
    "img_02_020.png",
    "img_03_030.png",
    "img_04_040.png",
    "img_05_050.png",
    "img_06_060.png",
    "img_07_070.png",
    "img_08_080.png",
    "img_09_090.png",
    "img_10_100.png",
    "img_11_110.png",
    "img_12_120.png",
    "img_13_130.png",
    "img_14_140.png",
    "img_15_150.png",
    "img_16_160.png",
    "img_17_170.png",
    "img_18_180.png",
    "img_19_190.png"
]

# =====================================================
# PROCESAMIENTO DATASET
# =====================================================

for nombre in imagenes:

    path = os.path.join(INPUT_FOLDER, nombre)

    img = cv2.imread(path)

    if img is None:
        print(f"No se encontro: {nombre}")
        continue

    # =================================================
    # ESCALA DE GRISES
    # =================================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # =================================================
    # MASCARA FIGURA
    # =================================================

    _, mask = cv2.threshold(
        gray,
        240,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =================================================
    # NUEVO FONDO
    # =================================================

    fondo = np.ones_like(img) * 255

    # =================================================
    # NUEVO COLOR FIGURA
    # =================================================

    figura = np.zeros_like(img)

    # COLOR VERDE
    figura[:, :] = (0, 255, 0)

    # Aplicar figura sobre fondo
    fondo[mask > 0] = figura[mask > 0]

    # =================================================
    # REDIMENSIONAMIENTO
    # =================================================

    escala = 1.2

    h, w = fondo.shape[:2]

    centro = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(
        centro,
        0,
        escala
    )

    procesada = cv2.warpAffine(
        fondo,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255)
    )

    # =================================================
    # GUARDAR IMAGEN
    # =================================================

    salida = os.path.join(
        OUTPUT_FOLDER,
        nombre
    )

    cv2.imwrite(
        salida,
        procesada
    )

    print(f"Procesada: {nombre}")

print("\nDataset generado correctamente.")