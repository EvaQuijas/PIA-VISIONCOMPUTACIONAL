import cv2
import numpy as np
import pandas as pd
import glob
import os

# =====================================================
# CARPETAS
# =====================================================
os.makedirs("resultados_finales", exist_ok=True)

# =====================================================
# DIMENSIONES IMAGEN
# =====================================================
W_px = 1920
H_px = 1080

# =====================================================
# DIMENSIONES MESA (mm)
# =====================================================
W_mm = 320
H_mm = 180

# =====================================================
# FRAME MESA EN ROBOT
# =====================================================
Wx = 200
Wy = -50
Wz = 400

# =====================================================
# ROTACION MESA
# =====================================================
theta_mesa = -30  # grados

# =====================================================
# LONGITUD HERRAMIENTA
# =====================================================
L = 180  # mm

# =====================================================
# ESCALA px -> mm
# =====================================================
sx = W_mm / W_px
sy = H_mm / H_px

# =====================================================
# NORMALIZAR ANGULO
# =====================================================
def normalizar_angulo(angle):

    angle = angle % 180

    if angle > 90:
        angle = angle - 90

    return abs(angle)

# =====================================================
# MATRIZ ROTACION Y
# =====================================================
theta_rad = np.deg2rad(theta_mesa)

Ry = np.array([
    [np.cos(theta_rad), 0, np.sin(theta_rad)],
    [0, 1, 0],
    [-np.sin(theta_rad), 0, np.cos(theta_rad)]
])

# =====================================================
# NORMAL ORIGINAL
# =====================================================
n0 = np.array([0, 0, 1])

# =====================================================
# NORMAL MESA
# =====================================================
n = Ry @ n0

nx, ny, nz = n

print("NORMAL DE LA MESA:", n)

# =====================================================
# DATAFRAME
# =====================================================
data = []

# =====================================================
# RECORRER DATASET
# =====================================================
for path in glob.glob("dataset/*.png"):

    # =================================================
    # CARGA
    # =================================================
    img_color = cv2.imread(path)

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
  
    # =================================================
    # PREPROCESAMIENTO
    # =================================================
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur[0:120, 0:350] = blur[121,351]
    
    # =================================================
    # CANNY
    # =================================================
    edges = cv2.Canny(blur, 30, 120)

    kernel = np.ones((5,5), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =================================================
    # CONTORNOS
    # =================================================
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =================================================
    # FILTRAR CONTORNOS
    # =================================================
    filtered = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # MOMENTOS DEL CONTORNO
        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        # CENTROIDE DEL CONTORNO
        cx_test = int(M["m10"] / M["m00"])
        cy_test = int(M["m01"] / M["m00"])

        
        # FILTRO AREA
        if area > 1:
            filtered.append(cnt)

    if len(filtered) == 0:
        continue    
    # =================================================
    # CONTORNO PRINCIPAL
    # =================================================
    main_cnt = max(filtered, key=cv2.contourArea)

    # =================================================
    # MOMENTOS
    # =================================================
    M = cv2.moments(main_cnt)

    if M["m00"] == 0:
        continue

    # =================================================
    # CENTROIDE
    # =================================================
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])




    # =================================================
    # DIBUJAR CENTROIDE
    # =================================================
    cv2.circle(
        img_color,
        (cx, cy),
        12,
        (0,0,255),
        -1
    )

    # =================================================
    # minAreaRect
    # =================================================
    rect = cv2.minAreaRect(main_cnt)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    angle_rect = rect[-1]

    w_px = rect[1][0]
    h_px = rect[1][1]

    if w_px < h_px:
        angle_rect += 90

    if angle_rect < 0:
        angle_rect += 180

    # =================================================
    # DIMENSIONES FISICAS PIEZA
    # =================================================
    pieza_w_mm = w_px * sx
    pieza_h_mm = h_px * sy

    # =================================================
    # ALTURA PIEZA
    # =================================================
    approx = cv2.approxPolyDP(
        main_cnt,
        0.02 * cv2.arcLength(main_cnt, True),
        True
    )

    if len(approx) <= 5:
        figura = "Rectangulo"
        h_pieza = 20

    else:
        figura = "L"
        h_pieza = 30

    # =================================================
    # VECTOR DIRECTOR
    # =================================================
    theta_obj = np.deg2rad(angle_rect)

    vx = np.cos(theta_obj)
    vy = np.sin(theta_obj)

    # =================================================
    # VECTOR VISUAL
    # =================================================
    length = 300

    x2 = int(cx + vx * length)
    y2 = int(cy + vy * length)

    cv2.arrowedLine(
        img_color,
        (cx, cy),
        (x2, y2),
        (255,0,255),
        6
    )

    # =================================================
    # BOUNDING BOX
    # =================================================
    cv2.drawContours(
        img_color,
        [box],
        0,
        (255,0,0),
        4
    )

    # =================================================
    # ANGULO REAL
    # =================================================
    angle_real = int(
        path.split("_")[-1].split(".")[0]
    )

    # =================================================
    # ERROR
    # =================================================
    angle_real_norm = normalizar_angulo(angle_real)

    angle_rect_norm = normalizar_angulo(angle_rect)

    error = min(
        abs(angle_real_norm - angle_rect_norm),
        abs(angle_real_norm - (angle_rect_norm + 90)),
        abs(angle_real_norm - (angle_rect_norm - 90))
    )

    # =================================================
    # px -> mm
    # =================================================
    dx = cx - (W_px / 2)
    dy = cy - (H_px / 2)

    x_mm = dx * sx
    y_mm = dy * sy

    #FRAME LOCAL MESA
    # =================================================
    P_local = np.array([x_mm, -y_mm, 0])

    # =================================================
    # TRANSFORMACION MESA -> ROBOT
    # =================================================
    P_robot = (Ry @ P_local) + np.array([Wx, Wy, Wz])

    # =================================================
    # TCP
    # =================================================
    offset_tcp = (h_pieza / 2) * n

    X_tcp = P_robot[0] + offset_tcp[0]
    Y_tcp = P_robot[1] + offset_tcp[1]
    Z_tcp = P_robot[2] + offset_tcp[2]

    # =================================================
    # APPROACH
    # =================================================
    offset_app = 100 * n

    X_app = X_tcp + offset_app[0]
    Y_app = Y_tcp + offset_app[1]
    Z_app = Z_tcp + offset_app[2]

    # =================================================
    # WRIST CENTER
    # =================================================
    offset_wrist = L * n

    X_w = X_tcp - offset_wrist[0]
    Y_w = Y_tcp - offset_wrist[1]
    Z_w = Z_tcp - offset_wrist[2]
        
    # =================================================
    # NORMAL VISUAL
    # =================================================
    normal_scale = 220

    xn = int(cx + nx * normal_scale)
    yn = int(cy - nz * normal_scale)

    cv2.arrowedLine(
        img_color,
        (cx, cy),
        (xn, yn),
        (0,255,255),
        5
    )

    # =================================================
    # WRIST VISUAL
    # =================================================
    wrist_px_x = int(cx - nx * 150)
    wrist_px_y = int(cy + nz * 150)

    cv2.circle(
        img_color,
        (wrist_px_x, wrist_px_y),
        12,
        (0,165,255),
        -1
    )

    cv2.line(
        img_color,
        (cx, cy),
        (wrist_px_x, wrist_px_y),
        (0,165,255),
        4
    )

    # =================================================
    # TEXTO ANGULO
    # =================================================
    cv2.putText(
        img_color,
        f"{angle_rect:.1f} deg",
        (cx + 80, cy - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255,0,255),
        3
    )

    # =================================================
    # PANEL INFORMACION
    # =================================================
    tx = 1320

    cv2.putText(
        img_color,
        f"Real: {angle_real:.1f}",
        (tx, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255,255,255),
        3
    )

    cv2.putText(
        img_color,
        f"Detectado: {angle_rect:.2f}",
        (tx, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255,255,0),
        2
    )

    cv2.putText(
        img_color,
        f"Error: {error:.2f}",
        (tx, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0,255,255),
        2
    )

    cv2.putText(
        img_color,
        f"Figura: {figura}",
        (tx, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255,100,255),
        2
    )

    cv2.putText(
        img_color,
        f"Tamano: {pieza_w_mm:.1f} x {pieza_h_mm:.1f} mm",
        (tx, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )

    cv2.putText(
        img_color,
        f"Altura pieza: {h_pieza} mm",
        (tx, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )

    cv2.putText(
        img_color,
        f"TCP: ({X_tcp:.1f},{Y_tcp:.1f},{Z_tcp:.1f})",
        (tx, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        img_color,
        f"CW: ({X_w:.1f},{Y_w:.1f},{Z_w:.1f})",
        (tx, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,165,255),
        2
    )

    cv2.putText(
        img_color,
        f"Approach: ({X_app:.1f},{Y_app:.1f},{Z_app:.1f})",
        (tx, 540),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,255),
        2
    )

    cv2.putText(
        img_color,
        f"Normal: ({nx:.2f},{ny:.2f},{nz:.2f})",
        (tx, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,255),
        2
    )

    # =================================================
    # GUARDAR DATOS
    # =================================================
    data.append([
        os.path.basename(path),
        figura,
        angle_real,
        angle_rect,
        error,
        pieza_w_mm,
        pieza_h_mm,
        h_pieza,
        cx,
        cy,
        x_mm,
        y_mm,
        P_robot[0],
        P_robot[1],
        P_robot[2],
        X_tcp,
        Y_tcp,
        Z_tcp,
        X_w,
        Y_w,
        Z_w,
        X_app,
        Y_app,
        Z_app,
        vx,
        vy,
        nx,
        ny,
        nz
    ])

    # =================================================
    # GUARDAR IMAGEN
    # =================================================
    filename = os.path.basename(path)

    cv2.imwrite(
        f"resultados_finales/{filename}",
        img_color
    )

# =====================================================
# EXCEL
# =====================================================
df = pd.DataFrame(data, columns=[
    "imagen",
    "figura",
    "angulo_real",
    "angulo_detectado",
    "error",
    "pieza_w_mm",
    "pieza_h_mm",
    "altura_pieza_mm",
    "cx_px",
    "cy_px",
    "x_mm",
    "y_mm",
    "X_robot",
    "Y_robot",
    "Z_robot",
    "X_tcp",
    "Y_tcp",
    "Z_tcp",
    "X_w",
    "Y_w",
    "Z_w",
    "X_app",
    "Y_app",
    "Z_app",
    "vx",
    "vy",
    "nx",
    "ny",
    "nz"
])

df.to_excel(
    "resultados_roboticos_fin.xlsx",
    index=False
)

print(df)
print("SISTEMA ROBOTICO COMPLETO FINALIZADO")