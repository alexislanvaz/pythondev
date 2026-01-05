import cv2
import pytesseract
import numpy as np
import time

# ===================== CONFIGURACIÓN =====================

CAM_INDEX = "./videos/video1.mp4"
CAM_INDEX = 0

UMBRAL_AREA = 2000
LINE_VALUE = 4000
ZOOM_FACTOR = 3
OCR_INTERVALO = 0.3

# ROI iniciales
roi_capacitor = [90, 280, 170, 200]   # [x, y, w, h] en coordenadas globales
roi_texto     = [20, 40, 102, 40]     # [x, y, w, h] relativos al capacitor

rotar = True          # toggle rotación

# Estados mouse
dragging = False
ix, iy = -1, -1
mode_roi = 0          # 0 = capacitor, 1 = texto

# =========================================================

def mouse_callback(event, x, y, flags, param):
    """
    Mouse:
      - Clic izq + arrastrar en modo 'Capacitor'  -> dibuja/mueve ROI capacitor
      - Clic izq + arrastrar en modo 'Texto'      -> dibuja/mueve ROI texto (relativo)
    Teclado:
      - C -> modo capacitor
      - T -> modo texto
    """
    global ix, iy, dragging, mode_roi, roi_capacitor, roi_texto

    frame = param[0]
    if frame is None:
        return

    h, w = frame.shape[:2]

    # Normalizar coordenadas dentro del frame
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        if mode_roi == 0:
            # ROI capacitor en coords absolutas
            x0 = min(ix, x)
            y0 = min(iy, y)
            x1 = max(ix, x)
            y1 = max(iy, y)
            roi_capacitor = [x0, y0, max(5, x1-x0), max(5, y1-y0)]
        else:
            # ROI texto en coords relativas al capacitor
            cx, cy, cw, ch = roi_capacitor
            # Clamp dentro del capacitor
            x_rel0 = max(0, min(min(ix, x) - cx, cw-1))
            y_rel0 = max(0, min(min(iy, y) - cy, ch-1))
            x_rel1 = max(1, min(max(ix, x) - cx, cw))
            y_rel1 = max(1, min(max(iy, y) - cy, ch))
            roi_texto = [
                x_rel0,
                y_rel0,
                max(5, x_rel1 - x_rel0),
                max(5, y_rel1 - y_rel0)
            ]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def detectar_presencia(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    return cv2.countNonZero(th) > UMBRAL_AREA, th

def ocr_capacitor(roi_bgr):
    # Recortar ROI de texto relativo al capacitor
    tx, ty, tw, th_h = roi_texto
    texto_roi = roi_bgr[ty:ty+th_h, tx:tx+tw].copy()

    if rotar:
        texto_rotado = cv2.rotate(texto_roi, cv2.ROTATE_90_CLOCKWISE)
    else:
        texto_rotado = texto_roi

    texto_big = cv2.resize(
        texto_rotado, None,
        fx=ZOOM_FACTOR, fy=ZOOM_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(texto_big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    txt = pytesseract.image_to_string(
        th,
        config="--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
    )
    txt_norm = txt.strip().replace(" ", "")

    valor = None
    if "3000" in txt_norm:
        valor = 3000
    elif "4000" in txt_norm:
        valor = 4000

    print(f"Tesseract: '{txt_norm}' -> {valor}")

    cv2.imshow("texto_roi", texto_roi)
    cv2.imshow("texto_final", th)

    return valor, texto_roi, texto_big, th

def main():
    global rotar, mode_roi

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir")
        return
    # # Apagar autofocus (si el driver lo soporta)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # # Fijar un foco manual (0–255 aprox, depende de la cámara)
    # cap.set(cv2.CAP_PROP_FOCUS, 30)

    '''
    # Ver controles
    v4l2-ctl -d /dev/video0 --list-ctrls

    # Desactivar autofocus y fijar foco
    v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0
    v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=20
    '''
    cv2.namedWindow("frame")
    # se pasa un frame dummy para evitar errores de None
    cv2.setMouseCallback("frame", mouse_callback, [np.zeros((480, 640, 3), dtype=np.uint8)])

    ultimo_ocr = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # actualizar param del callback con el frame actual
        cv2.setMouseCallback("frame", mouse_callback, [frame])

        # ROI capacitor con límites
        x1, y1, w1, h1 = roi_capacitor
        h_f, w_f = frame.shape[:2]
        w1 = max(5, min(w1, w_f - 1))
        h1 = max(5, min(h1, h_f - 1))
        x1 = max(0, min(x1, w_f - w1))
        y1 = max(0, min(y1, h_f - h1))
        roi_capacitor[:] = [x1, y1, w1, h1]

        roi = frame[y1:y1+h1, x1:x1+w1]

        # Clamp ROI texto dentro del capacitor
        tx, ty, tw, th_h = roi_texto
        tw = max(5, min(tw, w1))
        th_h = max(5, min(th_h, h1))
        tx = max(0, min(tx, w1 - tw))
        ty = max(0, min(ty, h1 - th_h))
        roi_texto[:] = [tx, ty, tw, th_h]

        hay_objeto, th_roi = detectar_presencia(roi)
        tiempo_actual = time.time()

        if hay_objeto and (tiempo_actual - ultimo_ocr) >= OCR_INTERVALO:
            valor, _, _, _ = ocr_capacitor(roi)
            if valor:
                print(f"{'✅' if valor == LINE_VALUE else '❌'} {valor}")
            ultimo_ocr = tiempo_actual

        # DIBUJAR ROI CAPACITOR (verde)
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)

        # DIBUJAR ROI TEXTO (rojo, relativo)
        tx_abs = x1 + roi_texto[0]
        ty_abs = y1 + roi_texto[1]
        cv2.rectangle(frame, (tx_abs, ty_abs),
                      (tx_abs+roi_texto[2], ty_abs+roi_texto[3]),
                      (0, 0, 255), 2)

        info = f"ROI Cap: {x1},{y1},{w1}x{h1} | Texto: {roi_texto} | Rotar: {'ON' if rotar else 'OFF'}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Modo: {'Cap(C)' if mode_roi==0 else 'Texto(T)'} | R=Toggle Rotar | ESC=Salir",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("frame", frame)
        cv2.imshow("roi_bin", th_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key in (ord('c'), ord('C')):
            mode_roi = 0
            print("🔵 Modo: Seleccionar/Mover ROI CAPACITOR")
        elif key in (ord('t'), ord('T')):
            mode_roi = 1
            print("🔴 Modo: Seleccionar/Mover ROI TEXTO")
        elif key in (ord('r'), ord('R')):
            rotar = not rotar
            print(f"🔄 Rotación: {'ON' if rotar else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("ROI FINAL CAPACITOR:", roi_capacitor)
    print("ROI FINAL TEXTO (relativo):", roi_texto)
    print("ROTAR:", rotar)
    print("="*50)

if __name__ == "__main__":
    main()
