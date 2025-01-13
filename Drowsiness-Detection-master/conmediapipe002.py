import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ====================================
# Indices de MediaPipe FaceMesh
# ====================================
# OJOS (6 puntos para cada ojo)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

# BOCA (ejemplo de 6 puntos para calcular la "Mouth Aspect Ratio")
# Se seleccionan puntos alrededor del labio superior e inferior.
# Estos índices pueden variar según las definiciones de FaceMesh:
MOUTH_INDICES = [78, 308, 191, 95, 375, 17]  
# Ejemplo (p0, p1, p2, p3, p4, p5):
#  p0 - Extremo lateral izquierdo de la boca (aprox.)
#  p1 - Extremo lateral derecho de la boca (aprox.)
#  p2, p3, p4, p5 - Puntos intermedios arriba/abajo

# ====================================
# Parámetros y constantes
# ====================================
UMBRAL_EAR = 0.22
FRAMES_CONSECUTIVOS_CERRADOS = 15  # Para detectar un parpadeo largo o microsueño

# Para suavizar el EAR (evitar falsos positivos):
VENTANA_PROMEDIO_EAR = 5

# Umbrales y conteo para bostezos
UMBRAL_MAR = 0.78            # Umbral para considerar que la boca está abierta (bostezo)
FRAMES_CONSECUTIVOS_BOCA = 15  # Frames consecutivos para confirmar un bostezo
VENTANA_PROMEDIO_MAR = 5     # Suavizado de MAR

# ====================================
# Funciones de utilidad
# ====================================
def calcular_distancia(a, b):
    """
    Calcula la distancia euclidiana entre dos puntos 2D (x, y).
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def calcular_ear(landmarks, eye_indices):
    """
    Calcula la Relación de Aspecto del Ojo (EAR) dada la lista de (x, y)
    y los índices que conforman el ojo (6 puntos).
    
    EAR = (dist_v1 + dist_v2) / (2 * dist_h)
    """
    p0 = landmarks[eye_indices[0]]
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[2]]
    p3 = landmarks[eye_indices[3]]
    p4 = landmarks[eye_indices[4]]
    p5 = landmarks[eye_indices[5]]

    dist_v1 = calcular_distancia(p1, p5)
    dist_v2 = calcular_distancia(p2, p4)
    dist_h  = calcular_distancia(p0, p3)

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear

def calcular_mar(landmarks, mouth_indices):
    """
    Calcula la Relación de Aspecto de la Boca (MAR) a partir de 6 puntos.
    
    MAR = (dist_v1 + dist_v2) / (2 * dist_h)
    
    Donde:
      - dist_v1, dist_v2: distancias verticales (ej. top-lip a bottom-lip)
      - dist_h: distancia horizontal (labio izquierdo a derecho)
    """
    p0 = landmarks[mouth_indices[0]]
    p1 = landmarks[mouth_indices[1]]
    p2 = landmarks[mouth_indices[2]]
    p3 = landmarks[mouth_indices[3]]
    p4 = landmarks[mouth_indices[4]]
    p5 = landmarks[mouth_indices[5]]

    # Distancias verticales (ej. p2-p4, p3-p5)
    dist_v1 = calcular_distancia(p2, p4)
    dist_v2 = calcular_distancia(p3, p5)

    # Distancia horizontal (p0-p1)
    dist_h = calcular_distancia(p0, p1)

    mar = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return mar

def main():
    """
    Detecta parpadeos y bostezos mediante EAR y MAR usando OpenCV + Mediapipe.
    Lleva un contador de bostezos realizados.
    """
    # Variables para el conteo de parpadeos/microsueños
    contador_cerrados = 0
    drowsy_detected = False

    # Variables para el conteo de bostezos
    contador_boca_abierta = 0
    bostezos_totales = 0
    bostezo_detectado = False

    # Para suavizar el EAR y el MAR
    ear_history = deque(maxlen=VENTANA_PROMEDIO_EAR)
    mar_history = deque(maxlen=VENTANA_PROMEDIO_MAR)

    # Inicialización de captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    # Inicialización de Mediapipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Configuración de FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer un frame válido de la cámara.")
                break

            # Convertir a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(frame_rgb)

            if resultados.multi_face_landmarks:
                for face_landmarks in resultados.multi_face_landmarks:
                    h, w, _ = frame.shape
                    # Extraer landmarks (x, y) en pixeles
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append((cx, cy))

                    # =============== OJOS: EAR ===============
                    ear_izq = calcular_ear(landmarks, LEFT_EYE_INDICES)
                    ear_der = calcular_ear(landmarks, RIGHT_EYE_INDICES)
                    ear_promedio = (ear_izq + ear_der) / 2.0
                    
                    # Suavizamos la medida con promedio móvil
                    ear_history.append(ear_promedio)
                    ear_suavizado = np.mean(ear_history)

                    # Detección de parpadeo prolongado / microsueño
                    if ear_suavizado < UMBRAL_EAR:
                        contador_cerrados += 1
                    else:
                        contador_cerrados = 0
                        drowsy_detected = False
                    
                    if contador_cerrados >= FRAMES_CONSECUTIVOS_CERRADOS:
                        drowsy_detected = True
                    
                    # =============== BOCA: MAR ===============
                    mar = calcular_mar(landmarks, MOUTH_INDICES)
                    mar_history.append(mar)
                    mar_suavizado = np.mean(mar_history)

                    # Detección de bostezo (boca muy abierta varios frames)
                    if mar_suavizado > UMBRAL_MAR:
                        contador_boca_abierta += 1
                    else:
                        # Si la boca se cierra, reseteamos el conteo
                        contador_boca_abierta = 0
                        bostezo_detectado = False

                    # Si se sobrepasan N frames con la boca abierta
                    if contador_boca_abierta >= FRAMES_CONSECUTIVOS_BOCA:
                        # Incrementar contador de bostezos sólo si no estábamos ya en "bostezo_detectado"
                        if not bostezo_detectado:
                            bostezos_totales += 1
                            bostezo_detectado = True

                    # =============== Visualización ===============
                    # Muestra el EAR y si hay somnolencia
                    cv2.putText(frame, f"EAR: {ear_suavizado:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if drowsy_detected:
                        cv2.putText(frame, "SOMNOLENCIA DETECTADA!", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Muestra el MAR y el contador de bostezos
                    cv2.putText(frame, f"MAR: {mar_suavizado:.2f}", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cv2.putText(frame, f"Bostezos: {bostezos_totales}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # (Opcional) dibujar malla facial de referencia
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

            cv2.imshow("Deteccion de Parpadeos y Bostezos", frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
