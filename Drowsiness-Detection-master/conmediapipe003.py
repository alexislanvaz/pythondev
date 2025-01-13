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
MOUTH_INDICES = [78, 308, 191, 95, 375, 17]

# Para la distancia aproximada, tomaremos los puntos externos de los ojos
# (ej. 33 para ojo izquierdo y 263 para el derecho) como referencia:
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

# ====================================
# Parámetros y constantes
# ====================================
UMBRAL_EAR = 0.22
FRAMES_CONSECUTIVOS_CERRADOS = 15  # Para detectar un parpadeo largo / microsueño
VENTANA_PROMEDIO_EAR = 5           # Suavizado para el EAR

UMBRAL_MAR = 0.60
FRAMES_CONSECUTIVOS_BOCA = 15
VENTANA_PROMEDIO_MAR = 5           # Suavizado para el MAR

# Parámetros para estimar la distancia
# (Ajustar según tu cámara y promedio real entre los ojos)
FOCAL_LENGTH_PX = 800.0     # Focal en pixeles (apróx. sin calibración)
REAL_EYE_DIST_CM = 6.3      # Distancia real entre ojos (apróx. 6.3 cm)

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
    Calcula la Relación de Aspecto del Ojo (EAR).
    
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
    Calcula la Relación de Aspecto de la Boca (MAR).
    
    MAR = (dist_v1 + dist_v2) / (2 * dist_h)
    """
    p0 = landmarks[mouth_indices[0]]
    p1 = landmarks[mouth_indices[1]]
    p2 = landmarks[mouth_indices[2]]
    p3 = landmarks[mouth_indices[3]]
    p4 = landmarks[mouth_indices[4]]
    p5 = landmarks[mouth_indices[5]]

    dist_v1 = calcular_distancia(p2, p4)
    dist_v2 = calcular_distancia(p3, p5)
    dist_h  = calcular_distancia(p0, p1)

    mar = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return mar

def calcular_distancia_aproximada(landmarks, left_idx, right_idx):
    """
    Estima la distancia (en cm) entre la cámara y el rostro,
    basándose en la distancia (en pixeles) entre dos puntos conocidos (por ejemplo,
    esquinas externas de los ojos) y el modelo de cámara estenopeica.
    
    D = (f * real_dist) / pixel_dist

    - f: longitud focal en pixeles (FOCAL_LENGTH_PX).
    - real_dist: distancia real en cm (REAL_EYE_DIST_CM).
    - pixel_dist: distancia entre los puntos left_idx y right_idx en pixeles.
    """
    # Extraer puntos en pixeles
    p_left = landmarks[left_idx]
    p_right = landmarks[right_idx]

    # Distancia entre los dos puntos en pixeles
    pixel_dist = calcular_distancia(p_left, p_right)
    
    # Evitar división entre 0
    if pixel_dist < 1e-6:
        return 0.0

    # Modelo pinhole
    distance_cm = (FOCAL_LENGTH_PX * REAL_EYE_DIST_CM) / pixel_dist
    return distance_cm

def main():
    """
    Detecta parpadeos, bostezos y calcula la distancia aproximada
    entre la cámara y el rostro.
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

                    # Suavizar EAR
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

                    # Detección de bostezo
                    if mar_suavizado > UMBRAL_MAR:
                        contador_boca_abierta += 1
                    else:
                        contador_boca_abierta = 0
                        bostezo_detectado = False

                    if contador_boca_abierta >= FRAMES_CONSECUTIVOS_BOCA:
                        if not bostezo_detectado:
                            bostezos_totales += 1
                            bostezo_detectado = True

                    # =============== DISTANCIA ===============
                    # Calcular distancia aproximada usando las esquinas externas de los ojos
                    dist_aprox = calcular_distancia_aproximada(
                        landmarks, LEFT_EYE_CORNER, RIGHT_EYE_CORNER
                    )

                    # =============== Visualización ===============
                    cv2.putText(frame, f"EAR: {ear_suavizado:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if drowsy_detected:
                        cv2.putText(frame, "SOMNOLENCIA DETECTADA!", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    cv2.putText(frame, f"MAR: {mar_suavizado:.2f}", (30, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cv2.putText(frame, f"Bostezos: {bostezos_totales}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # Mostramos la distancia aproximada
                    cv2.putText(frame, f"Dist (aprox): {dist_aprox:.2f} cm", (30, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    # (Opcional) Dibujar la malla facial de referencia
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

            cv2.imshow("Parpadeos, Bostezos y Distancia Aproximada", frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
