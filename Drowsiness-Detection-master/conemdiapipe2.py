import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ===========================
# Configuración y Constantes
# ===========================

# Índices (en la topología de MediaPipe FaceMesh) que envuelven cada ojo:
# Se pueden ajustar si MediaPipe cambia la numeración.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

# Parámetros para la detección
UMBRAL_EAR = 0.22
FRAMES_CONSECUTIVOS_CERRADOS = 15  # Cuadros consecutivos para marcar ojos cerrados prolongados
VENTANA_PROMEDIO = 5               # Tamaño de la ventana para el promedio móvil de EAR

# ===========================
# Funciones de utilidad
# ===========================

def calcular_distancia(punto_a, punto_b):
    """
    Calcula la distancia euclidiana entre dos puntos 2D.
    """
    return np.linalg.norm(np.array(punto_a) - np.array(punto_b))

def calcular_ear(landmarks, eye_indices):
    """
    Calcula la Relación de Aspecto del Ojo (EAR) dada la lista de (x, y)
    y los índices específicos del ojo (6 puntos).
    
    EAR = (dist_v1 + dist_v2) / (2 * dist_h)

    Donde:
      - dist_v1 es la distancia entre (p1, p5)
      - dist_v2 es la distancia entre (p2, p4)
      - dist_h  es la distancia entre (p0, p3)
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

def main():
    """
    Punto de entrada principal. Configura la cámara, inicializa
    MediaPipe FaceMesh y detecta parpadeos prolongados (somnolencia).
    """
    # ===========================
    # Inicialización de variables
    # ===========================
    contador_cerrados = 0  # Cuadros consecutivos con ojos cerrados
    drowsy_detected = False

    # Para suavizar el EAR y evitar falsos positivos, usaremos una cola (deque):
    ear_history = deque(maxlen=VENTANA_PROMEDIO)

    # ===================================
    # Inicialización de cámara y FaceMesh
    # ===================================
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("No se pudo acceder a la cámara.")
    except Exception as e:
        print(f"Error al abrir la cámara: {e}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Creamos el FaceMesh con parámetros óptimos
    # max_num_faces=1 => solo queremos detectar un rostro
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer un frame válido de la cámara.")
                    break

                # Convertimos a RGB para procesar con MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = face_mesh.process(frame_rgb)

                if resultado.multi_face_landmarks:
                    for face_landmarks in resultado.multi_face_landmarks:
                        # Extraer puntos (x, y) escalados al tamaño de la imagen
                        h, w, _ = frame.shape
                        landmarks = []
                        for lm in face_landmarks.landmark:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks.append((cx, cy))
                        
                        # Calculamos EAR para ambos ojos
                        ear_izq = calcular_ear(landmarks, LEFT_EYE_INDICES)
                        ear_der = calcular_ear(landmarks, RIGHT_EYE_INDICES)
                        ear_promedio = (ear_izq + ear_der) / 2.0

                        # Agregamos la medición a la cola para suavizar
                        ear_history.append(ear_promedio)
                        # Usamos el promedio móvil de la cola
                        ear_suavizado = np.mean(ear_history)

                        # Comprobamos si el EAR está por debajo del umbral
                        if ear_suavizado < UMBRAL_EAR:
                            contador_cerrados += 1
                        else:
                            contador_cerrados = 0
                            drowsy_detected = False

                        # Verificamos si se superan los frames mínimos cerrados
                        if contador_cerrados >= FRAMES_CONSECUTIVOS_CERRADOS:
                            drowsy_detected = True

                        # Mostrar en pantalla valores y estado
                        cv2.putText(frame, f"EAR: {ear_suavizado:.3f}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if drowsy_detected:
                            cv2.putText(frame, "SOMNOLENCIA DETECTADA!", (30, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        # (Opcional) dibujar la malla para referencia
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                # Mostrar la imagen procesada
                cv2.imshow("Deteccion de Somnolencia", frame)

                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error procesando frame: {e}")
                break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()