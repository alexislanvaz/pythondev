import cv2
import mediapipe as mp
import numpy as np
import time

# Inicialización de Mediapipe: face mesh y drawing utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Índices de Mediapipe que conforman cada ojo (según la topología de FaceMesh)
# Puedes ajustarlos si MediaPipe cambia la enumeración de los landmarks.
# Aquí se usan algunos puntos que envuelven la parte externa e interna del ojo.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

def calcular_distancia(a, b):
    """
    Calcula la distancia euclidiana entre dos puntos (x1, y1) y (x2, y2).
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def calcular_ear(landmarks, eye_indices):
    """
    Calcula la Relación de Aspecto del Ojo (EAR) dada la lista de landmarks (puntos) 
    y los índices que corresponden al ojo.
    
    EAR = (distancia entre los párpados verticales) / (2 * distancia horizontal)
    
    En este caso tomamos dos segmentos verticales y uno horizontal para robustez.
    """
    # Ojo formado por 6 puntos: eye_indices = [p0, p1, p2, p3, p4, p5]
    # Se asume la siguiente convención (ejemplo para el ojo izquierdo):
    #   - p1 y p5 para un par vertical
    #   - p2 y p4 para el otro par vertical
    #   - p0 y p3 para la distancia horizontal
    p0 = landmarks[eye_indices[0]]
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[2]]
    p3 = landmarks[eye_indices[3]]
    p4 = landmarks[eye_indices[4]]
    p5 = landmarks[eye_indices[5]]

    # Distancias verticales
    dist_v1 = calcular_distancia(p1, p5)
    dist_v2 = calcular_distancia(p2, p4)

    # Distancia horizontal
    dist_h = calcular_distancia(p0, p3)

    # EAR
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear

def main():
    # Parámetros configurables
    UMBRAL_EAR = 0.22           # Umbral para considerar el ojo "cerrado"
    FRAMES_PARA_SOMNOLENCIA = 15  # Cantidad de cuadros consecutivos con el ojo cerrado para detectar somnolencia

    contador_cerrados = 0  # Cuántos cuadros consecutivos llevan los ojos cerrados
    drowsy_detected = False
    
    # Inicializamos la cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    # Creamos la instancia de FaceMesh. Se pueden ajustar parámetros para mayor precisión o velocidad.
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Solo procesar 1 rostro
            refine_landmarks=True,  # Landmarks refinados para ojos y labios
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break
            
            # Convertimos la imagen a RGB (Mediapipe trabaja en RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = face_mesh.process(frame_rgb)

            # Si encontramos al menos una cara
            if resultado.multi_face_landmarks:
                for face_landmarks in resultado.multi_face_landmarks:
                    
                    # Extraemos los puntos (x, y) escalados al tamaño de la imagen
                    h, w, _ = frame.shape
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append((cx, cy))
                    
                    # Calculamos EAR para cada ojo
                    ear_izq = calcular_ear(landmarks, LEFT_EYE_INDICES)
                    ear_der = calcular_ear(landmarks, RIGHT_EYE_INDICES)
                    
                    # EAR promedio de ambos ojos
                    ear_promedio = (ear_izq + ear_der) / 2.0

                    # Verificamos si está por debajo del umbral
                    if ear_promedio < UMBRAL_EAR:
                        contador_cerrados += 1
                    else:
                        # Si se abren los ojos, reiniciamos el contador
                        contador_cerrados = 0
                        drowsy_detected = False
                    
                    # Verificamos si se supera el número de cuadros consecutivos
                    if contador_cerrados >= FRAMES_PARA_SOMNOLENCIA:
                        drowsy_detected = True
                    
                    # Mostramos en pantalla la detección
                    cv2.putText(frame, f"EAR: {ear_promedio:.3f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if drowsy_detected:
                        cv2.putText(frame, "SOMNOLENCIA DETECTADA!", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # (Opcional) Dibujamos la malla facial para referencia
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
            cv2.imshow("Deteccion de somnolencia", frame)
            
            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
