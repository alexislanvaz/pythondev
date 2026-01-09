import cv2
import pytesseract
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ===================== CONFIGURACIÓN =====================
CAM_INDEX = 0
LINE_VALUE = 4000
UMBRAL_AREA = 2000
ZOOM_FACTOR = 3
OCR_INTERVALO = 0.3

class CapacitorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Control de Calidad - Vista Dual")
        
        # Variables de estado original
        self.roi_capacitor = [90, 280, 170, 200]
        self.roi_texto = [20, 40, 102, 40]
        self.rotar = True
        self.mode_roi = 0 
        self.filtro_morf = 0 # Grosor del negro
        
        self.conteo_ok = 0
        self.conteo_error = 0
        self.ultimo_leido = "---"
        
        self.vid = cv2.VideoCapture(CAM_INDEX)
        self.last_ocr_time = 0
        self.dragging = False
        self.ix, self.iy = -1, -1
        self.img_ocr_display = np.zeros((150, 150), dtype=np.uint8) # Frame secundario inicial

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # --- SECCIÓN IZQUIERDA: VIDEO PRINCIPAL ---
        self.video_container = ttk.LabelFrame(self.main_frame, text=" Cámara en Vivo ")
        self.video_container.grid(row=0, column=0, padx=5, pady=5)
        
        self.canvas_main = tk.Canvas(self.video_container, width=640, height=480, bg="black")
        self.canvas_main.pack()
        self.canvas_main.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas_main.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas_main.bind("<ButtonRelease-1>", self.on_mouse_up)

        # --- SECCIÓN DERECHA: CONTROLES Y PREVIEW OCR ---
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky="ns", padx=5)

        # Preview del ROI (Lo que ve el OCR)
        self.preview_container = ttk.LabelFrame(self.right_panel, text=" Vista Pre-procesamiento (OCR) ")
        self.preview_container.pack(fill="x", pady=5)
        
        self.canvas_ocr = tk.Canvas(self.preview_container, width=250, height=180, bg="#2e2e2e")
        self.canvas_ocr.pack(padx=5, pady=5)

        # Estadísticas
        self.stats_frame = ttk.LabelFrame(self.right_panel, text=" Resultados ")
        self.stats_frame.pack(fill="x", pady=5)
        
        self.lbl_stats = tk.Label(self.stats_frame, text="OK: 0  |  ERR: 0", font=("Arial", 14, "bold"), fg="green")
        self.lbl_stats.pack(pady=5)
        self.lbl_read = tk.Label(self.stats_frame, text="VALOR: ---", font=("Arial", 12), fg="blue")
        self.lbl_read.pack(pady=5)

        # Controles de Grosor y Selección
        self.ctrl_frame = ttk.LabelFrame(self.right_panel, text=" Ajustes de Precisión ")
        self.ctrl_frame.pack(fill="x", pady=5)

        ttk.Label(self.ctrl_frame, text="Grosor Negro (Fuerza):").pack()
        self.scale_morf = ttk.Scale(self.ctrl_frame, from_=-5, to=5, orient="horizontal", command=self.update_morf)
        self.scale_morf.set(0)
        self.scale_morf.pack(fill="x", padx=10, pady=5)

        self.btn_cap = tk.Button(self.ctrl_frame, text="MODO: CAPACITOR", bg="#2c3e50", fg="white", command=lambda: self.set_mode(0))
        self.btn_cap.pack(fill="x", pady=2)
        self.btn_txt = tk.Button(self.ctrl_frame, text="MODO: TEXTO", bg="#2c3e50", fg="white", command=lambda: self.set_mode(1))
        self.btn_txt.pack(fill="x", pady=2)
        
        self.check_rot = ttk.Checkbutton(self.ctrl_frame, text="Rotar 90°", command=self.toggle_rotate)
        self.check_rot.pack(pady=5)

    def update_morf(self, val):
        self.filtro_morf = int(float(val))

    def set_mode(self, m):
        self.mode_roi = m
        self.btn_cap.config(bg="#27ae60" if m == 0 else "#2c3e50")
        self.btn_txt.config(bg="#c0392b" if m == 1 else "#2c3e50")

    def on_mouse_down(self, event):
        self.ix, self.iy = event.x, event.y
        self.dragging = True

    def on_mouse_move(self, event):
        if self.dragging:
            x, y = event.x, event.y
            if self.mode_roi == 0:
                self.roi_capacitor = [min(self.ix, x), min(self.iy, y), abs(self.ix-x), abs(self.iy-y)]
            else:
                cx, cy = self.roi_capacitor[0], self.roi_capacitor[1]
                self.roi_texto = [max(0, self.ix - cx), max(0, self.iy - cy), abs(self.ix-x), abs(self.iy-y)]

    def on_mouse_up(self, event):
        self.dragging = False

    def toggle_rotate(self):
        self.rotar = not self.rotar

    def preprocesar(self, img):
        """Ajuste de grosor morfológico"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        big = cv2.resize(gray, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(th) < 127: th = cv2.bitwise_not(th)
        
        # Ajuste de grosor (Morfología)
        kernel = np.ones((2,2), np.uint8)
        if self.filtro_morf > 0:
            th = cv2.erode(th, kernel, iterations=self.filtro_morf) # Engrosar negro
        elif self.filtro_morf < 0:
            th = cv2.dilate(th, kernel, iterations=abs(self.filtro_morf)) # Adelgazar negro
        return th

    def update_loop(self):
        ret, frame = self.vid.read()
        if ret:
            x, y, w, h = self.roi_capacitor
            if w > 10 and h > 10:
                roi_cap = frame[y:y+h, x:x+w]
                
                # Detección de presencia original
                gray_roi = cv2.cvtColor(roi_cap, cv2.COLOR_BGR2GRAY)
                _, th_p = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)
                
                if cv2.countNonZero(th_p) > UMBRAL_AREA:
                    t_actual = time.time()
                    if t_actual - self.last_ocr_time > OCR_INTERVALO:
                        threading.Thread(target=self.do_ocr, args=(roi_cap.copy(),), daemon=True).start()
                        self.last_ocr_time = t_actual

                # Dibujo en frame principal
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                tx, ty, tw, th_h = self.roi_texto
                cv2.rectangle(frame, (x+tx, y+ty), (x+tx+tw, y+ty+th_h), (0, 0, 255), 2)

            # Actualizar Canvas Principal
            self.display_in_canvas(frame, self.canvas_main)
            
            # Actualizar Canvas OCR (Preview)
            if hasattr(self, 'img_ocr_display'):
                self.display_in_canvas(self.img_ocr_display, self.canvas_ocr)

        self.window.after(10, self.update_loop)

    def display_in_canvas(self, img_cv, canvas):
        """Convierte y muestra imágenes de OpenCV en Canvas de Tkinter"""
        if len(img_cv.shape) == 2: # Si es escala de grises
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Ajustar al tamaño del canvas
        c_w, c_h = canvas.winfo_width(), canvas.winfo_height()
        if c_w > 1 and c_h > 1:
            img_cv = cv2.resize(img_cv, (c_w, c_h))
            
        img_pil = Image.fromarray(img_cv)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

    def do_ocr(self, roi_cap):
        tx, ty, tw, th = self.roi_texto
        h_m, w_m = roi_cap.shape[:2]
        crop = roi_cap[ty:min(ty+th, h_m), tx:min(tx+tw, w_m)]
        
        if crop.size > 0:
            if self.rotar: crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            img_prep = self.preprocesar(crop)
            self.img_ocr_display = img_prep # Para el canvas secundario
            
            # Configuración Tesseract
            config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
            txt = pytesseract.image_to_string(img_prep, config=config).strip().replace("\n", "")
            
            if txt:
                self.ultimo_leido = txt
                if str(LINE_VALUE) in txt:
                    self.conteo_ok += 1
                elif any(v in txt for v in ["3000", "4000"]):
                    self.conteo_error += 1
                
                self.lbl_stats.config(text=f"OK: {self.conteo_ok}  |  ERR: {self.conteo_error}")
                self.lbl_read.config(text=f"VALOR: {self.ultimo_leido}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("950x550")
    app = CapacitorApp(root)
    root.mainloop()