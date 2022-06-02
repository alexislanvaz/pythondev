# Importamos librerias Al
import torch 
import cv2
import numpy as np
import serial

model = torch.hub.load('yltralytics/yolovs', 'custom' ,path = 'C:/Users/santi/Desktop/Universidad/9 Semestre/Vision Python/ArduinoCon/Red.pt') 


# color verde
verded = np.array([46,80,80])
verdeu = np.array([80,220,220])


# Pierto serial 
com = serial.Serial("COM3", 9600, write_timeout= 10)
a= 'a'
c= 'c'
pos = ''

# realiza video captura
cap = cv2.VideoCapture(0)

# variables
contafot = 0
contacar = 0
marca = 0
flag1 =0
flag2 =0
# empeazamos 
while True:
    # realizamos lectura e frames
    ret,frame = cap.read()

    # creamos copia
    copia = frame.copy()

    # mostramos el numero de carros
    cv2.putText(frame, "Ocupacion: ", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0 ), 2)
    cv2.putText(frame, str(contacar), (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Carros", (240, 456), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mejoramos rendimiento
    contafot += 1
    if contafot % 3 != 0:
        continue

    # realizamos las dete4cciones
    detect = model (frame,size= 640)

    # Extraemos la info 
    info = detect.pandas().xyxy[0].to_dict(orient='records') # Predicciones

    # preguntamos si hay detecciones
    if len(info ) !=0:
        # creamos for
        for result in info:

            # confianza
            conf = result['confidence']
            # print(conf)

            if conf >=70:
                # clase
                cls = int(result['class'])
                # xi
                xi = int(result['xmin'])
                # yi
                yi = int(result['ymin'])
                # xf
                xf = int(result['xmax'])
                # yf
                yf = int(result['ymax'])

                # Dibujamos
                cv2.rectangle(frame,(xi,yi),(xf,yf)(0,0,255),2)

                #buscamos la marca del suelo
                # copia = cv2.cvtColor(copia,cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(copia, cv2.COLOR_BGR2HSV)

                # creamos mascara
                mask = cv2.inRange(hsv,verded,verdeu)

                # contornos
                contornos, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contornos = sorted(contornos,key=lambda  x: cv2.contourArea(x), reverse=True)

                # Detectamos la marca
                for ctn in contornos:
                    # Extraemos informacion de la marca
                    xiz, yiz,ancho, alto = cv2.boundingRect(ctn)
                    # xf e yf de la region de la marca
                    xfz,yfz = ancho + xiz, alto + yiz
                    # dibujamos un rectangulo alrededor de la marca
                    cv2.rectangle(frame,(xiz,yiz),(xfz,yfz),(0,255,0),2)

                    # extraemos el centro
                    cxm, cym = (xiz + xfz) // 2, (yiz+yfz) // 2
                    # dibujamos
                    cv2.circle(frame,(cxm,cym),2,(0,255,0),3)

                    # delimitamos zonas de interes
                    # entrada
                    linxe = cxm + 90

                    # salida
                    linxs = cxm - 70

                    # demarcamos zona en rojo
                    cv2.line(frame,(linxe,yiz),(linxe,yfz),(0,0,255),2)
                    cv2.line(frame,(linxs,yiz),(linxs,yfz),(0,0,255),2)
                    cv2.circle(frame, (20,20),15,(0,0,255),cv2.FILLED)

                    # si el carro esta en zona de entrada
                    if xi< linxe < xf and flag1 == 0 and flag2 == 0 or marca ==1:
                        print("entrad")
                        # activamos primer marca
                        flag1=1

                        # podemos hacer l oque queramos placas,etc
                        cv2.circle(frame,(20,20),15,(0,0,255),cv2.FILLED)
                        cv2.line(frame,(linxe,yiz),(linxe,yfz),(0,255,255),2)

                        # enviamos sema; y movemos sefvo
                        com.write(a.encode('ascii'))

                        marca = 1
                        # punto medio
                        if xi> linxs <xf and flag1==1:
