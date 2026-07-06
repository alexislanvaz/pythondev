# Guía de Desarrollo e Introducción al Sistema CMMS (Manufactura)
**Propósito**: Documento de inducción y especificación de requerimientos de software para que cualquier agente de IA o desarrollador comprenda el flujo del negocio, los requisitos tecnológicos y comience el desarrollo utilizando **Python + Flask + Jinja2**.
**Enlace a la Base de Datos**: Consulta la estructura de tablas y procedimientos almacenados en [cmms_system_specification.md](file:///c:/Users/alexi/.gemini/antigravity/brain/208e3082-54a2-4077-b5aa-d1b91d34227c/cmms_system_specification.md).

---

## 1. Resumen General del Sistema

Este CMMS (Computerized Maintenance Management System) está diseñado para una sola planta de manufactura con un enfoque dinámico en la reducción de **Down Time (Tiempo Perdido)**.
La aplicación es una **Web App responsive** renderizada en el servidor por Flask/Jinja2, optimizada para dos entornos físicos principales:
1.  **Laptops/Desktops**: Para uso de Líderes de Línea y personal de Ingeniería.
2.  **Tablets Industriales**: De uso rudo para los Técnicos de Mantenimiento que se desplazan por las líneas de producción, equipadas con cámara para lectura de códigos QR y captura de fotos.

---

## 2. Roles de Usuario e Interfaces Requeridas

### A. Rol: Líder de Línea (Interfaz: Laptop / Tablet)
*   **Caso de Uso**: El líder detecta una falla en una estación (ej. estación de atornillado) y la reporta de inmediato para detener el cronómetro de la línea.
*   **Requerimientos del Formulario de Reporte**:
    *   Selección de Área, Línea y Estación afectada.
    *   Selección del **Síntoma de Falla** observable desde un menú desplegable (ej. "Robot no cicla", "PC no responde").
    *   Selección de la **Clasificación del Problema** (Calidad, Falta de Material, Operación, Problema de Partes, Otros).
    *   Selección de la Causa Inicial de las **4Ms** (*Machine, Manpower, Material, Method*).
    *   Especificación del **Impacto** (Línea Completa, Cuello de Botella, Estación Regular).
    *   Número de **Operadores Afectados** (detenidos) por el paro.
    *   Comentario abierto opcional.
*   **Comportamiento**: Al enviar, se genera una Orden de Trabajo (OT) correctiva en estado "Creada" y se notifica al equipo de técnicos de esa área.

### B. Rol: Técnico de Mantenimiento (Interfaz: Tablet con Cámara)
*   **Caso de Uso**: El técnico recibe la alerta de una nueva OT correctiva o una rutina preventiva asignada. Se desplaza físicamente al lugar con su tablet.
*   **Flujo de la Interfaz del Técnico**:
    1.  **Panel de Tareas**: Vista de OTs asignadas o pendientes por atender en su área/turno.
    2.  **Escaneo QR de Arribo**: Utilizando la cámara integrada y la librería open-source `html5-qrcode` en la tablet, el técnico escanea el código QR de la estación. El cliente hace un POST al servidor, registra la hora de arribo (`FechaArriboQR`) y cambia el estado de la OT a "En Proceso".
    3.  **Pantalla de Diagnóstico y Soporte**: Una vez escaneado el QR, Jinja2 renderiza o JS carga de forma dinámica:
        *   Los **Equipos** presentes en esa estación (ej. Robot de Atornillado y PC).
        *   **Fallas Comunes** y **Soluciones Sugeridas** asociadas al tipo de equipo.
        *   **Recursos de Soporte**: Enlaces rápidos para abrir manuales en PDF, guías de trabajo rápido, videos de calibración o imágenes instructivas directamente desde el servidor.
    4.  **Captura de Evidencias (Cámara)**: El técnico utiliza la cámara del dispositivo móvil para tomar fotos de la falla y de la solución. Estas fotos se suben al backend de Flask mediante formularios HTTP multipart.
    5.  **Registro de Materiales**: El técnico puede buscar números de parte y registrar las refacciones consumidas.
    6.  **Control de Pausas**: Si requiere esperar una refacción o terminar su turno, presiona "Pausar" indicando el motivo estandarizado. Al regresar, reescanea el QR para continuar.
    7.  **Cierre de OT**: El técnico asocia cuál **Equipo específico** falló (ej. el líder reportó falla en la estación, pero el técnico diagnostica que falló la PC y no el Robot), selecciona el código de **Falla Real Técnica**, toma la foto de la solución y cierra la OT.

### C. Rol: Ingeniería de Mantenimiento / Administración (Interfaz: Desktop)
*   **Caso de Uso**: Monitorear la salud de la planta, programar preventivos y analizar KPIs.
*   **Funcionalidades**:
    *   **Dashboard de Control**: Vista de la planta con código de colores.
    *   **Métricas de Desempeño**: Gráficas interactivas con el análisis de tiempos (MTTR Bruto/Neto, MTBF, Down Time, análisis de discrepancias).
    *   **Módulo Preventivo**: Visualización de la programación de preventivos en un calendario mensual interactivo.
    *   **Catálogos**: Pantallas CRUD para administración de activos, usuarios, fallas comunes y recursos de soporte.

---

## 3. Arquitectura Tecnológica Implementada

La aplicación utiliza un esquema híbrido donde Flask/Jinja2 renderizan la mayor parte de las interfaces en el servidor (SSR) y Javascript se encarga del manejo local del hardware de la tablet (cámara/QR) y de la interactividad de las librerías cliente.

```
[ FRONTEND CLIENTE ]
├── HTML5 / CSS Nativo / JavaScript (ES6)
└── Librerías Open Source:
    ├── html5-qrcode (Lectura QR mediante Cámara de Tablet)
    ├── Chart.js / ApexCharts (Gráficas de Down Time, MTTR, MTBF)
    ├── DataTables (Búsqueda, ordenamiento y paginación de tablas)
    └── FullCalendar (Calendario interactivo de preventivos programados)
      ▲
      │  (HTML renderizado / JSON en Fetch API)
      ▼
[ BACKEND SERVER ] (Python + Flask)
├── Jinja2 (Motor de plantillas para renderizar páginas)
├── SQLAlchemy o pyodbc (Conexión directa a SQL Server 2017)
└── File Storage Helper (Carga y lectura física de imágenes, manuales y videos)
      ▲
      │  (T-SQL / Stored Procedures)
      ▼
[ BASE DE DATOS ] (SQL Server 2017 14.0)
```

---

## 4. Librerías Open Source Seleccionadas para el Frontend

Para evitar el desarrollo de componentes complejos desde cero y acelerar la entrega del proyecto, se deben utilizar las siguientes librerías open-source integradas mediante CDN o archivos locales en el directorio `/static`:

1.  **Lectura de códigos QR**: [html5-qrcode](https://github.com/mebjas/html5-qrcode)
    *   *Uso*: Permite acceder a la cámara trasera de la tablet mediante `navigator.mediaDevices` del navegador web y decodificar códigos QR de forma rápida en JS Vanilla.
2.  **Gráficas e Indicadores (KPIs)**: [Chart.js](https://www.chartjs.org/) o [ApexCharts](https://apexcharts.com/)
    *   *Uso*: Renderiza en tiempo real las gráficas de barras para Down Time por 4Ms, gráficas de líneas de MTBF por equipo, e indicadores de cumplimiento de preventivos.
3.  **Tablas de Datos**: [DataTables.net](https://datatables.net/)
    *   *Uso*: Convierte tablas HTML estándar (`<table>`) de órdenes de trabajo, catálogo de equipos e insumos de almacén en tablas interactivas con paginación, filtros dinámicos instantáneos por texto y ordenamiento de columnas en el lado del cliente sin recargar la página.
4.  **Calendario Operativo**: [FullCalendar](https://fullcalendar.io/)
    *   *Uso*: Despliega el calendario mensual de mantenimiento preventivo, permitiendo al equipo de ingeniería visualizar visualmente qué días y estaciones están programados para paros preventivos.
5.  **Estilos CSS**: Uso de un diseño limpio, moderno e intuitivo (Dark Mode opcional o paleta corporativa moderna) adaptado para pantallas táctiles (botones grandes y zonas fáciles de presionar en tablets).

---

## 5. Estructura de Rutas y Controladores en Flask (Python)

El desarrollador que trabaje con Flask estructurará la aplicación web de la siguiente manera:

```python
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pyodbc # Conector nativo de SQL Server

app = Flask(__name__)

# Configuración de conexión a SQL Server 2017
CONN_STRING = "Driver={ODBC Driver 17 for SQL Server};Server=TU_SERVIDOR;Database=CMMS_Db;Uid=usuario;Pwd=password;"

def get_db_connection():
    return pyodbc.connect(CONN_STRING)

# ---------------------------------------------------------
# RUTAS DE INTERFAZ DE USUARIO (Renders Jinja2)
# ---------------------------------------------------------

@app.route('/')
def index():
    # Página principal (Dashboard general)
    return render_template('dashboard.html')

@app.route('/ot/crear', methods=['GET'])
def vista_crear_ot():
    # Formulario del líder para reportar falla
    return render_template('lider_crear_ot.html')

@app.route('/ot/tablet/panel')
def vista_tablet_tecnico():
    # Panel móvil para técnicos en la tablet
    return render_template('tecnico_panel.html')

@app.route('/ot/tablet/atencion/<int:ot_id>')
def vista_ot_atencion(ot_id):
    # Pantalla interactiva tras escanear el QR
    # Aquí Jinja2 pasa datos de la OT y los equipos de la estación
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM OrdenesTrabajo WHERE ID = ?", ot_id)
    ot = cursor.fetchone()
    conn.close()
    return render_template('tecnico_atencion.html', ot=ot)

@app.route('/preventivos/calendario')
def vista_calendario_pm():
    # Vista de calendario FullCalendar para planeación de preventivos
    return render_template('preventivo_calendario.html')


# ---------------------------------------------------------
# ENDPOINTS API (Peticiones dinámicas AJAX/Fetch en JS)
# ---------------------------------------------------------

@app.route('/api/ot/crear', methods=['POST'])
def api_crear_ot():
    # API que recibe JSON del líder y ejecuta sp_OT_Crear_Correctiva
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        EXEC sp_OT_Crear_Correctiva ?, ?, ?, ?, ?, ?, ?, ?
    """, (
        data['estacion_id'], data['lider_id'], data['sintoma_id'], 
        data['clasificacion_id'], data['causa_4m_id'], 
        data['impacto'], data['empleados_afectados'], data['comentario']
    ))
    nueva_ot = cursor.fetchone()
    conn.commit()
    conn.close()
    return jsonify({"success": True, "ot_id": nueva_ot[0]})

@app.route('/api/ot/arribo', methods=['POST'])
def api_registrar_arribo():
    # Técnico escanea QR en tablet y la API ejecuta sp_OT_RegistrarArriboQR
    data = request.json
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("EXEC sp_OT_RegistrarArriboQR ?, ?, ?", 
                       (data['ot_id'], data['tecnico_id'], data['codigo_qr']))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Arribo registrado."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/ot/pausar', methods=['POST'])
def api_pausar_ot():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("EXEC sp_OT_Pausar ?, ?, ?, ?", 
                   (data['ot_id'], data['motivo_id'], data['tecnico_id'], data['comentario']))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/api/ot/reanudar', methods=['POST'])
def api_reanudar_ot():
    data = request.json
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("EXEC sp_OT_Reanudar ?, ?, ?", 
                       (data['ot_id'], data['tecnico_id'], data['codigo_qr']))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/ot/completar', methods=['POST'])
def api_completar_ot():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("EXEC sp_OT_Completar_V2 ?, ?, ?, ?, ?", 
                   (data['ot_id'], data['equipo_id'], data['falla_real_id'], data['tecnico_id'], data['comentarios_solucion']))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/api/ot/<int:ot_id>/subir-evidencia', methods=['POST'])
def api_subir_evidencia(ot_id):
    # Recibe la foto enviada por la cámara de la tablet y la guarda físicamente
    tecnico_id = request.form['tecnico_id']
    tipo_evidencia = request.form['tipo'] # FALLA o SOLUCION
    archivo = request.files['foto']
    
    # Generar ruta física
    path_guardado = f"\\\\file-server\\evidencias\\ot_{ot_id}_{tipo_evidencia.lower()}.jpg"
    archivo.save(path_guardado)
    
    # Guardar en base de datos
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO EvidenciasOT (OrdenTrabajoID, UsuarioID, TipoEvidencia, PathArchivo)
        VALUES (?, ?, ?, ?)
    """, (ot_id, tecnico_id, tipo_evidencia, path_guardado))
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "path": path_guardado})

# ---------------------------------------------------------
# APIS PARA LIBRERIAS FRONTEND (DataTables, FullCalendar, Chart.js)
# ---------------------------------------------------------

@app.route('/api/kpi/downtime-data')
def api_kpi_downtime_data():
    # Retorna datos JSON listos para Chart.js
    # Consulta a base de datos de Down Time agrupada por 4Ms
    ...
    return jsonify(datos_grafica)

@app.route('/api/preventivos/eventos')
def api_preventivos_eventos():
    # Retorna JSON con formato de eventos para FullCalendar
    # [ {"title": "PM Celda SMT", "start": "2026-07-10"}, ... ]
    ...
    return jsonify(eventos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## 6. Criterios de Aceptación y Pruebas Críticas

1.  **Carga de Imágenes**: La funcionalidad de subida de fotos debe tolerar compresión en el lado del cliente (JS) antes del envío para reducir el ancho de banda del Wi-Fi de la planta.
2.  **Lectura del QR con html5-qrcode**: El sistema debe enfocar y decodificar correctamente etiquetas QR físicas de tamaño estándar (5x5 cm) a una distancia de hasta 30 cm bajo condiciones de iluminación industrial.
3.  **Visualización en Tablets**: Todas las pantallas de la interfaz del técnico deben ser navegables de forma táctil (inputs grandes, sin campos pequeños que requieran zoom).
4.  **Cálculo Exacto de Tiempos**: Si se reanuda una orden pausada, el temporizador de tiempo de reparación debe reanudarse y la base de datos debe descontar los intervalos de las pausas cerradas en la métrica final de MTTR.
