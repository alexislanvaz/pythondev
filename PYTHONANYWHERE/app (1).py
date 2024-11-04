
from flask import Flask, request, jsonify,render_template
import mysql.connector
from functools import wraps
from dataclasses import dataclass
from typing import List

app = Flask(__name__)

API_TOKEN = r'*ECF_DxD#@$PC4-_-3xt'

@dataclass
class Option:
    id: str
    text: str
    value: str

@dataclass
class Question:
    id: int
    text: str
    type: str  # 'radio' or 'checkbox'
    options: List[Option]
    correct_answers: List[str]

# Decorador para validar el token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')  # Obtener el token de los headers
        if not token or token != API_TOKEN:
            return jsonify({"error": "Token no válido o ausente"}), 403  # Si el token no es válido o no existe
        return f(*args, **kwargs)
    return decorated

# Configuración de conexión a MySQL
def get_db_connection():
    connection = mysql.connector.connect(
        host='alanda26.mysql.pythonanywhere-services.com',
        user='alanda26',
        password='DXProject',
        database='alanda26$DXProject'
    )
    return connection

def clean_data():
    try:
        # Conectar a la base de datos
        connection = get_db_connection()
        cursor = connection.cursor()

        # Truncate la tabla (limpia los datos y reinicia los valores de AUTO_INCREMENT)
        query = "TRUNCATE TABLE Registros_Transporte_x_Dia;"

        # Ejecutar el TRUNCATE
        cursor.execute(query)
        connection.commit()

        # Cerrar cursor y conexión
        cursor.close()
        connection.close()

        print("Tabla limpiada y el ID reiniciado.")

    except mysql.connector.Error as err:
        print(f"Error al limpiar la tabla: {err}")


# Consulta para obtener la cantidad de usuarios por día
def get_transport_data():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """-- OBTIENE LAS CANTIDADES DE REGISTROS PARA TRANSPORTE Y EL NOMBRE DE LA RUTA
            SELECT rtd.Fecha, rtd.Hora, rt.NombreRuta, rtd.CantUsuarios
            FROM Registros_Transporte_x_Dia rtd
            JOIN Rutas_Transportes rt ON rtd.idRuta = rt.idRuta
            WHERE rtd.Activo = 1;
            """

    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    connection.close()

    return data

# Ruta para la página principal
@app.route('/')
def index():
    data = get_transport_data()
    return render_template('index.html', data=data)


@app.route('/insert', methods=['POST'])
@token_required  # Requiere un token válido
def insert_data():
    clean_data()
    # Obtener los datos del request (asumiendo que se envían en formato JSON)
    # Esperamos una lista de objetos con Fecha, idRuta y CantUsuarios
    data = request.get_json()

    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Datos no válidos o vacíos"}), 400

    # Crear una lista de valores para insertar múltiples filas
    values = []
    for entry in data:
        fecha = entry.get('Fecha')
        hora = entry.get('Hora')
        idRuta = entry.get('idRuta')
        cantUsuarios = entry.get('CantUsuarios')

        # Validar que los datos existan y sean correctos
        if not fecha or not hora or not idRuta or not cantUsuarios:
            return jsonify({"error": "Faltan datos en una de las entradas"}), 400

        if not isinstance(idRuta, int) or not isinstance(cantUsuarios, int):
            return jsonify({"error": "idRuta y CantUsuarios deben ser enteros"}), 400

        # Añadir los valores a la lista
        values.append((fecha, hora, idRuta, cantUsuarios))

    try:
        # Conectar a la base de datos
        connection = get_db_connection()
        cursor = connection.cursor()

        # Preparar la consulta SQL para múltiples inserciones
        query = """
            INSERT INTO Registros_Transporte_x_Dia (Fecha, Hora, idRuta, CantUsuarios, Activo, FechaRegistro)
            VALUES (%s, %s, %s, %s, 1, NOW())
        """

        # Ejecutar el insert de múltiples filas
        cursor.executemany(query, values)
        connection.commit()  # Asegurarse de guardar los cambios

        # Cerrar la conexión
        cursor.close()
        connection.close()

        return jsonify({"message": "Datos insertados correctamente"}), 201

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return jsonify({"error": "Error al insertar los datos"}), 500


# ----------------------------- QM 2024 -------------------------------------
def get_questions_from_db(idtest):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            qp.idpregunta,
            qp.pregunta,
            qp.tipopregunta,
            qr.idrespuesta,
            qr.respuesta,
            qr.escorrecta
        FROM
            QM_Preguntas qp
        JOIN
            QM_Respuestas qr ON qp.idpregunta = qr.idpregunta
        WHERE
            qp.activo = 1 AND qr.activa = 1 AND qp.idtest = %s
        ORDER BY
            qp.idpregunta, qr.idrespuesta;
    """, (idtest,))

    questions_dict = {}
    for row in cursor.fetchall():
        question_id = row['idpregunta']
        if question_id not in questions_dict:
            questions_dict[question_id] = {
                'id': question_id,
                'text': row['pregunta'],
                'type': row['tipopregunta'],
                'options': [],
                'correct_answers': []
            }

        option = Option(
            id=f"q{question_id}{row['idrespuesta']}",
            text=row['respuesta'],
            value=row['idrespuesta']
        )
        questions_dict[question_id]['options'].append(option)

        if row['escorrecta']:
            questions_dict[question_id]['correct_answers'].append(str(row['idrespuesta']))

    # Convertir a una lista de objetos Question
    questions = [
        Question(
            id=details['id'],
            text=details['text'],
            type=details['type'],
            options=details['options'],
            correct_answers=details['correct_answers']
        ) for details in questions_dict.values()
    ]

    cursor.close()
    connection.close()
    return questions


# Ruta de Flask para mostrar las preguntas
@app.route('/qm/actividades')
def qm_actividades_index():
    questions = get_questions_from_db(1)
    return render_template('qm_examen_entrada.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        employee_number = data.get('employeeNumber')
        responses = data.get('responses')

        if not employee_number or not str(employee_number).isdigit():
            return jsonify({'error': 'Número de empleado inválido o no proporcionado'}), 400

        # Supongamos que el idtest se envía como parte de la solicitud o se obtiene de otra manera
        idtest = data.get('idtest', 1)  # Por defecto, usa idtest=1 si no se proporciona otro

        # Cargar las preguntas activas desde la base de datos para el idtest dado
        questions = get_questions_from_db(idtest)

        if not questions:
            return jsonify({'error': 'No se encontraron preguntas para el test especificado.'}), 404

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Verificar si el empleado ya ha respondido al test
        cursor.execute("""
            SELECT COUNT(*) AS count
            FROM QM_RespuestasUsuarios
            WHERE numero_empleado = %s AND idtest = %s;
        """, (employee_number, idtest))
        result = cursor.fetchone()

        if result['count'] > 0:
            return jsonify({'error': 'Ya existe una respuesta registrada para este empleado en este test.'}), 400

        # Calcular el puntaje
        score = 0
        for question in questions:
            user_answers = next((item['answers'] for item in responses if item['questionId'] == f'q{question.id}'), [])
            if not isinstance(user_answers, list):
                user_answers = [user_answers]

            if sorted(user_answers) == sorted(question.correct_answers):
                score += 1

            # Guardar cada respuesta en QM_RespuestasUsuarios
            for answer in user_answers:
                cursor.execute("""
                    INSERT INTO QM_RespuestasUsuarios (idpregunta, idtest, numero_empleado, respuesta, fecharegistro)
                    VALUES (%s, %s, %s, %s, NOW());
                """, (question.id, idtest, employee_number, answer))

        # Guardar la calificación en QM_Calificaciones
        cursor.execute("""
            INSERT INTO QM_Calificaciones (idtest, calificacion, comentario, activo, fecharegistro)
            VALUES (%s, %s, %s, 1, NOW());
        """, (idtest, score, f'Resultado de la calificación: {score}/{len(questions)}'))

        # Confirmar los cambios
        connection.commit()

        percentage = (score / len(questions)) * 100
        return jsonify({
            'employeeNumber': employee_number,
            'score': score,
            'total': len(questions),
            'percentage': percentage
        })

    except mysql.connector.Error as db_err:
        # Manejo de errores de la base de datos
        return jsonify({'error': f'Error en la base de datos: {str(db_err)}'}), 500

    except Exception as e:
        # Manejo de otros errores generales
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

    finally:
        # Asegurar que el cursor y la conexión se cierren adecuadamente
        try:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        except NameError:
            # Si cursor o connection no fueron definidos, no hacer nada
            pass




# Endpoint para insertar datos
# @app.route('/insert', methods=['POST'])
# def insecdccscdsrt_data():
#     # Obtener los datos del request (asumiendo que se envían en formato JSON)
#     data = request.get_json()

#     fecha = data.get('Fecha')
#     idRuta = data.get('idRuta')
#     cantUsuarios = data.get('CantUsuarios')

#     if not fecha or not idRuta or not cantUsuarios:
#         return jsonify({"error": "Faltan datos"}), 400

#     try:
#         # Conectar a la base de datos
#         connection = get_db_connection()
#         cursor = connection.cursor()

#         # Preparar la consulta SQL
#         query = """
#             INSERT INTO Registros_Transporte_x_Dia (Fecha, idRuta, CantUsuarios, Activo, FechaRegistro)
#             VALUES (%s, %s, %s, 1, NOW())
#         """
#         values = (fecha, idRuta, cantUsuarios)

#         # Ejecutar la consulta
#         cursor.execute(query, values)
#         connection.commit()  # Asegurarse de guardar los cambios

#         # Cerrar la conexión
#         cursor.close()
#         connection.close()

#         return jsonify({"message": "Datos insertados correctamente"}), 201

#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#         return jsonify({"error": "Error al insertar los datos"}), 500


if __name__ == "__main__":
    app.run(debug=True)
