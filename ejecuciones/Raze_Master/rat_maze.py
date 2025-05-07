# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt  # Para graficar recompensas y resultados
import numpy as np  # Para trabajar con matrices y operaciones numéricas
import random  # Generación de valores aleatorios
import time  # Gestión de tiempo, como timestamps
import pickle  # Para guardar y cargar la tabla Q
import cv2  # Para la visualización en tiempo real del entorno
from PIL import Image, ImageEnhance  # Para manipular y redimensionar imágenes
from matplotlib import style  # Estilo para gráficos
style.use("ggplot")  # Aplicar estilo a las gráficas

# Parámetros del entorno
TAMANO = 10  # Tamaño del laberinto (10x10)
NUM_EPISODIOS = 15000 # Número de episodios para entrenamiento
PENA_MOV = -1  # Penalización por movimiento básico
PENA_MURO = -9999  # Penalización por chocar contra un muro
PENA_REVISITA = -100  # Penalización por volver a una celda visitada
RECOMP_QUE = 100  # Recompensa por alcanzar el queso
epsilon = 0.9  # Parámetro de exploración inicial (exploración vs explotación)
DECAY_EPS = 0.999  # Decaimiento de epsilon por episodio
MOSTRAR_CADA = 1000  # Intervalo de visualización de episodios

# Ubicaciones de muros y queso
UBICACION_MUROS = [
    (0, 2), (0, 4), (1, 3), (2, 1), (2, 2), (2, 6), (3, 5), (4, 0), (4, 3),
    (4, 8), (5, 1), (5, 7), (6, 2), (6, 4), (6, 6), (7, 0), (7, 4), (8, 7),
    (8, 9), (9, 2), (9, 6)
]
UBICACION_QUE = (TAMANO - 1, TAMANO - 3)  # Posición del queso

# Parámetros del algoritmo de aprendizaje
tabla_q_inicial = None  # Tabla Q inicial (cargar archivo si existe)
TASA_APRENDIZAJE = 0.1  # Tasa de aprendizaje (alpha)
DESCUENTO = 0.95  # Factor de descuento (gamma)

# Identificadores para elementos del entorno
RATA = 1
QUESO = 2
MURO = 3
HUELLA = 5

# Paleta de colores para visualización del entorno
colores = {
    RATA: (255, 255, 255),  # Rata en color blanco
    QUESO: (0, 128, 255),   # Queso en color naranja
    MURO: (43, 255, 0),    # Muro en color verde limón
    HUELLA: (255, 255, 255)  # Huella en color blanco
}

# Clase para representar la Rata (agente)
class Rata:
    def __init__(self, x, y):
        self.x = x  # Posición en X
        self.y = y  # Posición en Y
        self.celdas_visitadas = []  # Lista de celdas visitadas

    def __str__(self):
        return f"Posición actual: ({self.x}, {self.y})"

    def accion(self, accion):
        # Definir acción: 0 = arriba, 1 = derecha, 2 = abajo, 3 = izquierda
        if accion == 0:
            self.mover(x=0, y=-1)
        elif accion == 1:
            self.mover(x=1, y=0)
        elif accion == 2:
            self.mover(x=0, y=1)
        elif accion == 3:
            self.mover(x=-1, y=0)

    def mover(self, x, y):
        # Actualizar posición de la rata
        self.x += x
        self.y += y
        # Limitar movimiento dentro del laberinto
        self.x = max(0, min(self.x, TAMANO - 1))
        self.y = max(0, min(self.y, TAMANO - 1))
        self.celdas_visitadas.append((self.x, self.y))  # Registrar celda visitada

    def obtener_recompensa(self, x, y):
        # Calcular la recompensa basada en la celda actual
        if (x, y) == UBICACION_QUE:
            return RECOMP_QUE  # Recompensa máxima por alcanzar el queso
        elif (x, y) in UBICACION_MUROS:
            return PENA_MURO  # Penalización por chocar contra un muro
        elif (x, y) in self.celdas_visitadas:
            return PENA_REVISITA  # Penalización por revisitar celdas
        else:
            return PENA_MOV  # Penalización por moverse

    def reset(self):
        # Reiniciar posición y estado de la rata
        self.x = 0
        self.y = 0
        self.celdas_visitadas = []

# Crear instancia de la rata
rata = Rata(0, 0)

# Inicialización de la tabla Q
if tabla_q_inicial is None:
    # Crear tabla Q con valores iniciales aleatorios
    tabla_q = {
        (x, y): [np.random.uniform(-5, 0) for i in range(4)]
        for x in range(-TAMANO + 1, TAMANO)
        for y in range(-TAMANO + 1, TAMANO)
    }
else:
    # Cargar tabla Q desde archivo
    with open(tabla_q_inicial, "rb") as f:
        tabla_q = pickle.load(f)

# Entrenamiento con Q-learning
recompensas_episodios = []
start_time = time.time()  # Iniciar el temporizador

for ep in range(NUM_EPISODIOS):
    rata.reset()
    if ep % MOSTRAR_CADA == 0:
        print(f"Episodio #{ep}, Epsilon actual: {epsilon:.5f}")
        if len(recompensas_episodios) >= MOSTRAR_CADA:
            print(f"Recompensa promedio en los últimos {MOSTRAR_CADA} episodios: {np.mean(recompensas_episodios[-MOSTRAR_CADA:]):.2f}")
        else:
            print(f"Recompensa promedio aún no calculable (menos de {MOSTRAR_CADA} episodios registrados)")
        mostrar = True
    else:
        mostrar = False

    recompensa_episodio = 0
    for i in range(200):
        observacion_actual = (rata.x, rata.y)
        if np.random.random() > epsilon:
            accion = np.argmax(tabla_q[observacion_actual])  # Elegir acción óptima
        else:
            accion = np.random.randint(0, 4)  # Acción aleatoria (exploración)

        rata.accion(accion)
        recompensa = rata.obtener_recompensa(rata.x, rata.y)
        nueva_observacion = (rata.x, rata.y)
        max_q_futuro = np.max(tabla_q[nueva_observacion])
        q_actual = tabla_q[observacion_actual][accion]

        # Actualizar tabla Q con la fórmula Q-learning
        if recompensa == RECOMP_QUE:
            nueva_q = RECOMP_QUE
        else:
            nueva_q = (1 - TASA_APRENDIZAJE) * q_actual + TASA_APRENDIZAJE * (recompensa + DESCUENTO * max_q_futuro)
        tabla_q[observacion_actual][accion] = nueva_q

        if mostrar:
            entorno = np.zeros((TAMANO, TAMANO, 3), dtype=np.uint8)
            entorno[rata.x][rata.y] = colores[RATA]
            entorno[UBICACION_QUE[0]][UBICACION_QUE[1]] = colores[QUESO]
            for muro in UBICACION_MUROS:
                entorno[muro[0]][muro[1]] = colores[MURO]

            # Cálculo del tiempo transcurrido
            tiempo_transcurrido = int(time.time() - start_time)
            cv2.putText(entorno, f"Tiempo: {tiempo_transcurrido}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Redimensionar el entorno
            img = Image.fromarray(entorno, 'RGB')
            img = img.resize((500, 500), resample=Image.NEAREST)

            # Cargar y ajustar el tamaño de la imagen adicional
            imagen_adicional = cv2.imread('images/RAZE.webp')
            imagen_adicional = cv2.resize(imagen_adicional, (500, 500))

            # Combinar la imagen del laberinto y la imagen adicional
            entorno_con_imagen = np.hstack((np.array(img), imagen_adicional))

            # Mostrar la interfaz combinada
            cv2.imshow("SIMULACION DEL LABERINTO", entorno_con_imagen)

            if cv2.waitKey(100 if recompensa != RECOMP_QUE else 500) & 0xFF == ord('q'):
                break

        recompensa_episodio += recompensa
        if recompensa == RECOMP_QUE:
            break

    recompensas_episodios.append(recompensa_episodio)
    epsilon *= DECAY_EPS  # Decaer epsilon para más explotación

# Visualizar resultados
promedio_movil = np.convolve(recompensas_episodios, np.ones((MOSTRAR_CADA,)) / MOSTRAR_CADA, mode="valid")
plt.plot(promedio_movil)
plt.xlabel("Episodios")
plt.ylabel("Recompensa promedio")
plt.title("Evolución de las recompensas a lo largo del tiempo")
plt.show()
