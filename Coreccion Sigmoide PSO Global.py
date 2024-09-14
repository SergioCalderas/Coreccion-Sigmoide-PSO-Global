# PSO ENFOQUE GLOBAL PARA LA MEJORA DE IMAGENES CON LA FUNCION SIGMOIDE
import numpy as np
import random
from PIL import Image

# FUNCION PARA CALCULAR LA ENTROPIA DE UNA IMAGEN
def calcularEntropia(imagen):
    histograma, _ = np.histogram(imagen, bins=256, range=(0, 256), density=True)
    histograma = histograma[histograma > 0]  # Ignorar las probabilidades cero
    entropia = -np.sum(histograma * np.log2(histograma))
    return entropia

# FUNCION PARA APLICAR LA SIGMOIDE A LA IMAGEN
def aplicarSigmoide(imagenOriginal, alpha, delta):
    imagenGrises = imagenOriginal.convert("L")  # Convertir a escala de grises
    datos = np.asarray(imagenGrises) / 255.0    # Convertir la imagen a un array y normalizar de 0 a 1
    sigmoide = 1 / (1 + np.exp(-alpha * (delta -  datos)))  # Aplicar la funcion sigmoide
    imagen_transformada = np.uint8(sigmoide * 255) # Reescalar los pixeles de 0 a 255
    return imagen_transformada

# FUNCION OBJETIVO PARA CALCULAR LA ENTROPIA DE UNA IMAGEN SIGMOIDE
def calcularAptitud(imagenOriginal, variables):
    alpha = variables[0]  # Factor de contraste
    delta = variables[1]  # Punto medio de la curva
    imagenConSigmoide = aplicarSigmoide(imagenOriginal, alpha, delta) # Primero aplicar la funcion sigmoide a la imagen original
    aptitud = calcularEntropia(imagenConSigmoide) # Calcular la entropia de la imagen con sigmoide generada anteriormente
    return - aptitud # Se cambia el signo de la aptitud para que en lugar de minimizar se haga la maximizacion de la FO

#############################################################################################################################################################
# CARGAR LA IMAGEN ORIGINAL
ruta_imagen = "C:\\Users\\S ALBERT FC\\Documents\\ESCOM\\6° semestre\\Topicos Selectos\\Correccion de Imagenes\\kodim23.png"
imagenOriginal = Image.open(ruta_imagen)
print("La Entropia de la Imagen Original es:", round(calcularEntropia(imagenOriginal), 9))

#############################################################################################################################################################
# PARAMETROS PARA LA FUNCION DEL ALGORITMO PSO GLOBAL
li = [0, 0]
ls = [10, 1]
w = 0.5
c1 = 2
c2 = 2
maxIteracion = 50
numParticulas = 150

#############################################################################################################################################################
# GENERAR CUMULO INICIAL
cumuloIncial = [] 
for i in range(numParticulas):
    arregloX = []  # Arreglo para guardar los valores de las variables
    arregloV = []  # Arreglo para guardar los valores de
    for j in range(2):
        rand = random.random()
        x = round(li[j] + (rand) * (ls[j] - li[j]), 2)
        d = ls[j] - li[j]
        v = round(-d + 2 * rand * d, 2)
        arregloV.append(v)
        arregloX.append(x)
    cumuloIncial.append((arregloX, arregloV))

#############################################################################################################################################################    
# EVALUACION EN FO
cumulo = []
for i in range(numParticulas):
    aptitud = round(calcularAptitud(imagenOriginal, cumuloIncial[i][0]), 2)
    x1 = cumuloIncial[i][0][0]
    x2 = cumuloIncial[i][0][1]
    v1 = cumuloIncial[i][1][0]
    v2 = cumuloIncial[i][1][1]
    cumulo.append(((x1, x2), (v1, v2), aptitud))

#############################################################################################################################################################    
# ASIGNAR POSICIONES VISITADAS A xpBEST
xpBEST = []
for i in range(numParticulas):
    xpBEST.append(cumuloIncial[i][0])

#############################################################################################################################################################
# Inicializar el mejor valor global de aptitud y la mejor posición global
mejorAptitudGlobal = float('inf')
xgBEST = []

# CICLO PRINCIPAL DEL ALGORITMO PSO CON ENFOQUE GLOBAL
for iteracion in range(maxIteracion):
    aptitudes = [individuo[2] for individuo in cumulo]
    particulaMenorAptitud = aptitudes.index(min(aptitudes))
    minimaAptitud = cumulo[particulaMenorAptitud][2]
    print("La Iteracion Actual es:", iteracion,", Alpha:", cumulo[particulaMenorAptitud][0][0],", Delta:", cumulo[particulaMenorAptitud][0][1], ", Entropia:",
          cumulo[particulaMenorAptitud][2])

    #####################################################################################################################################################
    # DETERMINAR xgBEST
    aptitudes = [particula[2] for particula in cumulo]  # Obtener las aptitudes del cumulo
    particulaMenorAptitud = aptitudes.index(min(aptitudes))  # Obtener el índice del individuo con la menor aptitud
    aptitudActual = cumulo[particulaMenorAptitud][2]
    
    # Actualizar xgBEST solo si la aptitud de la iteracion actual es mejor que la mejor aptitud del xgBest
    if aptitudActual < mejorAptitudGlobal:
        # print("  Valor Anterior de xgBEST:", xgBEST, "aptitud anterior:", aptitudActual) # Mostrar xgBest anterior
        mejorAptitudGlobal = aptitudActual
        xgBEST = cumulo[particulaMenorAptitud][0]
        # print("  Nuevo mejor global encontrado:", xgBEST, "con aptitud:", mejorAptitudGlobal) # Mostrar el nuevo xgBest

    #####################################################################################################################################################
    # ACTUALIZAR LA VELOCIDAD Y POSICION DE LAS PARTICULAS
    nuevoCumulo = []
    for i in range(numParticulas):
        nuevoArregloX = []  # Arreglo para guardar los nuevos valores de las variables
        nuevoArregloV = []  # Arreglo para guardar los nuevos valores de v
        for j in range(2):
            r1 = random.random()
            r2 = random.random()
            nuevaV = round(w * cumulo[i][1][j] + c1 * r1 * (xpBEST[i][j] - cumulo[i][0][j]) + c2 * r2 * (xgBEST[j] - cumulo[i][0][j]), 2)
            nuevaX = round(cumulo[i][0][j] + nuevaV, 2)
            nuevoArregloX.append(nuevaX)
            nuevoArregloV.append(nuevaV)
        nuevoCumulo.append((nuevoArregloX, nuevoArregloV))

    #####################################################################################################################################################
    # RECTIFICAR LA VIOLACION DE RESTRICCIONES DE DOMINIO
    for i in range(numParticulas):
        for j in range(2):
            if nuevoCumulo[i][0][j] < li[j] or nuevoCumulo[i][0][j] > ls[j]:  # Si los nuevos valores no estan dentro de los limites
                rand = random.random()
                nuevoValor = round(li[j] + rand * (ls[j] - li[j]), 2)
                d = ls[j] - li[j]
                nuevaVelocidad = round(-d + 2 * rand * d, 2)
                nuevoCumulo[i][0][j] = nuevoValor
                nuevoCumulo[i][1][j] = nuevaVelocidad

    #####################################################################################################################################################
    # EVALUACION DE LAS NUEVAS POSICIONES DE LAS PARTICULAS
    cumuloFinal = []
    for i in range(numParticulas):
        aptitud = round(calcularAptitud(imagenOriginal, nuevoCumulo[i][0]), 2)  # Calcular la aptitud de la particula
        x1 = nuevoCumulo[i][0][0]
        x2 = nuevoCumulo[i][0][1]
        v1 = nuevoCumulo[i][1][0]
        v2 = nuevoCumulo[i][1][1]
        cumuloFinal.append(((x1, x2), (v1, v2), aptitud))  # Dar formato a la nueva particula

    #####################################################################################################################################################
    # ACTUALIZAR LAS MEJORES POSICIONES DE LAS PARTICULAS
    for i in range(numParticulas):
        if cumuloFinal[i][2] < cumulo[i][2]:  # Si el valor de la nueva aptitud es mejor que el que se tenia
            xpBEST[i] = cumuloFinal[i][0]
    
    # Actualiza el cumulo con el cumuloFinal
    cumulo = cumuloFinal

#############################################################################################################################################################
# MOSTRAR EL RESULTADO FINAL DE TODAS LAS ITERACIONES
aptitudes = [particula[2] for particula in cumulo]
menorAptitud = aptitudes.index(min(aptitudes))
minimaAptitud = cumulo[menorAptitud][2]
print("\nLa Mejor Particula es:", cumulo[menorAptitud])
print("Su aptitud es:", - minimaAptitud)  # Se cambia el signo para que el valor de salida de la entropia sea positivo