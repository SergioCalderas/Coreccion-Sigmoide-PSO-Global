# PSO CON ENFOQUE GLOBAL PARA LA MEJORA DE IMAGENES CON LA FUNCION SIGMOIDE
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# FUNCION PARA CALCULAR LA ENTROPIA DE UNA IMAGEN
def calcularEntropia(imagen):
    histograma, _ = np.histogram(imagen, bins=256, range=(0, 256), density=True)
    histograma = histograma[histograma > 0]  # Ignorar las probabilidades cero
    entropia = -np.sum(histograma * np.log2(histograma))
    return -entropia

# FUNCION PARA APLICAR LA SIGMOIDE A LA IMAGEN
def aplicarSigmoide(imagenOriginal, alpha, delta):
    imagenGrises = imagenOriginal.convert("L")  # Convertir a escala de grises
    datos = np.asarray(imagenGrises) / 255.0    # Convertir la imagen a un array y normalizar de 0 a 1

    sigmoide = 1 / (1 + np.exp(-alpha * (datos - delta)))  # Aplicar la funcion sigmoide
    imagen_transformada = np.uint8(sigmoide * 255) # Reescalar los pixeles de 0 a 255
    return imagen_transformada

# FUNCION OBJETIVO PARA CALCULAR LA ENTROPIA DE UNA IMAGEN
def calcularAptitud(imagenOriginal, variables):
    alpha = variables[0]  # Factor de contraste
    delta = variables[1]  # Punto medio de la curva

    # Aplicar la funcion sigmoide a la imagen
    imagen_transformada = aplicarSigmoide(imagenOriginal, alpha, delta)

    # Calcular la entropia de la imagen transformada
    aptitud = calcularEntropia(imagen_transformada) 
    return aptitud

# FUNCION PRINCIPAL DEL ALGORITMO PSO ENFOQUE GLOBAL
def PSO_GLOBAL(cumulo, xpBEST, xgBEST, maxIteracion, numIteracion):
    print("Iteracion Actual:", numIteracion - 1)
    #########################################################################################################################################################
    # FRENO DE LA RECURSION
    if maxIteracion == 0:
        print("\nIteracion:", numIteracion - 1) 
        
        aptitudes = [particula[2] for particula in cumulo] # Obtener las aptitudes del cumulo de la ultima iteracion
        mejorAptitud = aptitudes.index(min(aptitudes)) # Obtener el indice de la particula con la mejor aptitud (maximizar)
        minimaAptitud = cumulo[mejorAptitud][2]  # Obtener el valor de la aptitud de la particula seleccionada
        print("\nLa Mejor Particula es:", cumulo[mejorAptitud]) # Mostrar la mejor particula
        print(" Valor de Alpha:", cumulo[mejorAptitud][0][0], ", Valor de Umbral Delta:", cumulo[mejorAptitud][0][1])
        print(" Su Entropia es:", -minimaAptitud) # Mostrar la aptitud de la mejor particula

        # APLICAR SIGMOIDE CON LOS DATOS DE SALIDA
        imagen_transformada = aplicarSigmoide(imagenOriginal, cumulo[mejorAptitud][0][0], cumulo[mejorAptitud][0][1])
        
        # Convertir la imagen transformada a formato RGB
        imagen_transformada_rgb = Image.fromarray(imagen_transformada).convert("RGB")

        plt.imshow(imagen_transformada_rgb)
        plt.axis('off')  # Ocultar los ejes
        plt.show()

        return 

    #########################################################################################################################################################    
    # DETERMINAR xgBEST
    xgBEST = []
    aptitudes = [particula[2] for particula in cumulo] # Obtener las aptitudes del cumulo
    particulaMenorAptitud = aptitudes.index(min(aptitudes)) # Obtener el indice del individuo con la mayor aptitud
    xgBEST = cumulo[particulaMenorAptitud][0]

    #########################################################################################################################################################
    # ACTUALIZAR LA VELOCIDAD Y POSICION DE LAS PARTICULAS
    nuevoCumulo = []
    for i in range(numParticulas):
        nuevoArregloX = [] # Arreglo para guardar los nuevos valores de las variables
        nuevoArregloV = [] # Arreglo para guardar los nuevos valores de v
        for j in range(2):
            r1 = random.random()
            r2 = random.random()
            nuevaV = round(w * cumulo[i][1][j] + c1*r1*(xpBEST[i][j] - cumulo[i][0][j]) + c2*r2*(xgBEST[j] -  cumulo[i][0][j]), 2)
            nuevaX = round(cumulo[i][0][j] + nuevaV, 2)
            nuevoArregloX.append(nuevaX)
            nuevoArregloV.append(nuevaV)
        nuevoCumulo.append((nuevoArregloX, nuevoArregloV))

    #########################################################################################################################################################
    # RECTIFICAR LA VIOLACION DE RESTRICCIONES DE DOMINIO
    for i in range(numParticulas):
        for j in range(2):
            if nuevoCumulo[i][0][j] < li[j] or nuevoCumulo[i][0][j] > ls[j]: # Si los nuevos valores no estan dentro de los limites
                rand = random.random()
                nuevoValor = round(li[j] + rand * (ls[j] - li[j]), 2)
                d = ls[j] - li[j]
                nuevaVelocidad = round(-d + 2 * rand * d, 2)
                nuevoCumulo[i][0][j] = nuevoValor
                nuevoCumulo[i][1][j] = nuevaVelocidad

    #########################################################################################################################################################
    # EVALUACION DE LAS NUEVAS POSICIONES DE LAS PARTICULAS
    cumuloFinal = []
    for i in range(numParticulas):
        aptitud = round(calcularAptitud(imagenOriginal, nuevoCumulo[i][0]), 2) # Calcular la aptitud de la particula
        x1 = nuevoCumulo[i][0][0]
        x2 = nuevoCumulo[i][0][1]
        v1 = nuevoCumulo[i][1][0]
        v2 = nuevoCumulo[i][1][1]
        cumuloFinal.append(((x1, x2), (v1, v2), aptitud)) # Dar formato a la nueva particula

    #########################################################################################################################################################
    # ACTUALIZAR LAS MEJORES POSICIONES DE LAS PARTICULAS
    for i in range(numParticulas):
        if cumuloFinal[i][2] < cumulo[i][2]: # Si el valor de la nueva aptitud es mejor que el que se tenia
            xpBEST[i] = cumuloFinal[i][0]

    #########################################################################################################################################################
    # LLAMADA RECURSIVA DE LA FUNCION
    PSO_GLOBAL(cumuloFinal, xpBEST, xgBEST, maxIteracion - 1, numIteracion + 1)

#############################################################################################################################################################
# CARGAR LA IMAGEN ORIGINAL
ruta_imagen = "C:\\Users\\S ALBERT FC\\Documents\\ESCOM\\4° semestre\\Procesamiento Digital de Imagenes\\Imagenes\\antena512.jpg"
imagenOriginal = Image.open(ruta_imagen)

#############################################################################################################################################################
# PARAMETROS PARA LA FUNCION DEL ALGORITMO PSO GLOBAL
li = [0, 0]
ls = [10, 1]
w = 0.5
c1 = 2
c2 = 2
maxIteracion = 50 # Estos dos parametros se pueden modificar para que no sea tan tardado el tiempo de ejecucion
numParticulas = 200

#############################################################################################################################################################
# GENERAR CUMULO INICIAL
cumuloIncial = [] 
for i in range(numParticulas):
    arregloX = [] # Arreglo para guardar los valores de las variables
    arregloV = [] # Arreglo para guardar los valores de V
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
# PRIMER LLAMADA A LA FUNCION PSO_GLOBAL
xgBEST = []
numIteracion = 1
PSO_GLOBAL(cumulo, xpBEST, xgBEST, maxIteracion, numIteracion)

#############################################################################################################################################################
# MOSTRAR IMAGEN ORIGINAL Y RESULTADO
plt.imshow(imagenOriginal)
plt.title("Imagen Original")
plt.axis('off')  # Ocultar los ejes
plt.show()



