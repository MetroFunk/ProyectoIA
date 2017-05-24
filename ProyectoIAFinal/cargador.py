import pickle
import gzip
import numpy as np

def cargarDatos1():
    f = gzip.open('mnist.pkl.gz', 'rb')
    datosEntrenamiento, datosValidacion, datosTest = pickle.load(f, encoding='latin1')
    f.close()
    return (datosEntrenamiento, datosValidacion, datosTest)

#Se cargan los datos a los diferentes data sets
#es usado en la funcion principal para decidir a que hacer match
def cargarDatosReshape():
    datosT, datosV, datosTe = cargarDatos1()
    entradaTr = [np.reshape(x, (784, 1)) for x in datosT[0]]
    resultadosTr = [vectorizarResultados(y) for y in datosT[1]]
    datosEntrenamiento = zip(entradaTr, resultadosTr)
    entradasV = [np.reshape(x, (784, 1)) for x in datosV[0]]
    datosValidacion = zip(entradasV, datosV[1])
    entradasTe = [np.reshape(x, (784, 1)) for x in datosTe[0]]
    datosTest = zip(entradasTe, datosTe[1])
    return (datosEntrenamiento, datosValidacion, datosTest)


#Esto se usa para convertir un digito en un output aceptado en la red neuronal
def vectorizarResultados(x):
    e = np.zeros((10, 1))
    e[x] = 1.0
    return e
