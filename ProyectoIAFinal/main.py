import redneuronal
import cargador
import numpy as np

def cargarDatos():
    
    train, val, test = cargador.cargarDatosReshape()
    #se requirio pasar a la lista por problemas con Mnist, de aca esta funcion
    train = list(train)
    val = list(val)
    test = list(test)
    return train, val, test

if __name__ == "__main__":
    datosEntrenamiento, datosValidacion, datosTest = cargarDatos()

    #Parametros de la red nuronal
    capas = [784, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50, 10]
    lambda1 = 5.0
    learningRate = 0.01
    epocas = 5
    numeroBatch = 10


    # Crea la red neuronal, test y validacion
    redO = redneuronal.Red(capas, costo=redneuronal.FuncionCosto)
    redO.inicializadorPesos2()
    datos = redO.iniciar(datosEntrenamiento, epocas, numeroBatch, learningRate, datosEvaluacion=datosTest)
    redO.salvar('data1.txt')
    datos1 = [x ** 2 for x in datos[0]]
    datos1 = np.array(datos1)
    datos2 = np.array(datos[2])
    redO.graficar(datos1,datos2, range(len(datos1)), range(len(datos2)))
    redO.salvar1('data2.txt',np.array_str(datos1),np.array_str(datos2))
    

    
