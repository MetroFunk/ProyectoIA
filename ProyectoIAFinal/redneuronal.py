import random
import numpy as np
import matplotlib.pyplot as plt

#Clase de la funcion de costo 
class FuncionCosto(object):
    #funcion de costo formula 
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    #Calculo del Delta(back)
    @staticmethod
    def delta(z, a, y):
        return (a-y)
#Es para  pasar de regero de numeros a 0-9
def vectorizarResultados(numeroEpoca):
    e = np.zeros((10, 1))
    e[numeroEpoca] = 1.0
    return e
#Funciones de Activacion
def sigmoide(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoideGradiente(z):
    return sigmoide(z)*(1-sigmoide(z))


datosSalvar =""

#Clase principal Red neuronal
class Red(object):
    #recibe tamahno de la red y la funcion de costo
    def __init__(self, tamanhos, costo=FuncionCosto):
        self.num_capas = len(tamanhos)
        self.tamanhos = tamanhos
        self.inicializadorPesos2()
        self.costo=costo

    #incializa los pesos de manera aleatoria
    def inicializadorPesos2(self):
        self.biases = [np.random.randn(y, 1) for y in self.tamanhos[1:]]
        self.pesos = [np.random.randn(y, x)
                        for x, y in zip(self.tamanhos[:-1], self.tamanhos[1:])]
    #Regularizacion de la funcion de costo
    def costoT(self, data, lambda1, convert=False):
        costo = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorizarResultados(y)
            costo += self.costo.fn(a, y)/len(data)
        costo += 0.5*(lambda1/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.pesos)
        return costo
    #Devuelve la precicion de los datos de entrenamiento
    def precision(self, data):
        resultados = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in resultados)
    def precision2(self, data):
        resultados = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in resultados)
    #incia el feedforward 
    def feedforward(self, k):
        for b, w in zip(self.biases, self.pesos):
            k = sigmoide(np.dot(w, k)+b)
        return k
    #funcion de entreamiento, recibe los datos, la cantidad de epochs,
    #el tamahno del minibatch, el learningrate, valor del lambda y los datos.
    def iniciar(self, datosEntrenamiento, epocas, mini_batch_tamanho, learningRate,
            lambda1 = 0.0,datosEvaluacion=None):
        if datosEvaluacion: numeroDatos = len(datosEvaluacion)
        n = len(datosEntrenamiento)
        evaluacionCosto, evaluacionPrecision = [], []
        costoEntrenamiento, precisionEntrenamiento = [], []
        global datosSalvar
        global plotArray
        for numeroEpoca in range(epocas):
            random.shuffle(datosEntrenamiento)
            mini_batches = [
                datosEntrenamiento[k:k+mini_batch_tamanho]
                for k in range(0, n, mini_batch_tamanho)]
            for mini_batch in mini_batches:
                self.actualizarMiniBatch(
                    mini_batch, learningRate, lambda1, len(datosEntrenamiento))
            #Inserts al script para insertar a la base de datos en la nube
            #ademas de imprimir los resultados en consola para control
            print("Epoca {}".format(numeroEpoca))
            datosSalvar =datosSalvar +"INSERT INTO Prueba1 VALUES ({},".format(numeroEpoca)
            precision = self.precision2(datosEntrenamiento)
            precisionEntrenamiento.append(precision)
            tPrecision = precision/ n
            print ("precision en datos entrenamiento: {}".format(tPrecision))
            datosSalvar = datosSalvar+"{},".format(tPrecision)
            costo = self.costoT(datosEntrenamiento, lambda1)
            costoEntrenamiento.append(costo)
            print ("costo en datos entrenamiento: {}".format(costo))
            datosSalvar = datosSalvar+"{},".format(costo)  
            precision = self.precision(datosEvaluacion)
            evaluacionPrecision.append(precision)
            ePrecision = self.precision(datosEvaluacion)/ numeroDatos
            print ("precision en datos de evaluacion: {}".format(ePrecision) )       
            datosSalvar = datosSalvar+"{},".format(ePrecision)
            costo = self.costoT(datosEvaluacion, lambda1, convert=True)
            evaluacionCosto.append(costo)
            print ("costo en datos evaluacion: {}".format(costo))
            datosSalvar =datosSalvar+"{});\n".format(costo) 
        return evaluacionCosto, evaluacionPrecision,costoEntrenamiento, precisionEntrenamiento

    
    def actualizarMiniBatch(self, mini_batch, learningRate, lambda1, n):
        holderW = [np.zeros(w.shape) for w in self.pesos]
        holderB = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            deltaholderB, deltaholderW = self.clasificar(x, y)
            holderW = [nw+dnw for nw, dnw in zip(holderW, deltaholderW)]
            holderB = [nb+dnb for nb, dnb in zip(holderB, deltaholderB)]
        self.pesos = [(1-learningRate*(lambda1/n))*w-(learningRate/len(mini_batch))*nw
                        for w, nw in zip(self.pesos, holderW)]
        self.biases = [b-(learningRate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, holderB)]
    #clasifica las imagenes dependiendo del error 
    def clasificar(self, x, y):
        holderW = [np.zeros(w.shape) for w in self.pesos]
        holderB = [np.zeros(b.shape) for b in self.biases]
        # feedforward
        activacion = x
        activaciones = [x] 
        zs = []
        for b, w in zip(self.biases, self.pesos):
            z = np.dot(w, activacion)+b
            zs.append(z)
            activacion = sigmoide(z)
            activaciones.append(activacion)
        # backward pass
        delta = (self.costo).delta(zs[-1], activaciones[-1], y)
        holderW[-1] = np.dot(delta, activaciones[-2].transpose())
        holderB[-1] = delta
        for t in range(2, self.num_capas):
            z = zs[-t]
            sp = sigmoideGradiente(z)
            delta = np.dot(self.pesos[-t+1].transpose(), delta) * sp
            holderW[-t] = np.dot(delta, activaciones[-t-1].transpose())
            holderB[-t] = delta
        return (holderB, holderW)

    def graficar(self, error_entrenamiento, error_validation, lambda_entrenamiento, lambda_validation):
        
        plt.plot(lambda_entrenamiento, error_entrenamiento, lambda_validation, error_validation)
        plt.title('Bias/Variance')
        #plt.xlable('Tamnho Data Set')
        #plt.ylabel('Error')
        plt.show()
    
    def salvar(self, nombreArchivo):
        archivo = open(nombreArchivo, "w")
        archivo.write(datosSalvar)
        archivo.close()
    def salvar1(self, nombreArchivo, datos1, datos2):
        datos1 ='INSERT INTO Prueba2 VALUES ("{}");'.format(datos1)
        datos2 ='INSERT INTO Prueba2 VALUES ("{}");'.format(datos2) 
        archivo = open(nombreArchivo, "w")
        archivo.write(datos1+'\n'+datos2)
        archivo.close()

