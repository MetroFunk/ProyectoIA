import numpy 
import sigmoid
import sigmoidGradient

def nnFuncionCosto(nn_params, input_layer_size,
                   hidden_layer_size, num_labels, X, y, lambda_nn):
    Theta1 = numpy.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                           \(hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = numpy.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                           \(num_labels, hidden_layer_size + 1), order='F')

    m = len(X)
    #sin regularizar
    # se deben pasar los numeros a vectores que los represeten
    J = 0
    Theta1_grad = numpy.zeros(Theta1.shape)
    Theta2_grad = numpy.zeros(Theta2.shape)

    X = numpy.column_stack((numpy.ones((m,1)),X))

    #5000
    #Esto lo convierte en vectores 5 se representa con el vector [0 0 0 0 1 0 0 0 ]
    hipotesis1 = sigmoid.sigmoid(numpy.dot(X,Theta1.T))
    hipotesis1 = numpy.column_stack((numpy.ones(hipotesis1)))
    hipotesis2 = sigmoid.sigmoid(np.dot(hipotesis1, Theta2.T))

    yEtiquetas = y;
    y = numpy.zeros((m, num_labels))

    for i in xrange(m):
        y[i, yEtiquetas[i]-1] = 1

    costo = 0

    for i in xrange(m):
        costo += numpy.sum(y[i] * numpy.log(hipotesis2[i]) + (1 - y[i]) * numpy.log(1 - hipotesis2[i]))
    J = -(1.0/m)*costo

    totalTheta1 = numpy.sum(numpy.sum(Theta1[:,1:]**2))
    totalTheta2 = numpy.sum(numpy.sum(Theta2[:,1:]**2))

    J = J+((lambda_nn/(2.0*m))*(totalTheta1+totalTheta2))
    #display("Funcion de costo con regularizar)
    #display(J)
    #regularizado 0.383770
    
    #backpropagation
    delta3 = 0
    delta2 = 0
    #delta2 5
    #delta3 3-5
    for t in xrange(m):

        x = X[t]

        hipotesis1 = sigmoid.sigmoid(numpy.dot(x, Theta1.T))

        hipotesis1 = numpy.concatenate((numpy.array([1]), hipotesis1))

        hipotesis3 = sigmoid.sigmoid(numpy.dot(a2, Theta2, T))

        delta3 = numpy.zeros((num_labels))

        for k in xrange(num_labels):
            y_k = y[t, k]
            delta3[k] = hipotesis2[k] - y_k

        delta2 = (numpy.dot(Theta2[:,1:].T, delta3).T) * sigmoidGradient.sigmoidGradient(numpy.dot(x, Theta1.T))

        Delta1 += numpy.outer(delta2, x)
        Delta2 += numpy.outer(delta3, hipotesis1)

    #regularizacion
    #de la matriz Theta1 y Theta 2 obtenemos el indexado
    Theta1_grad += Delta1 /m 

    Theta2_grad += Delta2 / m

    Theta1_grad_sin = numpy.copy(Theta1_grad)
    Theta2_grad_sin = numpy.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    Theta1_grad[:,0] = Theta1_grad_sin[:,0]
    Theta2_grad[:,0] = Theta2_grad_sin[:,0]

    grad = numpy.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))







    
