import scipy.io as sio
import numpy

mat = sio.loadmat("hw5data1.mat")
print(mat)

#  IA
#  —----------
#

# inicializar
clear ; close all; clc

# parametros
input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: carga y visualización =============
#

# cargando
print('Cargando y visualizando datos ...\n')

datos = sio.load('hw5data1.mat');
m = size(X, 1);

% aleatoriamente seleccione 100 imágenes
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Programa pausado.\n');
pause;

## ============ Part 2: regresión logística vectorizada ============
#

fprintf('\nEntrenando una regresión logística One-vs-All...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Programa pausado.\n');
pause;


## ================ Part 3: Predicciones para One-Vs-All ================
#  Luego ...
pred = predecirOneVsAll(all_theta, X);

fprintf('\nPrecisión sobre el training set: %f\n', mean(double(pred == y)) * 100);
