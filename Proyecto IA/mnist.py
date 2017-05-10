import numpy as np
import csv
import matplotlib.pyplot 

## descargue un archivo que pasa mnist a csv

with open('mnist_test_10.csv', 'r') as csv_file:
    for data in csv.reader(csv_file):
        etiqueta = data[0]
        pixeles = data[1:]
        pixeles = np.array(pixeles, dtype='uint8')
        pixeles = pixeles.reshape((28, 28))
        matplotlib.pyplot .title('Label is {etiqueta}'.format(etiqueta=etiqueta))
        matplotlib.pyplot .imshow(pixeles, cmap='gray')
        matplotlib.pyplot .show()

       #muestra todas para detenerlo ctrl c
