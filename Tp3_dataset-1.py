import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Armar entradas y salidas
entrada1, entrada2, entrada3, entrada4, entrada5 = np.loadtxt('c:/Users/Facu/OneDrive/Escritorio/FACU/Maestría en Sistemas Espaciales (MSE)/Cursada/Optativas/Introducción al Aprendizaje Automático/Trabajos Prácticos/Varios/TP3 Datos/dataset-1-training-Entradas.txt', unpack='True')

salida1, salida2, salida3, salida4, salida5 = np.loadtxt('c:/Users/Facu/OneDrive/Escritorio/FACU/Maestría en Sistemas Espaciales (MSE)/Cursada/Optativas/Introducción al Aprendizaje Automático/Trabajos Prácticos/Varios/TP3 Datos/dataset-1-training-targets.txt', unpack='True')

entrada1_nueva, entrada2_nueva, entrada3_nueva, entrada4_nueva, entrada5_nueva = np.loadtxt('c:/Users/Facu/OneDrive/Escritorio/FACU/Maestría en Sistemas Espaciales (MSE)/Cursada/Optativas/Introducción al Aprendizaje Automático/Trabajos Prácticos/Varios/TP3 Datos/dataset-1-Entradas.txt', unpack='True')


entrada_training = np.vstack((entrada1,entrada2,entrada3,entrada4))
entrada_training = entrada_training.reshape(-1,1)

salida_training = np.vstack((salida1,salida2,salida3,salida4))
salida_training = salida_training.reshape(-1,1)

entrada_test = entrada5
entrada_test = entrada_test.reshape(-1,1)

salida_test = salida5
salida_test = salida_test.reshape(-1,1)

entrada_nueva = np.vstack((entrada1_nueva, entrada2_nueva, entrada3_nueva, entrada4_nueva, entrada5_nueva))
entrada_nueva = entrada_nueva.reshape(-1,1)

# Normalizar datos de entrada y salida

def normalizar_matriz(matriz):
    min_valor = matriz.min()
    max_valor = matriz.max()
    rango = max_valor - min_valor
    matriz_normalizada= (matriz - min_valor)/rango * 2 - 1
    return matriz_normalizada


entrada_training_norm = normalizar_matriz(entrada_training)
salida_traininig_norm = normalizar_matriz(salida_training)

entrada_test_norm = normalizar_matriz(entrada_test)
salida_test_norm = normalizar_matriz(salida_test)

entrada_nueva_norm = normalizar_matriz(entrada_nueva)

# Armar Matriz Phi con un polinomio de 1er grado, es decir, phi1 = 1 y phi2 = x.

M = 1

Phi = np.zeros((len(entrada_training_norm), M+1))
for m in range(M+1):
    Phi[:,m] = entrada_training_norm[:,0]**m


Phi_transpuesta = np.transpose(Phi)
Phi_pseudo_inversa = np.dot(np.linalg.inv(np.dot(Phi_transpuesta,Phi)),Phi_transpuesta)

w_ML = np.dot(Phi_pseudo_inversa,salida_traininig_norm)


# Armar Matriz Phi con entradas de test
M = 1

Phi_test = np.zeros((len(entrada_test_norm), M+1))
for m in range(M+1):
    Phi_test[:,m] = entrada_test_norm[:,0]**m



# Calcular salida

t = np.dot(np.transpose(w_ML),np.transpose(Phi_test))

entrada_salida = np.hstack((entrada_test_norm, np.transpose(t)))
entrada_salida_ordenada = entrada_salida[np.argsort(entrada_salida[:, 0])]

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(salida_test_norm, np.transpose(t)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(salida_test_norm, np.transpose(t)))


# Plot outputs
plt.scatter(entrada_training_norm, salida_traininig_norm, color="black", label= 'Datos de entrenamiento')
plt.plot(entrada_salida_ordenada[:,0], entrada_salida_ordenada[:,1], color="blue", linewidth=2, label= 'Datos de test - predicción')
plt.scatter(entrada_test_norm, salida_test_norm, color="red", linewidth=0.5, label= 'Datos de test - reales')


plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.legend()

plt.show()


# Hacer lo mismo que antes pero con un polinomio de grado 20, es decir, phi1 = 1, phi2 = x, phi3 = x^2, phi4 = x^3.......

# phi1, phi2 y phi3 ya están definidos


M = 2

Phi1 = np.zeros((len(entrada_training_norm), M+1))
for m in range(M+1):
    Phi1[:,m] = entrada_training_norm[:,0]**m

#Phi1 = np.hstack((phi1, phi2, phi3))



Phi_transpuesta1 = np.transpose(Phi1)
Phi_pseudo_inversa1 = np.dot(np.linalg.inv(np.dot(Phi_transpuesta1,Phi1)),Phi_transpuesta1)

w_ML1 = np.dot(Phi_pseudo_inversa1,salida_traininig_norm)

# Armar Matriz Phi con entradas de test

#phi1, phi2 y phi3 ya están definidas
M = 2

Phi_test1 = np.zeros((len(entrada_test_norm), M+1))
for m in range(M+1):
    Phi_test1[:,m] = entrada_test_norm[:,0]**m




# Calcular salida

t1 = np.dot(np.transpose(w_ML1),np.transpose(Phi_test1))

entrada_salida1 = np.hstack((entrada_test_norm, np.transpose(t1)))
entrada_salida_ordenada1 = entrada_salida1[np.argsort(entrada_salida1[:, 0])]


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(salida_test_norm, np.transpose(t1)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(salida_test_norm, np.transpose(t1)))

# Plot outputs
plt.scatter(entrada_training_norm, salida_traininig_norm, color="black", label='Datos de entrenamiento')
plt.plot(entrada_salida_ordenada1[:, 0], entrada_salida_ordenada1[:, 1], color="blue", linewidth=2, label= 'Datos de test - predicción')
plt.scatter(entrada_test_norm, salida_test_norm, color="red", linewidth=0.5, label= 'Datos de test - reales')

plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.legend()

plt.show()

# Aplicar a la entrada nueva

# Armar Matriz Phi con entradas de test

M = 2

Phi_nueva = np.zeros((len(entrada_nueva_norm), M+1))
for m in range(M+1):
    Phi_nueva[:,m] = entrada_nueva_norm[:,0]**m


# Calcular salida

t2 = np.dot(np.transpose(w_ML1),np.transpose(Phi_nueva))

entrada_salida2 = np.hstack((entrada_nueva_norm, np.transpose(t2)))
entrada_salida_ordenada2 = entrada_salida2[np.argsort(entrada_salida2[:, 0])]



# Plot outputs
plt.scatter(entrada_training_norm, salida_traininig_norm, color="black", label= 'Datos de entrenamiento')
plt.plot(entrada_salida_ordenada2[:, 0], entrada_salida_ordenada2[:, 1], color="blue", linewidth=2, label= 'Datos nuevos - predicción')
plt.scatter(entrada_test_norm, salida_test_norm, color="red", linewidth=0.5, label= 'Datos de test - reales')



plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.legend()

plt.show()