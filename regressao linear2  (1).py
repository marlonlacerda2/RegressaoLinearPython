import numpy as np
import matplotlib.pyplot as plt
from time import time


def load_data(filename):
  X = np.loadtxt(filename, delimiter=';', skiprows=1)
  Y = X[:, X.shape[1]-1] # Pega a última coluna do csv (nota)
  Y = np.expand_dims(Y, axis=1)
  X = X[:, 0:X.shape[1]-1] # Remove a última coluna de X (nota) das entradas

  return X, Y

def calculate_h_theta(X, thetas, m):
  h_theta = np.zeros((m,1))
  h_theta=np.dot(thetas,X)
  return h_theta
  
def calculate_J(X, Y, thetas):
  m, _ = np.shape(X)
  h_theta = calculate_h_theta(X, thetas, m)
  J=np.dot((h_theta-Y),X)
  J=np.dot(np.transpose(e),e)/(len(X))
  return J, h_theta

def do_train(X, Y, thetas, alpha, iterations):
  J = np.zeros(iterations)
  m, n = np.shape(X)
  for i in range(iterations):
    J[i], h_theta = calculate_J(X, Y, thetas)
  return J

def feature_scaling(X):
  normalized_X = X
  return normalized_X

if __name__ == "__main__":
  X, Y = load_data('/home/marlon/Documents/Topicos em ia/Trabalho2 TIA/vinhotinto.csv')

  X = feature_scaling(X)
  
  X = np.insert(X, 0, values=1, axis=1) 
  
  m, n = X.shape
  thetas = np.zeros((n,1))
  alpha = 0.1
  
  inicio = time()
  J = do_train(X, Y, thetas, alpha=alpha, iterations=50)
  fim = time()
  
  print("Tempo de execução: {}".format(fim-inicio))
  
  

  plt.figure(1)
  plt.plot(J)
  plt.title(r'$J(\theta$) vs iterações')
  plt.ylabel(r'$J(\theta$)', rotation=0)
  plt.xlabel("iteração")
  plt.show()
