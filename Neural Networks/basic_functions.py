
import numpy as np 

def sigmoid(x1):
  s1 = 1/(1+np.exp(-1*x1))
  return s1
  
def sigmoid_derivative(x1):
  s1 = 1/(1+np.exp(-1*x1))
  ds1 = s1*(1-s1)
  return ds1
