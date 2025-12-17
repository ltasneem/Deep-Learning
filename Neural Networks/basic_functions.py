
import numpy as np 

def sigmoid(x1):
  s1 = 1/(1+np.exp(-1*x1))
  return s1
  
def sigmoid_derivative(x1):
  s1 = 1/(1+np.exp(-1*x1))
  ds1 = s1*(1-s1)
  return ds1

def image2vector(image1):
  v1 = image.reshape(image1.shape[0]*image1.shape[1]*image1.shape[2],1)
  return v1
  
def normalizeRows(x1):
  x_norm1 = np.linalg.norm(x1,axis=1,keepdims=True)
  x1 = x1/x_norm1
  return x1

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1,keepdims = True)
    s = x_exp/x_sum
    return s
