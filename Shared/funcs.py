import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import linalg
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from math import comb
import random
import time
from sklearn.metrics import mean_squared_error
import copy
import scipy.linalg as lin

import torch
import torch.nn as nn
from scipy.optimize import brute
from scipy.optimize import Bounds

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import time as timer
from torch.utils.data import TensorDataset, random_split, DataLoader
import nengo

from scipy.interpolate import interp1d
import yaml
#Legendre domain
class legendre:
    def __init__(self,theta, dim, dt): 
        self.theta = theta # Representation window
        self.dim = dim # Order of representation
        self.dt = dt # Sampling time
        
        self.P = self.init_P()
        
    def init_P(self):
        window = int(self.theta/self.dt)
        P = np.zeros((window+1,self.dim))
        for t in range(window+1):
            r = t / window
            for i in range(self.dim):
                temp_c = pow((-1),i)
                temp_s = 0
                for j in range(i+1):
                    temp_s = temp_s + comb(i,j) * comb((i+j),j) * pow((-r),j)
             
                P[t,i] = temp_s*temp_c
        return P
    
    def decode(self,M): #Returns time domain data given legendre coefficients M
        window = int(self.theta/self.dt)
        u = np.zeros(window)
        for t in range(window):
            u[t] = np.sum(np.multiply(self.P[t,:],M))
        return u
    
    def encode(self, u): #Returns legendre coefficients M given time domain data u
        a = np.zeros(self.dim)
        for n in range(self.dim):
            for k in range(int(self.theta/self.dt)):
                a[n] = a[n] + self.P[k,n]*u[k]*self.dt
            a[n] = ( (1/self.theta)*(2*n+1) ) * a[n]
        return a
        
class cart_pole:
    def __init__(self,g,l,mp,mc,dt):
        self.g = g #Gravity
        self.l = l #Length of pendulum
        self.mp = mp #Mass of pendulum
        self.mc = mc #Mass of cart
        self.dt = dt #Sampling time
        
    def dynamics(self,x,u):
        x_dot = np.zeros(4)
        x_dot[0] = x[0] + x[1]*self.dt 

        x_dot[1] = x[1] + ( (u + self.mp * np.sin(x[2]) * (self.l * x[3]**2 - self.g * np.cos(x[2]))) / (self.mc + self.mp * np.sin(x[2])**2) )*(self.dt)
        
        x_dot[2] = x[2] + x[3]*self.dt
        
        x_dot[3] = x[3] + ( (-u*np.cos(x[2]) - self.mp*self.l*x[3]**2 * np.sin(x[2]) * np.cos(x[2]) + (self.mc + self.mp) * self.g * np.sin(x[2])) / (self.l * (self.mc + self.mp * np.sin(x[2])**2)) )*(self.dt)

        return x_dot

class pendulum:
    def __init__(self,g,l,m,dt):
        self.g = g
        self.l = l
        self.m = m 
        self.dt = dt
    def dynamics(self,x,u):
        x_dot = np.zeros(2)
        x_dot[0] = x[0] + x[1]*self.dt
        x_dot[1] = x[1] + (-(self.g/self.l)*np.sin(x[0])+u/(self.m*self.l**2))*(self.dt)
        return x_dot

class bike:
    def __init__(self,L, dt):
        self.L = L
        self.dt = dt
    def dynamics(self,x,v,u): #[x, y, theta, U Steer]
        x_dot = np.zeros(3)
        x_dot[0] = v * np.cos(x[2])
        x_dot[1] = v * np.sin(x[2])
        x_dot[2] = (v / self.L) * np.tan(u)
        return x+x_dot*self.dt

class LMU:
    def __init__(self,q,theta,dt):
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1)) 
        self.A = A / theta
        self.B = B / theta
        self.dim = q
        self.theta = theta
        self.dt = dt
    def dynamics(self,x,u):
        x_dot = np.matmul(self.A,x.T).T + (self.B*u).T
        return x + x_dot*dt

def get_mpc_reference(x_ref, y_ref, v, s0, Np, dt):
    s_ref = np.cumsum(np.sqrt(np.diff(x_ref, prepend=x_ref[0])**2 + np.diff(y_ref, prepend=y_ref[0])**2))
    x_interp = interp1d(s_ref, x_ref, kind='cubic', fill_value="extrapolate")
    y_interp = interp1d(s_ref, y_ref, kind='cubic', fill_value="extrapolate")

    s_mpc = s0 + np.arange(Np) * v * dt  
    
    x_mpc_ref = x_interp(s_mpc)
    y_mpc_ref = y_interp(s_mpc)

    return x_mpc_ref, y_mpc_ref
    
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def global_to_local(X,Y,x0,y0,theta):
    transform_mat = (np.matrix([[np.cos(theta),-np.sin(theta),x0],[np.sin(theta),np.cos(theta),y0],[0,0,1]]))**-1
    X_local = np.zeros(X.shape)
    Y_local = np.zeros(Y.shape)
    for j in range(len(X_local)):
        Pg = np.vstack((X[j],Y[j],1))
        Pl = np.matmul(transform_mat,Pg)
        X_local[j] = Pl[0,0]
        Y_local[j] = Pl[1,0]
    return X_local,Y_local

def local_to_global(X,Y,x0,y0,theta):
    transform_mat = (np.matrix([[np.cos(theta),-np.sin(theta),x0],[np.sin(theta),np.cos(theta),y0],[0,0,1]]))
    X_local = np.zeros(X.shape)
    Y_local = np.zeros(Y.shape)
    for j in range(len(X_local)):
        Pg = np.vstack((X[j],Y[j],1))
        Pl = np.matmul(transform_mat,Pg)
        X_local[j] = Pl[0,0]
        Y_local[j] = Pl[1,0]
    return X_local,Y_local
class LMU:
    def __init__(self,q,theta,dt):
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1)) 
        self.A = A / theta
        self.B = B / theta
        self.dim = q
        self.theta = theta
        self.dt = dt
    def dynamics(self,x,u):
        x_dot = np.matmul(self.A,x.T).T + (self.B*u).T
        return x + x_dot*dt

def add_to_buffer(buffer, new_data):
    buffer.append(new_data)
    if len(buffer) > buffer_size:
        buffer.pop(0)  # Remove oldest sample
    return buffer    

def sample_balanced_batch(buffer, batch_size, online_data_ratio=0.4):
    online_count = int(batch_size * online_data_ratio)
    offline_count = batch_size - online_count
    
    online_samples = random.sample(buffer, online_count)
    offline_samples = random.sample(offline_data, offline_count)
    
    return online_samples + offline_samples