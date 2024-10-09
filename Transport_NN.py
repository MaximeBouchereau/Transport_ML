#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 14:37:17 2022

@author: maximebouchereau
"""

# Code for numerical scheme performing for 1D-transport equation with periodic conditions via AI methods
# Domain in space: [0,1] with periodic boundary conditions
# Domain in time: [0,T] where T is the time of simulation (selected)

# Module imports

import numpy as np
import math as mt
from random import *
import random
import matplotlib.pyplot as plt
import sys
import warnings


warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

# Choice of parameters

# Maths parameters

c = 0.05     # Speed of transport
T = 100     # Time of simulation
ht = 0.1    # Time step for numerical simulation
J = 200     # Number of discretizations in space

# AI parameters

K = 200                   # Number of data
deg = 50                  # Degree of trigonometric polynomials used as initial conditions
p = 0.8                   # Proportion of data for training
alpha = 1e-3              # Step for stochastic gradient descent
BS =  10                  # Batch size for SGD gradient
N_epochs = 5000           # Number of epochs for SGD algorithm
N_epochs_print = 500      # Number of epochs between two prints of the Loss value for SGD algorithm


print(80*"_")
print(" ")
#print(80*"_")
print(" AI methods for transport equation numerical resolution performing")
print(80*"_")

print(" ")
print(" - Maths Parameters:")
print(" ")
print("   . Speed of transport:", c)
print("   . Time of simulation:", T)
print("   . Time step for numerical simulation:", ht)
print("   . Number of discretizations in space:", J)

print(" ")
print(" - AI parameters:")
print(" ")
print("   . Number of data:", K)
print("   . Degree of trigonometric polynomials for initial conditions:", deg)
print("   . Proportion of data for training:", round(100*p,1), " %")
print("   . Step for stochastic gradient descent:", format(alpha,'.2E'))
print("   . Batch size for SGD gradient:", BS)
print("   . Number of epochs for SGD algorithm:", N_epochs)
print("   . Number of epochs between two prints of the Loss value:", N_epochs_print)

if np.abs(c)*ht > 1/J:
    print(" ")
    print(" -  WARNING! - Possibility of unstable scheme for Lax-Wendroff:")
    print(" ")
    print("   . CFL Number - |c|ht/J:", round(np.abs(c)*ht*J,2))



class NA:
    """
    Class with tools for numerical resolution of the PDE
    """
    
    def u0(x):
        """
        Function of initial condition to the PDE. This function respects ths periodic conditions.

        Parameters
        ----------
        x : TYPE: Float
            DESCRIPTION: Input of the initial condition function, space variable.

        Returns the initial condition value at the input value
        -------
        None.

        """
        x = np.mod(x,1)
        y = np.exp(-25*(x-0.5)**2)
        return y
    
    def U0(X):
        """
        Function of initial condition to the PDE for arrays. This function respects ths periodic conditions.

        Parameters
        ----------
        X : TYPE: Array of shape (X.size,1) or (X.size,)
            DESCRIPTION: Inputs of the initial condition function, space variable

        Returns an array of shape (X.size,1) of the initial condition value at the input values.
        -------
        None.

        """
        X = X.reshape(X.size,)
        Y = X.copy()
        for j in range(X.size):
            Y[j] = NA.u0(X[j])
        return Y
    
    
    
    def D1(SD=J):
        """
        Creates the matrix of the first order space derivation operator with periodic conditions.

        Parameters
        ----------
        SD : TYPE: Integer
             DESCRIPTION: Number of space discretizations - Default: J

        Returns an array of shape (SD+1,SD+1) representing the first order space derivation operator with periodic
        conditions
        -------
        None.

        """
        
        D = np.diag(np.ones(SD),1) - np.diag(np.ones(SD),-1)
        D[-1,0] , D[0,-1] = 1 , -1
        D = (SD/2)*D
        return D
    
    def D2(SD=J):
        """
        Creates the matrix of the second order space derivation operator with periodic conditions.

        Parameters
        ----------
        SD : TYPE: Integer
             DESCRIPTION: Number of space discretizations - Default: J

        Returns an array of shape (SD+1,SD+1) representing the second order space derivation operator with periodic
        conditions
        -------
        None.

        """
        
        D = 2*np.diag(np.ones(SD+1),0) - np.diag(np.ones(SD),1) - np.diag(np.ones(SD),-1)
        D[-1,0] , D[0,-1] = -1 , -1
        D = (SD**2)*D
        return D
    
    def Integrate(M,SD=J,Tf=T,h=ht,v=c,plot=False):
        """
        Solves numerically the transport equation associated to the initial condition u0 by using the
        Backward Euler method (Implcit scheme) where the space derivation operator is centered.

        Parameters
        ----------
        M : TYPE: Array of shape (sqrt(M.size),sqrt(M.size))
            DESCRIPTION: Matrix used for the resolution
        SD : TYPE: Integer
            DESCRIPTION: Number of space discretizatiosn - Default: J.
        Tf : TYPE: Float
            DESCRIPTION: Time of integration. Default: T
        h : TYPE: Float
            DESCRIPTION: Time step of integration. Default: ht
        v : TYPE: Float
            DESCRIPTION: Velocity of the transport equation. Default: c
        plot : TYPE: Boolean
               DESCRIPTION: Plots the approximated solution which is computed of not. Default: False
        
        If plot = False:
            Returns an array of shape (SD,N+1) where N is the number of iterations of the numerical scheme, where the
            n-th column is the approximation of the solution at time t = nh
        If plot = True:
            Plots on a graph this array in order to show then approximated solution
        -------
        None.

        """        
        
        X = np.linspace(0,1,SD+1)
        N = int(Tf/h)                                          # Number of time iterations
        It = np.linalg.inv(np.eye(SD+1,SD+1) - (v*h)*M)        # Iteration matrix
        U = np.zeros((SD+1,N+1))                               # Matrix which contains the solution
        
        U[:,0] = NA.U0(X).reshape(SD+1,)
        
        for n in range(N):
            U[:,n+1] = It@U[:,n]
            
        if plot == False:
            return U
            
        if plot == True:
            T = np.arange(0,Tf,h)
            fig, ax = plt.subplots(1,1)
            plt.imshow(U,cmap="jet",extent=[0,Tf,0,1],aspect="auto")
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.title("Evolution of the approximated solution")
            
    def Integrate_LW(SD=J,Tf=T,h=ht,v=c,plot=False):
        """
        Solves numerically the transport equation associated to the initial condition u0 by using the
        Lax-Wendroff scheme where the first order space derivation operator is centered.

        Parameters
        ----------
        SD : TYPE: Integer
            DESCRIPTION: Number of space discretizatiosn - Default: J.
        Tf : TYPE: Float
            DESCRIPTION: Time of integration. Default: T
        h : TYPE: Float
            DESCRIPTION: Time step of integration. Default: ht
        v : TYPE: Float
            DESCRIPTION: Velocity of the transport equation. Default: c
        plot : TYPE: Boolean
               DESCRIPTION: Plots the approximated solution which is computed of not. Default: False
        
        If plot = False:
            Returns an array of shape (SD,N+1) where N is the number of iterations of the numerical scheme, where the
            n-th column is the approximation of the solution at time t = nh
        If plot = True:
            Plots on a graph this array in order to show then approximated solution
        -------
        None.

        """        
        
        X = np.linspace(0,1,SD+1)
        N = int(Tf/h)                                                       # Number of time iterations
        It = np.eye(SD+1,SD+1) + (v*h)*NA.D1(SD) - (v*h)**2/2*NA.D2(SD)     # Iteration matrix
        U = np.zeros((SD+1,N+1))                                            # Matrix which contains the solution
        
        U[:,0] = NA.U0(X).reshape(SD+1,)
        
        for n in range(N):
            U[:,n+1] = It@U[:,n]
            
        if plot == False:
            return U
            
        if plot == True:
            T = np.arange(0,Tf,h)
            fig, ax = plt.subplots(1,1)
            plt.imshow(U,cmap="jet",extent=[0,Tf,0,1],aspect="auto")
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.title("Evolution of the approximated solution")
            
            
class AI:
    """
    Class for the Artificial Intelligence methods
    """
    
    def TriPo(coeff,x):
        """
        Trigonometric polynomial function

        Parameters
        ----------
        coeff : TYPE: 1D Array (of shape (coeff.size,1) or (coeff.size,)) 
                DESCRIPTION: Array containing the coefficients of the trigonometric polynomial, coeff.size
                has to be odd.
        x : TYPE: Float
            DESCRIPTION: Input of the trigonometric polynomial function, space variable.

        Returns  the trigonometric polynomial value at the input value
        -------
        None.

        """
        
        coeff = coeff.reshape(coeff.size,)
        if coeff.size % 2 == 0:
            coeff = np.concatenate((coeff,np.array([0])))
        degree = coeff.size//2
        A = [coeff[0]]
        C = [coeff[n]*np.cos(2*np.pi*x) for n in range(1,degree+1)]
        S = [coeff[n]*np.sin(2*np.pi*x) for n in range(degree+1,2*degree+1)]
        return sum(A + C + S)
    
    
    def Data(K_data,SD=J,h=ht,save_data=True,name_data="Data_PDE"):
        """
        Creates data which are exact solutions of the equation for various initial conditions which are
        trigonometric polynomials

        Parameters
        ----------
        K_data : TYPE: Integer
                 DESCRIPTION: Number of data.
        SD : TYPE: Integer
             DESCRIPTION: Number of space discretizatiosn - Default: J.
        h : TYPE: Float
            DESCRIPTION: Time step for integration.
        save_data : TYPE: Boolean
                    DESCRIPTION: Saves the created data or not. default: True
        name_data : TYPE: Character string
                    DESCRIPTION: Name of the saved data (useful only if save_data = True). Default: "Data_PDE"

        Returns a tuple of length 2 which contains:
            - An array of shape (SD+1,K_data) whose the k-th column correspondsto the k-th initial condition
            - An array of shape (SD+1,K_data) whose the k-th column correspondsto the k-th solution at time t=h
        -------
        None.

        """
        
        Y0 , Y1 = np.zeros((SD+1,K_data)) , np.zeros((SD+1,K_data))
        
        print("   ")
        print("Data creation...")
        for k in range(K_data):
            count = round(100*k/K_data,1)
            sys.stdout.write("\r%d   "%count +"%")
            sys.stdout.flush()
            X = np.linspace(0,1,SD+1)
            coeff = np.random.uniform(low=-5,high=5,size=(2*deg+1,))
            for j in range(SD+1):
                Y0[j,k] = AI.TriPo(coeff,X[j])
                Y1[j,k] = AI.TriPo(coeff,X[j]+c*h)
                
        if save_data == True:
            np.save(name_data,(Y0,Y1))
            
        pass
    
    
    def Train(SD=J,h=ht,step=alpha,epochs=N_epochs,epochs_print=N_epochs_print,name_data="Data_PDE",save_model=True,name_model="model"):
        """
        Makes the train of the matrix W which is a special case of neural network (only one layer, SD neurons)
        and will replace the derivation matrix in order to limit numerical diffusion.

        Parameters
        ----------
        SD : TYPE: Integer
             DESCRIPTION: Number of space discretizatiosn - Default: J.
        h : TYPE: Float
            DESCRIPTION: Time step for integration.
        step : TYPE: Float
               DESCRIPTION: Step for the gradient descent. default: alpha
        epochs : TYPE: Integer
                 DESCRIPTION: Number of epochs for the gradient descent. default: N_epochs
        epochs_print: TYPE: Integer
                      DESCRIPTION: Number of epochs between two prints of the Loss value. default: N_epochs_print
        name_data : TYPE: Character string
                    DESCRIPTION: Name of the data which will be loaded and used for training. Default: "Data_PDE"
        save_model : TYPE: Boolean
                     DESCRIPTION: Saves the learned matrix, the Loss_train and Loss_test (model) or not. default: True
        name_model : TYPE: Character string
                     DESCRIPTION: Name of the saved model (useful only if save_model = True). Default: "model"
        Returns a tuple of length 3 which contains:
            - A list containing the values of the Loss_train
            - A list containing the values of the Loss_test
            - The matrix W
        -------
        None.

        """
        
        DATA = np.load(name_data+".npy")
        K = DATA[0].shape[1]     # Number of data used
        K0 = int(p*K)            # Number of data for training
        
        Y0_Train = DATA[0][:,0:K0]
        Y0_Test = DATA[0][:,K0:K-1]
        Y1_Train = DATA[1][:,0:K0]
        Y1_Test = DATA[1][:,K0:K-1]
        
        Loss_train , Loss_test = [] , []
        
        #W = np.zeros((SD+1,SD+1))
        W = NA.D1(SD)
        
        def dL(W,Y0_DATA,Y1_DATA,L):
            """
            Differential of the Loss function w.r.t. a batch of data

            Parameters
            ----------
            W : TYPE: Array
                DESCRIPTION: Input variable.
            Y0_DATA: TYPE: Array of shape (SD+1,len(L))
                     DESCRIPTION: Data set (time t=0) which is used in order to compute differentials for SGD
                     algorithm. Each column corresponds to the differential w.r.t. the corfresponding data.
            Y1_DATA: TYPE: Array of shape (SD+1,len(L))
                     DESCRIPTION: Data set (time t=h) which is used in order to compute differentials for SGD
                     algorithm. Each column corresponds to the differential w.r.t. the corfresponding data.
            L : TYPE: List of integers
                DESCRIPTION: Indices of the data which are used in order to compute the differentials for SGD algorithm

            Returns the mean of the differentials computed in an array of shape (SD+1,SD+1)
            -------
            None.

            """
            
            Diff = np.zeros((SD+1,SD+1))
            for k in L:
                u = Y0_DATA[:,k] - Y1_DATA[:,k]
                v = (c*h)*Y1_DATA[:,k]
                
                u , v = u.reshape(SD+1,1) , v.reshape(SD+1,1)
                
                Diff = Diff + 2*(u+W@v)@v.T/h**2
            
            return Diff/len(L)
            
            
            
        
        print(" ")
        print("Training...")
        print(" ")
        
        for n in range(epochs+1):
        
            Y1_Pred_Train = Y0_Train + (c*h)*(W)@Y1_Train
            Y1_Pred_Test = Y0_Test + (c*h)*(W)@Y1_Test
            
            loss_train = ((Y1_Pred_Train - Y1_Train)**2).mean()/h**2
            loss_test = ((Y1_Pred_Test - Y1_Test)**2).mean()/h**2
            
            Ind = random.sample(list(range(K0)),BS)
            
            #u = Y0_Train[:,k] - Y1_Train[:,k]
            #v = (c*h)*Y1_Train[:,k]
            
            W = W - (step/(n+1)**0.0)*dL(W,Y0_DATA=Y0_Train,Y1_DATA=Y1_Train,L=Ind)
            
            Loss_train.append(loss_train)
            Loss_test.append(loss_test)
            
            if n % epochs_print == 0:
                print("Step ",n,":","Loss_train=",format(loss_train,'.4E')," Loss_test=",format(loss_test,'.4E'))
        
        np.save( name_model, ( W , Loss_train , Loss_test ) )
        pass

class PG:
    """
    Class for plotting of graphs
    """
    
    def Solve(Ts=T,epochs=N_epochs,name_model="model",save_fig=False,name_fig="Transport_PDE_Learning"):
        """
        Plots the solution of the PDE with the matrix of the learned model and errors with the true solution

        Parameters
        ----------
        Ts : TYPE: Float
             DESCRIPTION: Time of the numerical simulation. Default: T

        epochs : TYPE: Integer
                 DESCRIPTION: Number of epochs used to train the model. default: N_epochs
     
        name_model : TYPE: Character string
                     DESCRIPTION: Name of the learned model which will be loaded. Default: "model"
                     
        save_fig : TYPE: Boolean
                   DESCRIPTION: Saves the figure or not. Default: False
                     
        name_model : TYPE: Character string
                     DESCRIPTION: Name of the saved figure (useful only if save-fig=True). Default: "Transport_PDE_Learning"
            

        Plots a graph of:
            - The solution with the matrix of the learned model
            - The Loss decay
            - The error between exact and approximated solutions
        -------
        None.

        """
        
        W , Loss_train , Loss_test = np.load(name_model+".npy",allow_pickle=True)
        
        X = np.linspace(0,1,J+1)    # Space interval of discretizations
        N = int(Ts/ht)              # Number of time discretizations
        S = np.linspace(0,Ts,N+1)   # Time interval of discretizations
        
        Uex = np.zeros((J+1,N+1))
        
        for n in range(N+1):
            Uex[:,n] = NA.U0(X+c*n*ht*np.ones(J+1))
            
        Unum = NA.Integrate(M=NA.D1(),Tf=Ts) # Numerical solution
        Ulwd = NA.Integrate_LW(Tf=Ts)        # Numerical solution with Lax-Wendroff scheme
        Uapp = NA.Integrate(M=W,Tf=Ts)       # Numerical solution with learned matrix
        
        ErrNum = np.abs(Uex - Unum)
        ErrLwd = np.abs(Uex - Ulwd)
        ErrApp = np.abs(Uex - Uapp)
        #plt.figure(0)
        #plt.imshow(Ulwd)
        #plt.figure(1)
        #plt.imshow(Uex)
        #plt.figure(2)
        #plt.imshow(ErrLwd)
        
        ErrNum_L1 = sum(ErrNum/(J+1)) # Evolution of the L1-norm error (numerical integration)
        ErrLwd_L1 = sum(ErrLwd/(J+1)) # Evolution of the L1-norm error (numerical integration with Lax-Wendroff scheme)
        ErrApp_L1 = sum(ErrApp/(J+1)) # Evolution of the L1-norm error (learning and numerical integration)
        
        ErrNum_L2 = sum(ErrNum**2/(J+1))**0.5 # Evolution of the L2-norm error (numerical integration)
        ErrLwd_L2 = sum(ErrLwd**2/(J+1))**0.5 # Evolution of the L2-norm error (numerical integration with Lax-Wendroff scheme)
        ErrApp_L2 = sum(ErrApp**2/(J+1))**0.5 # Evolution of the L2-norm error (learning and numerical integration)
        
        def Err_LI(Mat):
            """
            Computes the L-infty norm of the columns of a matrix.

            Parameters
            ----------
            Mat : TYPE: Array
                DESCRIPTION: The matrix whose columns are used to compute their L-infty norm

            Returns an array of shape (Mat.shape[1],) where coefficient k corresponds to the L-infty norm
            of the k-th column of Mat
            -------
            None.

            """
            
            Mat_LI = np.zeros(Mat.shape[1])
            for k in range(Mat.shape[1]):
                Mat_LI[k] = np.linalg.norm(Mat[:,k],ord = np.infty)
            return Mat_LI
            
        
        ErrNum_LI = Err_LI(ErrNum) # Evolution of the L-infty-norm error (numerical integration)
        ErrLwd_LI = Err_LI(ErrLwd) # Evolution of the L-infty-norm error (numerical integration with Lax-Wendroff scheme)
        ErrApp_LI = Err_LI(ErrApp) # Evolution of the L-infty-norm error (learning and numerical integration)
        
        print("L-infty error with numerical integration: ",format(max(ErrNum_LI),'.4E'))
        print("L-infty error with numerical integration with Lax-Wendroff scheme: ",format(max(ErrLwd_LI),'.4E'))
        print("L-infty error with numerical integration via training: ",format(max(ErrApp_LI),'.4E'))
        
        
        def write_size(option):
            """Changes the size of writings on all windows
            
            Parameters
            ----------
            option : TYPE: Integer
                     DESCRIPTION: Select the case where there are legends or not"""
            if option == 1:
                axes = plt.gca()
                axes.title.set_size(7)
                axes.xaxis.label.set_size(7)
                axes.yaxis.label.set_size(7)
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                plt.legend(fontsize=7)
            if option == 2:
                axes = plt.gca()
                axes.title.set_size(7)
                axes.xaxis.label.set_size(7)
                axes.yaxis.label.set_size(7)
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
            pass
        
        fig = plt.figure()
        
        #ax = fig.add_subplot(2, 1, 2)
        
        plt.subplot(2, 4, 1)
        
        plt.imshow(Uex,cmap="jet",aspect="auto",extent=(0,Ts,0,1))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Exact solution")
        plt.colorbar()
        write_size(2)
        
        plt.subplot(2, 4, 2)
        
        plt.imshow(Unum,cmap="jet",aspect="auto",extent=(0,Ts,0,1))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution with Implicit Euler scheme")
        plt.colorbar()
        write_size(2)
        
        plt.subplot(2, 4, 3)
        
        plt.imshow(Ulwd,cmap="jet",aspect="auto",extent=(0,Ts,0,1))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution with Lax-Wendroff scheme")
        plt.colorbar()
        write_size(2)
        
        plt.subplot(2, 4, 4)
        
        plt.imshow(Uapp,cmap="jet",aspect="auto",extent=(0,Ts,0,1))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solution with learned matrix")
        plt.colorbar()
        write_size(2)
        
        plt.subplot(2, 4, 5)
        
        plt.plot(range(epochs + 1), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(epochs + 1), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss functions')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size(1)
        
        plt.subplot(2, 4, 6)
        
        plt.plot(S,ErrNum_L1,color="red",label="Backward Euler")
        plt.plot(S,ErrApp_L1,color="green",label="Learning")
        plt.plot(S,ErrLwd_L1,color="orange",label="Lax-Wendroff")
        plt.xlabel("t")
        plt.ylabel("$L^1$-error")
        plt.legend()
        plt.grid()
        plt.title("Evolution of the error - $L^1$ norm")
        write_size(1)
        
        plt.subplot(2, 4, 7)
        
        plt.plot(S,ErrNum_L2,color="red",label="Backward Euler")
        plt.plot(S,ErrApp_L2,color="green",label="Learning")
        plt.plot(S,ErrLwd_L2,color="orange",label="Lax-Wendroff")
        plt.xlabel("t")
        plt.ylabel("$L^2$-error")
        plt.legend()
        plt.grid()
        plt.title("Evolution of the error - $L^2$ norm")
        write_size(1)
        
        plt.subplot(2, 4, 8)
        
        plt.plot(S,ErrNum_LI,color="red",label="Backward Euler")
        plt.plot(S,ErrApp_LI,color="green",label="Learning")
        plt.plot(S,ErrLwd_LI,color="orange",label="Lax-Wendroff")
        plt.xlabel("t")
        plt.ylabel("$L^{\infty}$-error")
        plt.legend()
        plt.grid()
        plt.title("Evolution of the error - $L^{\infty}$ norm")
        write_size(1)
        
        
        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 2, w * 2)
        
        if save_fig == True:
            plt.savefig(name_fig+".pdf")
        else:
            plt.show()
        
        pass
        

















