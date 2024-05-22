import os
import GPy
import random
import igraph as ig
import numpy as np
import copy

import torch
from torch.distributions import MultivariateNormal, Normal, Laplace, Gumbel
from torch.distributions import Uniform


class Dist(object):
    def __init__(self, d, SummaryDAG, AutoREG_1st, AutoREG_2nd, gunc = 'linear', func = 'linear', norm = False, 
                 noise_std = [1]+[0.4]*12, noise_type = ['Gauss']*12, 
                 lengthscale = 1, f_magn = 1, GraNDAG_like = False, IVlayer=0):

        self.d = d
        if isinstance(noise_std, (int, float)):
            noise_std = noise_std * torch.ones(self.d)
        self.func = func
        self.gunc = gunc
        self.norm = norm
        self.lengthscale = lengthscale
        self.f_magn = f_magn
        self.GraNDAG_like = GraNDAG_like
        
        if self.GraNDAG_like:
            noise_std = [torch.ones(d), torch.ones(d), torch.ones(d), torch.ones(d)]

        self.noiseE0 = self.noiseDist(noise_type[0], noise_std[0])
        self.noiseE1 = self.noiseDist(noise_type[1], noise_std[1])
        self.noiseE2 = self.noiseDist(noise_type[2], noise_std[2])
        self.noiseE3 = self.noiseDist(noise_type[3], noise_std[3])
        self.noiseE4 = self.noiseDist(noise_type[4], noise_std[4])
        self.noiseE5 = self.noiseDist(noise_type[5], noise_std[5])
        self.noiseE6 = self.noiseDist(noise_type[6], noise_std[6])
        self.noiseE7 = self.noiseDist(noise_type[7], noise_std[7])
        self.noiseE8 = self.noiseDist(noise_type[8], noise_std[8])
        self.noiseE9 = self.noiseDist(noise_type[9], noise_std[9])
        self.noiseE10 = self.noiseDist(noise_type[10], noise_std[10])
        self.noiseE11 = self.noiseDist(noise_type[11], noise_std[11])
        if IVlayer == 12 or IVlayer == 23:
            self.noiseIV = self.noiseDist('Gauss', torch.ones(d))
        
        self.SummaryDAG = SummaryDAG
        self.AutoREG_1st = AutoREG_1st
        self.AutoREG_2nd = AutoREG_2nd

        assert(np.allclose(self.SummaryDAG, np.triu(self.SummaryDAG)))


    def noiseDist(self, noise_type_, noise_std_):
        if isinstance(noise_std_, (int, float)):
            noise_std_ = noise_std_ * torch.ones(self.d)

        if noise_type_ == 'Gauss':
            return Normal(0, noise_std_) # give standard deviation
        elif noise_type_ == 'Laplace':
            return Laplace(0, noise_std_ / np.sqrt(2))
        elif noise_type_ == 'Uniform':
            print("Hello Uniform.")
            return Uniform( noise_std_ * (-torch.ones(self.d)), noise_std_ * torch.ones(self.d))
        else:
            raise NotImplementedError("Unknown noise type for noise E1.")


    def sampleGP(self, X, lengthscale=1):
        ker = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=lengthscale,variance=self.f_magn)
        C = ker.K(X,X)
        X_sample = np.random.multivariate_normal(np.zeros(len(X)),C)
        return X_sample


    def sample(self, path, n, IVlayer=0, IVs=[]):

        _noiseE0 = self.noiseE0.sample((n,)) # n x d noise matrix
        _noiseE1 = self.noiseE1.sample((n,)) # n x d noise matrix
        _noiseE2 = self.noiseE2.sample((n,)) # n x d noise matrix
        _noiseE3 = self.noiseE3.sample((n,)) # n x d noise matrix
        _noiseE4 = self.noiseE4.sample((n,)) # n x d noise matrix
        _noiseE5 = self.noiseE5.sample((n,)) # n x d noise matrix
        _noiseE6 = self.noiseE6.sample((n,)) # n x d noise matrix
        _noiseE7 = self.noiseE7.sample((n,)) # n x d noise matrix
        _noiseE8 = self.noiseE8.sample((n,)) # n x d noise matrix
        _noiseE9 = self.noiseE9.sample((n,)) # n x d noise matrix
        _noiseE10 = self.noiseE10.sample((n,)) # n x d noise matrix
        _noiseE11 = self.noiseE11.sample((n,)) # n x d noise matrix
        if IVlayer == 12 or IVlayer == 23:
            _noiseIV = self.noiseIV.sample((n,)) # n x d noise matrix

        _X0 = copy.deepcopy(_noiseE0)
        _X1 = self.nextPanel_1(copy.deepcopy(_X0), copy.deepcopy(_noiseE1))
        if IVlayer == 12:
            _XIV = copy.deepcopy(_X1)
            _XIV[:,IVs] = _noiseIV[:,IVs]
            _X2 = self.nextPanel_23(copy.deepcopy(_X0), copy.deepcopy(_XIV), copy.deepcopy(_noiseE2))
        else:
            _X2 = self.nextPanel_23(copy.deepcopy(_X0), copy.deepcopy(_X1), copy.deepcopy(_noiseE2))
        
        if IVlayer == 23:
            _XIV = copy.deepcopy(_X2)
            _XIV[:,IVs] = _noiseIV[:,IVs]
            _X3 = self.nextPanel_23(copy.deepcopy(_X1), copy.deepcopy(_XIV), copy.deepcopy(_noiseE3))
        else:
            _X3 = self.nextPanel_23(copy.deepcopy(_X1), copy.deepcopy(_X2), copy.deepcopy(_noiseE3))

        _X4 = self.nextPanel_23(copy.deepcopy(_X2), copy.deepcopy(_X3), copy.deepcopy(_noiseE4))
        _X5 = self.nextPanel_23(copy.deepcopy(_X3), copy.deepcopy(_X4), copy.deepcopy(_noiseE5))
        _X6 = self.nextPanel_23(copy.deepcopy(_X4), copy.deepcopy(_X5), copy.deepcopy(_noiseE6))
        _X7 = self.nextPanel_23(copy.deepcopy(_X5), copy.deepcopy(_X6), copy.deepcopy(_noiseE7))

        _X8 = self.nextPanel_23(copy.deepcopy(_X6), copy.deepcopy(_X7), copy.deepcopy(_noiseE8))
        _X9 = self.nextPanel_23(copy.deepcopy(_X7), copy.deepcopy(_X8), copy.deepcopy(_noiseE9))
        _X10 = self.nextPanel_23(copy.deepcopy(_X8), copy.deepcopy(_X9), copy.deepcopy(_noiseE10))
        _X11 = self.nextPanel_23(copy.deepcopy(_X9), copy.deepcopy(_X10), copy.deepcopy(_noiseE11))


        if IVlayer == 12:
            _X = torch.stack([_X0, _XIV, _X2, _X3, _X4, _X5, _X6, _X7, _X8, _X9, _X10, _X11, _X1], dim=0)
        elif IVlayer == 23:
            _X = torch.stack([_X0, _X1, _XIV, _X3, _X4, _X5, _X6, _X7, _X8, _X9, _X10, _X11, _X2], dim=0)
        else:
            _X = torch.stack([_X0, _X1, _X2, _X3, _X4, _X5, _X6, _X7, _X8, _X9, _X10, _X11], dim=0)
            
        _Noise = torch.stack([_noiseE0, _noiseE1, _noiseE2, _noiseE3, _noiseE4, _noiseE5], dim=0)

        torch.save(_X,     path+'X.pt')
        torch.save(_Noise, path+'Noise.pt')

        print("Save to the Dir: {}. ".format(path))

        return _X, _Noise

    def non_linear_back(self, X, function):
        if function == 'sin':
            return torch.sin(X)
        elif function == 'pow2':
            return 0.1*torch.pow(X+2,2)
        elif function == 'pow3':
            return 0.1*torch.pow(X+2,2)
        elif function == 'poly':
            return 0.1*torch.pow(X+2,2)
        elif function == 'sigmoid':
            return 3/(1+torch.exp(X))
        else:
            return X

    def nextPanel_1(self, X0, X1):
        noise_var = np.zeros(self.d)
        if self.func == 'GP' or self.func == 'Gaussian Processes':
            for i in range(self.d):
                parents0 = np.nonzero(self.AutoREG_1st[:,i])[0]
                parents1 = np.nonzero(self.SummaryDAG[:,i])[0]
                if self.GraNDAG_like:
                    if len(parents1) == 0: 
                        noise_var[i] = np.random.uniform(1,2)
                    else: 
                        noise_var[i] = np.random.uniform(0.4,0.8)
                    X1[:, i] = np.sqrt(noise_var[i]) * X1[:, i]
                X1[:, i] += torch.tensor(self.sampleGP(np.array(X0[:,i]), self.lengthscale))
                if len(np.nonzero(self.AutoREG_1st[:,i])[0]) > 0:
                    X_par0 = X0[:,parents1]
                    X1[:, i] += torch.tensor(self.sampleGP(np.array(X_par0), self.lengthscale)) * 0.1
                if len(np.nonzero(self.SummaryDAG[:,i])[0]) > 0:
                    X_par1 = X1[:,parents1]
                    X1[:, i] += torch.tensor(self.sampleGP(np.array(X_par1), self.lengthscale))
        else:
            for i in range(self.d):
                X1[:, i] += self.non_linear_back(X0[:,i], self.gunc)
                for j in np.nonzero(self.AutoREG_1st[:,i])[0]:
                    X1[:, i] += self.non_linear_back(X0[:,j], self.gunc) * 0.1
                for j in np.nonzero(self.SummaryDAG[:,i])[0]:
                    X1[:, i] += self.non_linear_back(X1[:,j], self.func)
                
        if self.norm:
            X1 = (X1 - X1.mean(0))/X1.std(0)

        return X1

    def nextPanel_23(self, X0, X1, X2):
        noise_var = np.zeros(self.d)
        if self.func == 'GP' or self.func == 'Gaussian Processes':
            for i in range(self.d):
                parents2nd = np.nonzero(self.AutoREG_2nd[:,i])[0]
                parents1st = np.nonzero(self.AutoREG_1st[:,i])[0]
                parentsDAG = np.nonzero(self.SummaryDAG[:,i])[0]
                if self.GraNDAG_like:
                    if len(parents1) == 0: 
                        noise_var[i] = np.random.uniform(1,2)
                    else: 
                        noise_var[i] = np.random.uniform(0.4,0.8)
                    X2[:, i] = np.sqrt(noise_var[i]) * X2[:, i]

                X2[:, i] += torch.tensor(self.sampleGP(np.array(X1[:,i]), self.lengthscale))
                if len(np.nonzero(self.AutoREG_2nd[:,i])[0]) > 0:
                    X_par2nd = X0[:,parents2nd]
                    X2[:, i] += torch.tensor(self.sampleGP(np.array(X_par2nd), self.lengthscale)) * 0.1
                if len(np.nonzero(self.AutoREG_1st[:,i])[0]) > 0:
                    X_par1st = X1[:,parents1st]
                    X2[:, i] += torch.tensor(self.sampleGP(np.array(X_par1st), self.lengthscale)) * 0.1
                if len(np.nonzero(self.SummaryDAG[:,i])[0]) > 0:
                    X_parDAG = X2[:,parentsDAG]
                    X2[:, i] += torch.tensor(self.sampleGP(np.array(X_parDAG), self.lengthscale))
        else:
            for i in range(self.d):
                X2[:, i] += self.non_linear_back(X1[:,i], self.gunc)
                for j in np.nonzero(self.AutoREG_2nd[:,i])[0]:
                    X2[:, i] += self.non_linear_back(X0[:,j], self.gunc) * 0.1
                for j in np.nonzero(self.AutoREG_1st[:,i])[0]:
                    X2[:, i] += self.non_linear_back(X1[:,j], self.gunc) * 0.1
                for j in np.nonzero(self.SummaryDAG[:,i])[0]:
                    X2[:, i] += self.non_linear_back(X2[:,j], self.func)
                
        if self.norm:
            X2 = (X2 - X2.mean(0))/X2.std(0)
        return X2