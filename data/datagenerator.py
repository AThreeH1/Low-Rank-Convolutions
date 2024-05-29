import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.imports import *


import random
import numpy as np

def DataGenerate(net, sequence_length, dim):

# l = length of signal
# m = point selected at random at which the signal starts
# d[] = list of dimensions for sinosuidal waves
# sigs = list of signal-dimension for waves to be classified
# Cp = Coincedent signal point or number 
# Ap = Any signal point or number
# S1,2,3 = Signal Types
# p or Tar = Singal selected or Target Signal respectively 

    dim = dim - 1
    data = []

    for r in range(net):

        # initialising dimentions with random numbers
        d = [[] for i in range(dim)]
        sigs = []

        for i in range(sequence_length):
            for j in range(dim):
                d[j].append((random.random()) - 0.5)
            sigs.append((random.random()) - 0.5)

        def s1(t):
            if t < 0.5:
                return t
            else:
                return 1 - t
        def s2(t):
            if 0 <= t < 0.5:
                return 0.5
            else:
                return -0.5
        def s3(t):
            return (-(0.5**2 - (t - 0.5)**2)**0.5)

        Cp = random.randint(1, dim)
        Ap = [i+1 for i in range(dim+1) if (i+1) != Cp]
        l = random.randint(10, sequence_length // (dim + 1))
        m = random.randint(0, (sequence_length // (dim + 1)) - l)

        for sn in range(dim+1):

            p = random.randint(1,3)

            if (sn+1) == Cp:
                Tar = p
            for i in range(l):
                for k in range(dim):
                    if (sn+1) != Ap[k]:                    
                        d[k][m] = (np.sin(i*2*np.pi/(l-1)))/2

                j = i/(l-1)
                if p == 1:
                    sigs[m] = s1(j)
                if p == 2:
                    sigs[m] = s2(j)
                if p == 3:
                    sigs[m] = s3(j)

                m += 1

            l = (random.randint(10, sequence_length//(dim+1)))
            m = random.randint(m, ((sn+2)*(sequence_length//(dim+1))-l))

        Batch = (*d[:dim], sigs, Tar-1)
        data.append(Batch)

    print('data stored')
    
    return data