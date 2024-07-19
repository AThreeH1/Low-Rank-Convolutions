import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now you can import your modules
from utils.imports import *
from data.datagenerator import DataGenerate

# device = torch.device('cuda')

# Implementing ISS
def ISS(x_data, a):
    '''Caculates the low-rank expansion of
            < ISS_{s,t}, 1^(a/2 + 1)2 > = ..
                if a == even
            < ISS_{s,t}, 2^(a/2 + 0.5)1 > = ..
                if a == odd
        Args:
            x_data: shape (bs, 2, T)
            a: int 
        Returns:
            out: lowrank list of 4 tensors
    '''
    bs = x_data.size(0)
    out = []
    dims = x_data.size(1)

    if dims == 2:
        if a%2 == 0:
            D1, D2 = x_data[:, 1, :]**((a+2)/2), x_data[:, 0, :]
        if a%2 == 1:
            D1, D2 = x_data[:, 0, :]**((a+1)/2), x_data[:, 1, :]
    
    if dims == 3:
        if a%6 == 0:
            D1, D2 = x_data[:, 0, :]**((a+6)/6), x_data[:, 1, :]
        if a%6 == 1:
            D1, D2 = x_data[:, 1, :]**((a+5)/6), x_data[:, 2, :]
        if a%6 == 2:
            D1, D2 = x_data[:, 0, :]**((a+4)/6), x_data[:, 2, :]      
        if a%6 == 3:
            D1, D2 = x_data[:, 1, :]**((a+3)/6), x_data[:, 0, :]
        if a%6 == 4:
            D1, D2 = x_data[:, 2, :]**((a+2)/6), x_data[:, 1, :]
        if a%6 == 5:
            D1, D2 = x_data[:, 2, :]**((a+1)/6), x_data[:, 0, :]  

    T = D1.size(1) + 1

    Consicutive_Sum_D1 = torch.cat([torch.zeros(bs, 1, device=x_data.device), torch.cumsum(D1, dim=1)], dim=1)
    Consicutive_Sum_D2 = torch.cat([torch.zeros(bs, 1, device=x_data.device), torch.cumsum(D2, dim=1)], dim=1)
    QS = D1 * D2
    ISS_QS = torch.cat([torch.zeros(bs, 1, device=x_data.device), torch.cumsum(QS, dim=1)], dim=1)

    ISS_D1 = torch.zeros(bs, T, device=x_data.device)
    ISS_D2 = torch.zeros(bs, T, device=x_data.device)

    # Using broadcasting to perform element-wise operations on tensors
    range_tensor = torch.arange(2, T, device=x_data.device).unsqueeze(0)  # shape: (1, T-1)
    ISS_D1[:, 1:] = torch.cumsum((Consicutive_Sum_D1[:, :-1] * D2), dim=1)
    ISS_D2[:, 1:] = torch.cumsum((Consicutive_Sum_D2[:, :-1] * D1), dim=1)

    negated_Consicutive_Sum_D1 = -Consicutive_Sum_D1
    ISS_sum = ISS_D2 + ISS_QS

    out.append([ISS_sum, negated_Consicutive_Sum_D1, Consicutive_Sum_D2, ISS_D1])

    return out


def super_brute_force_two_letter_ISS(x_data, w):
    '''
        For 0 < t (in python indexing)
            \sum_{-1 < r1 < r2 <= t} x^{(w_1)}_{r1} x^{(w_2)}_{r2}
    '''
    bs, d, T = x_data.size()
    tmp = torch.zeros(bs, T)
    for t in range(4, T):
        for r1 in range(2, t):
            for r2 in range(r1+1, t):
                tmp[:, t] += x_data[:, w[0], r1] * x_data[:, w[1], r2]
    return tmp

# Define the FNN class
class FNNnew(nn.Module):
    def __init__(self):
        super(FNNnew, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)
    
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        
        # Bias of the last layer = 1
        nn.init.ones_(self.fc2.bias)

        self.a = -50

    def forward(self, x):
        z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        y = torch.exp(self.a * x)
        return z*y

# Function to perform convolution
def convolve_sequences(h_fft, f_fft):
    Y = h_fft * f_fft
    y = torch.fft.ifft(Y, dim=-1)
    return y.real

# Define the LowRankModel class
class LowRankModel(nn.Module):
    def __init__(self, words):
        """
            Sum_{s,s'}(h(t-s, t-s') * < ISS_{s,s'},Word >)
            Where,
            h(s,s') = f(s).g(s') + f'(s).g'(s')
            And chens identity for ISS = ... rank 3

            args:
                bs = batch size
                T = sequence length or the span of t
                f, g, f_prime, etc = Components of h
                AA, AB, BA, etc : Low rank components

            output:
                Sequence of size (bs, words, T+1)

        """
        super(LowRankModel, self).__init__()
        
        # Initialize the FFN models 
        self.f = FNNnew()
        self.g = FNNnew()
        self.f_prime = FNNnew()
        self.g_prime = FNNnew()
        self.words = words

    def forward(self, x_data):
        words = self.words
        T = x_data.size(2)
        bs = x_data.size(0)
        f_arr = self.f(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(x_data.device)).squeeze().flip(0)
        f_prime_arr = self.f_prime(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(x_data.device)).squeeze().flip(0)
        g_arr = self.g(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(x_data.device)).squeeze()
        g_prime_arr = self.g_prime(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(x_data.device)).squeeze()       
        
        f_arr = torch.nn.functional.pad(f_arr, (T, T))
        f_prime_arr = torch.nn.functional.pad(f_prime_arr, (T, T))
        g_arr = torch.nn.functional.pad(g_arr, (T, T))
        g_prime_arr = torch.nn.functional.pad(g_prime_arr, (T, T))

        ones = torch.ones(T+1, device=x_data.device)
        ones = torch.nn.functional.pad(ones, (T, T))

        H_f_arr = torch.fft.fft(f_arr, dim=0)
        H_f_prime_arr = torch.fft.fft(f_prime_arr, dim=0)
        H_g_arr = torch.fft.fft(g_arr, dim=0)
        H_g_prime_arr = torch.fft.fft(g_prime_arr, dim=0)
        H_ones = torch.fft.fft(ones, dim=0).reshape(1, 3*T+1)

        output_seq = torch.zeros((bs, words, 3*T+1), device=x_data.device, dtype=torch.complex64)

        for k in range(words):
            Array_LowRank = ISS(x_data, a=k)

            # Convert list of lists to tensor for batch processing
            ISS_sum = torch.stack([item[0] for item in Array_LowRank], dim=0).to(x_data.device)
            Consicutive_Sum_D1 = torch.stack([item[1] for item in Array_LowRank], dim=0).to(x_data.device)
            Consicutive_Sum_D2 = torch.stack([item[2] for item in Array_LowRank], dim=0).to(x_data.device)
            ISS_D1 = torch.stack([item[3] for item in Array_LowRank], dim=0).to(x_data.device) #[:,:,1:]

            ISS_sum = torch.nn.functional.pad(ISS_sum, (T, T))
            Consicutive_Sum_D1 = torch.nn.functional.pad(Consicutive_Sum_D1, (T, T))
            Consicutive_Sum_D2 = torch.nn.functional.pad(Consicutive_Sum_D2, (T, T))
            ISS_D1 = torch.nn.functional.pad(ISS_D1, (T, T))
            
            # Apply FFT to all elements in the batch
            ISS_sum_fft = torch.fft.fft(ISS_sum, dim=-1)
            Consicutive_Sum_D1_fft = torch.fft.fft(Consicutive_Sum_D1, dim=-1)
            Consicutive_Sum_D2_fft = torch.fft.fft(Consicutive_Sum_D2, dim=-1)
            ISS_D1_fft = torch.fft.fft(ISS_D1, dim=-1)

            # Perform batched convolution and accumulate results
            AA = convolve_sequences(H_f_arr, ISS_sum_fft)
            AB = convolve_sequences(H_g_arr, H_ones)
            BA = convolve_sequences(H_f_prime_arr, ISS_sum_fft)
            BB = convolve_sequences(H_g_prime_arr, H_ones)
            CA = convolve_sequences(H_f_arr, Consicutive_Sum_D1_fft)
            CB = convolve_sequences(H_g_arr, Consicutive_Sum_D2_fft)
            DA = convolve_sequences(H_f_prime_arr, Consicutive_Sum_D1_fft)
            DB = convolve_sequences(H_g_prime_arr, Consicutive_Sum_D2_fft)
            EA = convolve_sequences(H_f_arr, H_ones)
            EB = convolve_sequences(H_g_arr, ISS_D1_fft)
            FA = convolve_sequences(H_f_prime_arr, H_ones)
            FB = convolve_sequences(H_g_prime_arr, ISS_D1_fft)
            output_seq[:, k, :] = 0.5*((AA * AB) + (BA * BB) + (CA * CB) + (DA * DB) + (EA * EB) + (FA * FB))
        output_seq = output_seq.real
        return output_seq[:,:,2*T:]

def h(T_s, T_s_prime, f, g, f_prime, g_prime):

    T_s = torch.tensor([[T_s]], dtype=torch.float32) 
    T_s_prime = torch.tensor([[T_s_prime]], dtype=torch.float32)  
    
    # Compute f(T-s), g(T-s'), f'(T-s), g'(T-s')
    f_T_s = f(T_s)
    g_T_s_prime = g(T_s_prime)
    f_prime_T_s = f_prime(T_s)
    g_prime_T_s_prime = g_prime(T_s_prime)
    
    # Compute h(T-s, T-s') = f(T-s)g(T-s') + f'(T-s)g'(T-s')
    h_value = f_T_s * g_T_s_prime + f_prime_T_s * g_prime_T_s_prime
    # print('A', f_T_s , g_T_s_prime, 'B',f_prime_T_s , g_prime_T_s_prime)
    return h_value.item()

def super_brute_force_LowRank(x_data, words):
    f = FNNnew()
    # set_seed(f)
    g = FNNnew()
    # set_seed(g)
    f_prime = FNNnew()
    # set_seed(f_prime)
    g_prime = FNNnew()
    # set_seed(g_prime)
    bs,_,T = x_data.size()
    T += 1
    out = torch.zeros([3, words, T], device=x_data.device)


    for bs in range(bs):
        for k in range(words):
            for t in range(T):
                total_sum = 0
                for s in range(T):
                    for s_prime in range(T):
                        if t>= s and t>=s_prime:
                            h_value = h(s, t-s_prime, f, g, f_prime, g_prime)
                            Array = ISS(x_data, k)
                            Z = Array[0][0][bs][s] + Array[0][1][bs][s]*Array[0][2][bs][s_prime] + Array[0][3][bs][s_prime]
                            total_sum += h_value * Z
                        if t< s and t<s_prime:
                            h_value = h(s, T+(t-s_prime), f, g, f_prime, g_prime)
                            Array = ISS(x_data, k) 
                            Z = Array[0][0][bs][s] + Array[0][1][bs][s]*Array[0][2][bs][s_prime] + Array[0][3][bs][s_prime]
                            total_sum += h_value * Z
                        if t>= s and t<s_prime:
                            h_value = h(s, T+(t-s_prime), f, g, f_prime, g_prime)
                            Array = ISS(x_data, k) 
                            Z = Array[0][0][bs][s] + Array[0][1][bs][s]*Array[0][2][bs][s_prime] + Array[0][3][bs][s_prime]
                            total_sum += h_value * Z
                        if t< s and t>=s_prime:
                            h_value = h(s, t-s_prime, f, g, f_prime, g_prime)
                            Array = ISS(x_data, k) 
                            Z = Array[0][0][bs][s] + Array[0][1][bs][s]*Array[0][2][bs][s_prime] + Array[0][3][bs][s_prime]
                            total_sum += h_value * Z
                # Store the output
                out[bs][k][t] = 0.5*total_sum
    return out

def test_ISS():
    # torch.manual_seed(42)
    X = torch.randn(3, 2, 10)
    dX = torch.diff(X, dim=-1,prepend=torch.zeros(3,2,1))
    # dX = torch.tensor([[[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]]])
    ret = ISS( dX, 0 )
    # print(ret)
    a,b,c,d = ret[0]
    ##
    # k_{s,t} = a_s \otimes 1_t + b_s \otimes c_t + 1_s \otimes d_t
    ##
    # print(ret)
    retF = torch.zeros(3, 6)
    for i in range(4, dX.size(2)):
        for j in range(a.size(0)):
            retF[j][i-4] = (a[j][2] + b[j][2] * c[j][i] + d[j][i])

    # retF = d[:-1]

    # Test brute force:
    # ret2 = super_brute_force_two_letter_ISS(dX, [1, 0])
    # ret2 = ret2[:, 4:]
    # print(retF, 'RetF')
    # print(ret2, 'ret2')
    # assert torch.allclose(ret1, ret2)
    # assert torch.allclose( retF, ret2 ) 

    words = 4
    model = LowRankModel(words)   
    LowR1 = model(dX)

    # print('LowR1 = ', LowR1)
    LowR2 = super_brute_force_LowRank(dX, words)

    # print('LowR2 = ', LowR2)
    assert torch.allclose(LowR1, LowR2, atol=10**-3)

# if main
if __name__ == '__main__':
    test_ISS()