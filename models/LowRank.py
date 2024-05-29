import os
import sys
import time

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now you can import your modules
from utils.imports import *
from data.datagenerator import DataGenerate
from Preprocessing.ISS import ISS

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FNNnew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNNnew, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the architectures for f, g, f', g'
input_size = 1  # Since T-s and T-s' are scalars
hidden_size = 5
output_size = 1

# Set the base seed
base_seed = 42

# Instantiate separate models for f, g, f', g' with different initializations
set_seed(base_seed)
f = FNNnew(input_size, hidden_size, output_size)

set_seed(base_seed + 1)
g = FNNnew(input_size, hidden_size, output_size)

set_seed(base_seed + 2)
f_prime = FNNnew(input_size, hidden_size, output_size)

set_seed(base_seed + 3)
g_prime = FNNnew(input_size, hidden_size, output_size)

def convolve_sequences(h, f):
    # Determine the length of the result
    N = len(h)
    
    # Compute the FFT of both sequences
    H = np.fft.fft(h, N)
    F = np.fft.fft(f, N)
    
    # Element-wise multiplication in the frequency domain
    Y = H * F
    
    # Compute the inverse FFT to get the convolved result
    y = np.fft.ifft(Y)
    
    # Take the real part to remove any small imaginary parts due to numerical errors
    y = np.real(y)
    
    return y

def LowRank(x_data):


    T = len(x_data[0][0])
    Total_batches = len(x_data)
    output_seq = [[] for _ in range(Total_batches)]

    for k in range(2):

        Array_LowRank = ISS(x_data, Total_batches, a = k)

        for j in range(Total_batches):
            f_arr = []
            f_prime_arr = []
            g_arr = []
            g_prime_arr = []
            Ones = [1 for i in range(T)]

            for i in range(T):
                f_arr.append(f(torch.tensor([[i]], dtype=torch.float32)).detach())
                f_prime_arr.append(f_prime(torch.tensor([[i]], dtype=torch.float32)).detach())
                g_arr.append(g(torch.tensor([[i]], dtype=torch.float32)).detach())
                g_prime_arr.append(g_prime(torch.tensor([[i]], dtype=torch.float32)).detach())

            f_arr = [tensor.item() for t in f_arr for tensor in t]
            f_prime_arr = [tensor.item() for t in f_prime_arr for tensor in t]
            g_arr = [tensor.item() for t in g_arr for tensor in t]
            g_prime_arr = [tensor.item() for t in g_prime_arr for tensor in t]
            Array_LowRank[j][1] = [element.item() if isinstance(element, torch.Tensor) else element for element in Array_LowRank[j][1]]
            Array_LowRank[j][2] = [element.item() if isinstance(element, torch.Tensor) else element for element in Array_LowRank[j][2]]
            Array_LowRank[j][3] = [element.item() if isinstance(element, torch.Tensor) else element for element in Array_LowRank[j][3]]
            Array_LowRank[j][4] = [element.item() if isinstance(element, torch.Tensor) else element for element in Array_LowRank[j][4]]
            
            AA = convolve_sequences(f_arr, Array_LowRank[j][1])
            AB = convolve_sequences(g_arr, Ones)
            BA = convolve_sequences(f_prime_arr, Array_LowRank[j][1])
            BB = convolve_sequences(g_prime_arr, Ones)
            CA = convolve_sequences(f_arr, Array_LowRank[j][2])
            CB = convolve_sequences(g_prime_arr, Array_LowRank[j][3])
            DA = convolve_sequences(f_prime_arr, Array_LowRank[j][2])
            DB = convolve_sequences(g_prime_arr, Array_LowRank[j][3])
            EA = convolve_sequences(f_arr, Ones)
            EB = convolve_sequences(g_arr, Array_LowRank[j][4])
            FA = convolve_sequences(f_prime_arr, Ones)
            FB = convolve_sequences(g_prime_arr, Array_LowRank[j][4])

            output_seq[j].append((AA * AB) + (BA * BB) + (CA * CB) + (DA * DB) + (EA * EB) + (FA * FB))

    return (torch.tensor(np.array(output_seq), dtype=torch.float32))




# def h(T_s, T_s_prime, f, g, f_prime, g_prime):

#     T_s = torch.tensor([[T_s]], dtype=torch.float32) 
#     T_s_prime = torch.tensor([[T_s_prime]], dtype=torch.float32)  
    
#     # Compute f(T-s), g(T-s'), f'(T-s), g'(T-s')
#     f_T_s = f(T_s)
#     g_T_s_prime = g(T_s_prime)
#     f_prime_T_s = f_prime(T_s)
#     g_prime_T_s_prime = g_prime(T_s_prime)
    
#     # Compute h(T-s, T-s') = f(T-s)g(T-s') + f'(T-s)g'(T-s')
#     h_value = f_T_s * g_T_s_prime + f_prime_T_s * g_prime_T_s_prime
#     # print('A', f_T_s , g_T_s_prime, 'B',f_prime_T_s , g_prime_T_s_prime)
#     return h_value.item()

# time1_start = time.time()

# total_sum = 0.0
# for s in range(T):
#     for s_prime in range(T):

#         h_value = h(T-s, T-s_prime, f, g, f_prime, g_prime)
#         Array = ISS(x_data, Total_batches, l = s, r = s_prime)
#         Z = Array[0]
#         total_sum += h_value * Z

# print(total_sum.item())

# time1_end = time.time()

# time2_start = time.time()

# Array_LowRank = ISS(x_data, Total_batches, l = 1, r = 1)

# f_arr = []
# f_prime_arr = []
# g_arr = []
# g_prime_arr = []

# for i in range(T):
#     f_arr.append(f(torch.tensor([[T-i]], dtype=torch.float32)).detach())
#     f_prime_arr.append(f_prime(torch.tensor([[T-i]], dtype=torch.float32)).detach())
#     g_arr.append(g(torch.tensor([[T-i]], dtype=torch.float32)).detach())
#     g_prime_arr.append(g_prime(torch.tensor([[T-i]], dtype=torch.float32)).detach())

# f_ISS = [x * y for x, y in zip(f_arr, Array_LowRank[1])]
# f_prime_ISS = [x * y for x, y in zip(f_prime_arr, Array_LowRank[1])]
# f_D1 = [x * y for x, y in zip(f_arr, Array_LowRank[2])]
# f_prime_D1 = [x * y for x, y in zip(f_prime_arr, Array_LowRank[2])]
# g_D2 = [x * y for x, y in zip(g_arr, Array_LowRank[3])]
# g_prime_D2 = [x * y for x, y in zip(g_prime_arr, Array_LowRank[3])]
# g_ISS = [x * y for x, y in zip(g_arr, Array_LowRank[4])]
# g_prime_ISS = [x * y for x, y in zip(g_prime_arr, Array_LowRank[4])]
# g_arr_sum = g_arr.copy()
# g_prime_arr_sum = g_prime_arr.copy()


# for i in range(1,T):
#     # f_ISS[i] += f_ISS[i-1]
#     # f_prime_ISS[i] += f_prime_ISS[i-1]
#     # f_D1[i] += f_D1[i-1]
#     # f_prime_D1[i] += f_prime_D1[i-1]
#     g_D2[-i-1] += g_D2[-i]
#     g_prime_D2[-i-1] += g_prime_D2[-i]
#     g_ISS[-i-1] += g_ISS[-i]
#     g_prime_ISS[-i-1] += g_prime_ISS[-i]    
#     g_arr_sum[-i-1] += g_arr_sum[-i]   
#     g_prime_arr_sum[-i-1] += g_prime_arr_sum[-i]

# Result = 0.0
# for s in range(T-2):
#     Result += (f_ISS[s]*g_arr_sum[s+2] 
#                 + f_prime_ISS[s]*g_prime_arr_sum[s+2] 
#                 + f_D1[s]*g_D2[s+2] 
#                 + f_prime_D1[s]*g_prime_D2[s+2] 
#                 + f_arr[s]*g_ISS[s+2] 
#                 + f_prime_arr[s]*g_prime_ISS[s+2])

# print(Result.item())

# time2_end = time.time()
# time1 = time1_end - time1_start
# time2 = time2_end - time2_start
# print("Time by T^2:", time1)
# print("Time by TlogT:", time2)