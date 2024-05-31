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

device = torch.device('cpu')

# Implementing ISS
def ISS(x_data, Total_batches, a):
    out = []
    for j in range(Total_batches):
        D1, D2 = (x_data[j][0], x_data[j][1]) if a == 1 else (x_data[j][1], x_data[j][0])         
        T = len(D1) 
        # Compute Consicutive Sums for D1 and D2
        Consicutive_Sum_D1 = torch.cat([torch.zeros(1, device=device), torch.cumsum(D1, dim=0)])[:-1]
        Consicutive_Sum_D2 = torch.cat([torch.zeros(1, device=device), torch.cumsum(D2, dim=0)])[:-1]
        # Same indexed sums in QS
        QS = D1 * D2
        ISS_QS = torch.cat([torch.zeros(1, device=device), torch.cumsum(QS, dim=0)])[:-1]
        ISS_D1 = torch.zeros(T).to(device)
        ISS_D2 = torch.zeros(T).to(device)

        for i in range(2, T):
            ISS_D1[i] = ((ISS_D1[i - 1] + Consicutive_Sum_D1[i - 1]) * D2[i - 1])
            ISS_D2[i] = ((ISS_D2[i - 1] + Consicutive_Sum_D2[i - 1]) * D1[i - 1])

        negated_Consicutive_Sum_D1 = -Consicutive_Sum_D1
        ISS_sum = ISS_D2 + ISS_QS
        out.append([ISS_sum, negated_Consicutive_Sum_D1, Consicutive_Sum_D2, ISS_D1])
    return out



# Define the FNN class
class FNNnew(nn.Module):
    def __init__(self):
        super(FNNnew, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to perform convolution
def convolve_sequences(h_fft, f_fft):
    Y = h_fft * f_fft
    y = torch.fft.ifft(Y, dim=0)
    return y.real

# Define the LowRankModel class
class LowRankModel(nn.Module):
    def __init__(self):
        super(LowRankModel, self).__init__()
        
        # Initialize the FFN models 
        self.f = FNNnew()
        self.g = FNNnew()
        self.f_prime = FNNnew()
        self.g_prime = FNNnew()
    
    def forward(self, x_data):
        T = x_data.size(2)
        Total_batches = x_data.size(0)

        output_seq = [[] for _ in range(Total_batches)]

        f_arr = self.f(torch.arange(T, dtype=torch.float32).view(-1, 1)).to(device).squeeze()
        f_prime_arr = self.f_prime(torch.arange(T, dtype=torch.float32).view(-1, 1)).to(device).squeeze()
        g_arr = self.g(torch.arange(T, dtype=torch.float32).view(-1, 1)).to(device).squeeze()
        g_prime_arr = self.g_prime(torch.arange(T, dtype=torch.float32).view(-1, 1)).to(device).squeeze()

        H_f_arr = torch.fft.fft(f_arr, dim=0)
        H_f_prime_arr = torch.fft.fft(f_prime_arr, dim=0)
        H_g_arr = torch.fft.fft(g_arr, dim=0)
        H_g_prime_arr = torch.fft.fft(g_prime_arr, dim=0)
        H_ones = torch.fft.fft(torch.ones(T, device=device), dim=0)

        for k in range(2):
            Array_LowRank = ISS(x_data, Total_batches, a=k)

            for j in range(Total_batches):
                Array_LowRank[j] = [tensor.to(device) for tensor in Array_LowRank[j]]
                ISS_sum_fft = torch.fft.fft(Array_LowRank[j][0], dim=0)
                Consicutive_Sum_D1_fft = torch.fft.fft(Array_LowRank[j][1], dim=0)
                Consicutive_Sum_D2_fft = torch.fft.fft(Array_LowRank[j][2], dim=0)
                ISS_D1_fft = torch.fft.fft(Array_LowRank[j][3], dim=0)

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

                output_seq[j].append((AA * AB) + (BA * BB) + (CA * CB) + (DA * DB) + (EA * EB) + (FA * FB))

        return torch.stack([torch.stack(batch) for batch in output_seq]).to(device)

# model = LowRankModel().to(device)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Total_batches = 1
# sequence_length = 100
# dim = 2
# data = DataGenerate(Total_batches, sequence_length, dim)

# x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
# x_data = x_datax.permute(0, 2, 1)
# x_data = torch.tensor(x_data, dtype=torch.float32)
# labels = torch.tensor([data_point[-1] for data_point in data])

# Arr_1 = ISS(x_data, Total_batches, 1)
# Arrf_1 = Arr_1[0][0] + Arr_1[0][2] + Arr_1[0][2] + Arr_1[0][3]
# Arr_2 = ISS(x_data, Total_batches, 2)
# Arrf_2 = Arr_2[0][0] + Arr_2[0][2] + Arr_2[0][2] + Arr_2[0][3]

# print(data[0][0])
# print(data[0][1])
# print(Arrf_1)
# print(Arrf_2)
# LowRank = LowRankModel()
# print(LowRank(x_data))