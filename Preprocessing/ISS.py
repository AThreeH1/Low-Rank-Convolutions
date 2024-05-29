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

device = torch.device('cuda')
def ISS(x_data, Total_batches, a):

    out = []
    for j in range(Total_batches):

        D1, D2 = (x_data[j][0], x_data[j][1]) if a == 1 else (x_data[j][1], x_data[j][0])         
        T = len(D1) 
        # Compute Consicutive Sums for D1 and D2
        Consicutive_Sum_D1 = torch.cat([torch.tensor([0]).to(device), torch.cumsum(D1, dim=0)])
        Consicutive_Sum_D2 = torch.cat([torch.tensor([0]).to(device), torch.cumsum(D2, dim=0)])
        Consicutive_Sum_D1 = Consicutive_Sum_D1[:-1]
        Consicutive_Sum_D2 = Consicutive_Sum_D2[:-1]
        # Same indexed sums in QS
        QS = D1 * D2
        ISS_QS = torch.cat([torch.tensor([0]).to(device), torch.cumsum(QS, dim=0)])
        ISS_QS = ISS_QS[:-1]
        ISS_D1 = torch.zeros(T).to(device)
        ISS_D2 = torch.zeros(T).to(device)
        for i in range(2, T):
            ISS_D1[i] = ((ISS_D1[i - 1] + Consicutive_Sum_D1[i - 1]) * D2[i - 1]).item()
            ISS_D2[i] = ((ISS_D2[i - 1] + Consicutive_Sum_D2[i - 1]) * D1[i - 1]).item()

        negated_Consicutive_Sum_D1 = -Consicutive_Sum_D1
        ISS_sum = torch.add(ISS_D2, ISS_QS)
        out.append([torch.tensor(0), ISS_sum, negated_Consicutive_Sum_D1, Consicutive_Sum_D2, ISS_D1])
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
def convolve_sequences(h, f):

    H = torch.fft.fft(h, dim=0)
    F = torch.fft.fft(f, dim=0)
    Y = H * F
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
        device = x_data.device

        output_seq = [[] for _ in range(Total_batches)]

        for k in range(2):
            Array_LowRank = ISS(x_data, Total_batches, a=k)

            Ones = torch.ones(T, device=device)

            f_arr = self.f(torch.arange(T, dtype=torch.float32).view(-1, 1)).detach().squeeze().to(device)
            f_prime_arr = self.f_prime(torch.arange(T, dtype=torch.float32).view(-1, 1)).detach().squeeze().to(device)
            g_arr = self.g(torch.arange(T, dtype=torch.float32).view(-1, 1)).detach().squeeze().to(device)
            g_prime_arr = self.g_prime(torch.arange(T, dtype=torch.float32).view(-1, 1)).detach().squeeze().to(device)

            for j in range(Total_batches):

                Array_LowRank[j][1] = Array_LowRank[j][1].to(device)
                Array_LowRank[j][2] = Array_LowRank[j][2].to(device)
                Array_LowRank[j][3] = Array_LowRank[j][3].to(device)
                Array_LowRank[j][4] = Array_LowRank[j][4].to(device)

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
            
        return torch.stack([torch.stack(batch) for batch in output_seq]).to(device)

# Total_batches = 2
# sequence_length = 100
# dim = 2
# data = DataGenerate(Total_batches, sequence_length, dim)

# x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
# x_data = x_datax.permute(0, 2, 1)
# x_data = torch.tensor(x_data, dtype=torch.float32)
# labels = torch.tensor([data_point[-1] for data_point in data])

# LowRank = LowRankModel()
# print(LowRank(x_data))