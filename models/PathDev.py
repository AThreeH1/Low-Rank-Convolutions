import torch
import torch.nn as nn

class PathDev(nn.Module):
    def __init__(self, d):
        super(PathDev, self).__init__()
        self.d = d
        # Initialize two learnable 3x3 matrices
        self.A1 = nn.Parameter(torch.randn(d, d))
        self.A2 = nn.Parameter(torch.randn(d, d))

    def forward(self, Z):
        batch_size, seq_length, dimension = Z.size()
        assert dimension == 2, "Dimension should be 2 to match A1 and A2 matrices."
        # Ensure A1 and A2 are anti-symmetric
        A1 = self.A1 - self.A1.T
        A2 = self.A2 - self.A2.T
        # Initialize X_0 as the identity matrix
        X = torch.eye(self.d).unsqueeze(0).repeat(batch_size, 1, 1).to(Z.device)
        # Output container
        outputs = [X]

        for t in range(1, seq_length):
            Z1_t = Z[:, t, 0].unsqueeze(1).unsqueeze(2)  # Shape (batch_size, 1, 1)
            Z2_t = Z[:, t, 1].unsqueeze(1).unsqueeze(2)  # Shape (batch_size, 1, 1)
            
            # Compute the matrix multiplications
            term1 = A1 * Z1_t
            term2 = A2 * Z2_t
            sum_term = term1 + term2  # Shape (batch_size, d, d)
            
            # Compute the next X_t
            X_t = torch.bmm(X, sum_term)
            outputs.append(X_t)
            X = X_t
        
        return torch.stack(outputs, dim=1)

# Example usage
if __name__ == "__main__":
    d = 3
    total_batches = 1
    seq_length = 100
    dimension = 2

    # Create a sample input signal
    Z = torch.randn(total_batches, seq_length, dimension)

    # Initialize the model
    model = PathDev(d)

    # Get the output
    output = model(Z)
    print(output.shape)  # Expected output shape: (total_batches, seq_length, d, d)


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
        

class PathDevelopmentNetwork(nn.Module):
    def __init__(self, d):
        super(LowRankModel, self).__init__()
        
        # Initialize the FFN models 
        self.f = FNNnew().to(device)
        # set_seed(self.f)
        self.g = FNNnew()
        # set_seed(self.g)
        self.f_prime = FNNnew()
        # set_seed(self.f_prime)
        self.g_prime = FNNnew()
        # set_seed(self.g_prime)
        self.d = d

    def forward(self, x_data):
        d = self.d
        T = x_data.size(2)
        bs = x_data.size(0)
        f_arr = self.f(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(device)).squeeze().flip(0)
        f_prime_arr = self.f_prime(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(device)).squeeze().flip(0)
        g_arr = self.g(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(device)).squeeze()
        g_prime_arr = self.g_prime(torch.arange(T+1, dtype=torch.float32).view(-1, 1).to(device)).squeeze()       
        
        f_arr = torch.nn.functional.pad(f_arr, (T, T))
        f_prime_arr = torch.nn.functional.pad(f_prime_arr, (T, T))
        g_arr = torch.nn.functional.pad(g_arr, (T, T))
        g_prime_arr = torch.nn.functional.pad(g_prime_arr, (T, T))
        ones = torch.nn.functional.pad(ones, (T, T))

        H_f_arr = torch.fft.fft(f_arr, dim=0)
        H_f_prime_arr = torch.fft.fft(f_prime_arr, dim=0)
        H_g_arr = torch.fft.fft(g_arr, dim=0)
        H_g_prime_arr = torch.fft.fft(g_prime_arr, dim=0)
        H_ones = torch.fft.fft(torch.ones(T+1, device=device), dim=0).reshape(1, T+1)

        output_seq = torch.zeros((bs, words, 3*T+1), device=device, dtype=torch.complex64)

        for k in range(words):
            Array_LowRank = ISS(x_data, a=k)

            # Convert list of lists to tensor for batch processing
            ISS_sum = torch.stack([item[0] for item in Array_LowRank], dim=0).to(device)
            Consicutive_Sum_D1 = torch.stack([item[1] for item in Array_LowRank], dim=0).to(device)
            Consicutive_Sum_D2 = torch.stack([item[2] for item in Array_LowRank], dim=0).to(device)
            ISS_D1 = torch.stack([item[3] for item in Array_LowRank], dim=0).to(device) #[:,:,1:]

            ISS_sum = torch.nn.functional.pad(ISS_sum, (T, T))
            Consicutive_Sum_D1 = torch.nn.functional.pad(Consicutive_Sum_D1, (T, T))
            Consicutive_Sum_D2 = torch.nn.functional.pad(Consicutive_Sum_D2, (T, T))
            ISS_D1 = torch.nn.functional.pad(ISS_D1, (T, T))
            
            # Apply FFT to all elements in the batch
            ISS_sum_fft = torch.fft.fft(ISS_sum, dim=-1)
            Consicutive_Sum_D1_fft = torch.fft.fft(Consicutive_Sum_D1, dim=-1)
            Consicutive_Sum_D2_fft = torch.fft.fft(Consicutive_Sum_D2, dim=-1)
            ISS_D1_fft = torch.fft.fft(ISS_D1, dim=-1)
            
            # ISS_sum_fft = torch.nn.functional.pad(ISS_sum_fft, (T, T))
            # Consicutive_Sum_D1_fft = torch.nn.functional.pad(Consicutive_Sum_D1_fft, (T, T))
            # Consicutive_Sum_D2_fft = torch.nn.functional.pad(Consicutive_Sum_D2_fft, (T, T))
            # ISS_D1_fft = torch.nn.functional.pad(ISS_D1_fft, (T, T))

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