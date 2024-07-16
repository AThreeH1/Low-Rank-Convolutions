import torch
import torch.nn as nn

device = torch.device('cuda')

class PathDev(nn.Module):
    def __init__(self, d):
        super(PathDev, self).__init__()
        self.d = d
        # Initialize two learnable 3x3 matrices
        # self.A1 = nn.Parameter(torch.tensor([[1.00,2.00,3.00],[4.00,5.00,6.00],[7.00,8.00,9.00]]))
        # self.A2 = nn.Parameter(torch.tensor([[1.00,2.00,3.00],[4.00,5.00,6.00],[7.00,8.00,9.00]]))     
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

        Z1 = Z[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, seq_length, 1)
        Z2 = Z[:, :, 1].unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, seq_length, 1)

        # Compute the matrix multiplications
        term1 = A1 * Z1  # Shape (batch_size, seq_length-1, d, d)
        term2 = A2 * Z2  # Shape (batch_size, seq_length-1, d, d)
        sum_term = torch.matrix_exp(term1 + term2)  # Shape (batch_size, seq_length-1, d, d)

        # Compute X_t inside the loop using precomputed sum_term
        for t in range(1, seq_length):  # Adjust range since we start from 1
            # Compute the next X_t
            X_t = torch.bmm(X, sum_term[:, t])
            outputs.append(X_t)
            X = X_t


        A = torch.stack(outputs, dim=1)
        B = torch.linalg.inv(A)
         
        return [A, B]


# Define the FNN class
class FNNnew(nn.Module):
    def __init__(self):
        super(FNNnew, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)
    
        # nn.init.zeros_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.zeros_(self.fc2.weight)
        
        # # Bias of the last layer = 1
        # nn.init.ones_(self.fc2.bias)

        self.a = -5

    def forward(self, x):
        z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        y = torch.exp(self.a * x)
        return z*y


class PathDevelopmentNetwork(nn.Module):
    def __init__(self, d):
        super(PathDevelopmentNetwork, self).__init__()
        
        # Initialize the FFN models 
        self.f = FNNnew().to(device)
        # set_seed(self.f)
        self.g = FNNnew().to(device)
        # set_seed(self.g)
        self.f_prime = FNNnew().to(device)
        # set_seed(self.f_prime)
        self.g_prime = FNNnew().to(device)
        # set_seed(self.g_prime)

        self.path_dev = PathDev(d).to(device)
        self.d = d

    def forward(self, x_data):
        d = self.d
        T = x_data.size(1)
        bs = x_data.size(0)
        f_arr = self.f(torch.arange(T, dtype=torch.float32).view(-1, 1).to(device)).squeeze().flip(0)
        f_prime_arr = self.f_prime(torch.arange(T, dtype=torch.float32).view(-1, 1).to(device)).squeeze().flip(0)
        g_arr = self.g(torch.arange(T, dtype=torch.float32).view(-1, 1).to(device)).squeeze()
        g_prime_arr = self.g_prime(torch.arange(T, dtype=torch.float32).view(-1, 1).to(device)).squeeze()       
        
        f_arr = torch.nn.functional.pad(f_arr, (T-1, T-1))
        f_prime_arr = torch.nn.functional.pad(f_prime_arr, (T-1, T-1))
        g_arr = torch.nn.functional.pad(g_arr, (T-1, T-1))
        g_prime_arr = torch.nn.functional.pad(g_prime_arr, (T-1, T-1))

        H_f_arr = torch.fft.fft(f_arr, dim=0)
        H_f_prime_arr = torch.fft.fft(f_prime_arr, dim=0)
        H_g_arr = torch.fft.fft(g_arr, dim=0)
        H_g_prime_arr = torch.fft.fft(g_prime_arr, dim=0)

        tensor = self.path_dev(x_data.to(device))

        X = tensor[0]
        X_inv = tensor[1]
        a,b,c,_ = X.shape
        reshaped_X = X.view(a, b, d**2)
        reshaped_X_inv = X_inv.reshape(a, b, d**2)

        #Split tensor along the third dimension into d**2 tensors
        sliced_X = torch.split(reshaped_X, 1, dim=2)
        sliced_X_inv = torch.split(reshaped_X_inv, 1, dim=2)

        # Reshape each sliced tensor
        final_X = [t.view(a, b, 1) for t in sliced_X]
        final_X_inv = [t.view(a, b, 1) for t in sliced_X_inv]

        output = []

        for i in range(len(final_X)):
            AA_fft_np = torch.fft.fft(final_X_inv[i].squeeze(), dim = -1)
            AB_fft_np = torch.fft.fft(final_X[i].squeeze(), dim = -1)

            AA_fft = torch.nn.functional.pad(AA_fft_np, (T-1, T-1))
            AB_fft = torch.nn.functional.pad(AB_fft_np, (T-1, T-1))

            AA_mul = AA_fft * H_f_arr
            AB_mul = AB_fft * H_g_arr
            BA_mul = AA_fft * H_f_prime_arr
            BB_mul = AB_fft * H_g_prime_arr

            AA = torch.fft.ifft(AA_mul, dim = -1)
            AB = torch.fft.ifft(AB_mul, dim = -1)
            BA = torch.fft.ifft(BA_mul, dim = -1)
            BB = torch.fft.ifft(BB_mul, dim = -1)

            out = ((AA*AB) + (BA*BB))
            output.append(out[:,(2*T-2):])

        # Unsqueeze each tensor back 
        output_f = [t.unsqueeze(2) for t in output]

        # Concatenate the list of tensors along the third dimension
        concatenated_tensor = torch.cat(output_f, dim=2)

        # Reshape the concatenated tensor 
        final = concatenated_tensor.view(a, b, 3, 3)

        return final.real


def brute_force_path_dev(Z):
    A1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    A2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) 
    A1 = A1 - A1.T 
    A2 = A2 - A2.T
    return torch.matrix_exp(A1 * (Z[:,1,0].unsqueeze(-1).unsqueeze(-1)) + A2 * (Z[:,1,1].unsqueeze(-1).unsqueeze(-1)))


# Example usage
if __name__ == "__main__":
    d = 3
    total_batches = 3
    seq_length = 100
    dimension = 2
    
    torch.manual_seed(42)
    # Create a sample input signal
    Z = torch.randn(total_batches, seq_length, dimension)

    # Initialize the model
    model = PathDevelopmentNetwork(d)

    # Get the output
    output = model(Z)

    print('out', output.shape)  # Expected output shape: (total_batches, seq_length, d, d)

    X1_brute = brute_force_path_dev(Z)
    # print(X1_brute)
    path_dev = PathDev(d)
    X1_path_dev = path_dev(Z)[0][:,1,:,:]
    # print(X1_path_dev)


    # Please set A1 and A2 to the manual tensors before running the assertion 
    # assert torch.allclose(X1_brute, X1_path_dev)