import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now you can import your modules
from utils.imports import *
from models.DevLayer import development_layer
from models.Devnet import so

class PathDev(nn.Module):
    def __init__(self, d, liegroup):

        """
        X_{s-1,s} = exp(Sum_{i = 0 to dims}(A_i * Z_i[s]))
        X_{0,s} = X_{0,s-1} * X_{s-1,s}

        args:
            d = length/height of matrix X
            liegroup = Matrix Lie group in use

        Variables in code:
            A1 and A2 = parameterised matrices
            T = Sequence length
            A & B = X & X^-1 resp.

        output:
            List of two sequences of size (bs, T, d, d), first one being the X and second being X^-1

        """

        super(PathDev, self).__init__()
        self.d = d
        self.liegroup = liegroup
        # Initialize two learnable dxd matrices
        # self.A1 = nn.Parameter(torch.tensor([[1.00,2.00,3.00],[4.00,5.00,6.00],[7.00,8.00,9.00]]))
        # self.A2 = nn.Parameter(torch.tensor([[1.00,2.00,3.00],[4.00,5.00,6.00],[7.00,8.00,9.00]]))     
        self.A1 = nn.Parameter(torch.randn(d, d))
        self.A2 = nn.Parameter(torch.randn(d, d))

    def SE(self, X):
        X = torch.diff(X, dim=-1,prepend=torch.zeros(bs,dims,1, device = X.device))
        a, b, d, d = X.shape
        so = X[..., :-1, :-1] - X[..., :-1, :-1].T

        X = torch.cat(
            (torch.cat((so, X[..., :-1, -1].unsqueeze(-1)), dim=3
                    ), torch.zeros((a, b, 1, d)).to(X.device)), dim=2)

        return X

    def SP(X: torch.tensor, J: torch.tensor) -> torch.tensor:
        """ symplectic transformation

        Args:
            X (torch.tensor): (...,2n,2n)
            J (torch.tensor): (2n,2n), symplectic operator [[0,I],[-I,0]]

        Returns:
            torch.tensor: (...,2n,2n)
        """
        X = (X + X.transpose(-2, -1))/2

        return X.matmul(J.T)

    def SO(self, X):
        X = X - X.T
        return X

    def forward(self, Z):
        batch_size, seq_length, dimension = Z.size()
        assert dimension == 2, "Dimension should be 2 to match A1 and A2 matrices."
        
        # Ensure A1 and A2 are anti-symmetric
        if self.liegroup == 'SO':
            A1 = self.SO(self.A1)
            A2 = self.SO(self.A2)

        elif self.liegroup == 'SE':
            A1 = self.SE(self.A1)
            A2 = self.SE(self.A2)   

        elif self.linalg == 'SP':
            N = (A1.size(-1))/2
            if N%2 == 0:
                neg_I = -torch.diag(torch.ones(int(N/2)))
                pos_I = torch.diag(torch.ones(int(N/2)))
                J = torch.zeros(N, N)
                J[:int(N/2), int(N/2):] = pos_I
                J[int(N/2):, :int(N/2)] = neg_I    
                A1 = self.SP(self.A1, J)
                A2 = self.SP(self.A2, J)
            else:
                raise ValueError(
                    'size of symplectic lie algebra matrix needs to be an even number')     
        
        else:
            return "Undefined lie group"
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

        self.a = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x):
        z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        y = torch.exp(self.a * x)
        return z*y


class PathDevelopmentNetwork(nn.Module):
    def __init__(self, d, liegroup):
        """
        Sum_{s,s'}(h(t-s, t-s') * X_{s,s'})
        h(s,s') = f(s).g(s') + f'(s).g'(s')
        X_{s,s'} = (X_{0,s})^-1 * X_{0,s'}

        args:
            d = length/height of matrix X

        Variables in code:
            bs = Batch size
            T = Sequence length
            AA, AB, BA, etc = Low rank components

        output:
            Sequence of size (bs, T, d, d)

        """
        super(PathDevelopmentNetwork, self).__init__()
        
        # Initialize the FFN models 
        self.f = FNNnew()
        self.g = FNNnew()
        self.f_prime = FNNnew()
        self.g_prime = FNNnew()

        self.path_dev = PathDev(d, liegroup)
        self.d = d
        self.liegroup = liegroup

    def forward(self, x_data):
        d = self.d
        T = x_data.size(1)
        bs = x_data.size(0)

        Time = torch.arange(T, dtype=torch.float32).view(-1, 1).to(x_data.device)

        f_arr = self.f(Time).squeeze()
        f_prime_arr = self.f_prime(Time).squeeze()
        g_arr = self.g(Time).squeeze()
        g_prime_arr = self.g_prime(Time).squeeze()     
        
        f_arr = torch.nn.functional.pad(f_arr, (T-1, T-1))
        f_prime_arr = torch.nn.functional.pad(f_prime_arr, (T-1, T-1))
        g_arr = torch.nn.functional.pad(g_arr, (T-1, T-1))
        g_prime_arr = torch.nn.functional.pad(g_prime_arr, (T-1, T-1))

        H_f_arr = torch.fft.fft(f_arr, dim=0)
        H_f_prime_arr = torch.fft.fft(f_prime_arr, dim=0)
        H_g_arr = torch.fft.fft(g_arr, dim=0)
        H_g_prime_arr = torch.fft.fft(g_prime_arr, dim=0)

        tensor = self.path_dev(x_data)

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

            AA_pad = torch.nn.functional.pad(final_X_inv[i].squeeze(), (T-1, T-1))
            AB_pad = torch.nn.functional.pad(final_X[i].squeeze(), (T-1, T-1))

            AA_fft = torch.fft.fft(AA_pad, dim = -1)
            AB_fft = torch.fft.fft(AB_pad, dim = -1)

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
        final = concatenated_tensor.view(a, b, d, d)

        return final.real


def brute_force_path_dev(Z):
    A1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    A2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) 
    A1 = A1 - A1.T 
    A2 = A2 - A2.T
    X = torch.matrix_exp(A1 * (Z[:,1,0].unsqueeze(-1).unsqueeze(-1)) + A2 * (Z[:,1,1].unsqueeze(-1).unsqueeze(-1)))
    return X


# Example usage
if __name__ == "__main__":
    d = 4
    total_batches = 3
    seq_length = 100
    dimension = 2
    
    torch.manual_seed(42)
    # Create a sample input signal
    Z = torch.randn(total_batches, seq_length, dimension)

    # Initialize the model
    model = PathDevelopmentNetwork(d, 'SO')

    # Get the output
    output = model(Z)

    print('out', output.shape)  # Expected output shape: (total_batches, seq_length, d, d)

    X1_brute = brute_force_path_dev(Z)
    # print(X1_brute)
    path_dev = PathDev(d, 'SO')
    X1_path_dev = path_dev(Z)[0][:,1,:,:]
    # print(X1_path_dev)

    dev = development_layer(
        input_size=seq_length, hidden_size=d, param=so, return_sequence=False)
    out = dev(Z)
    print(out)

    # Please set A1 and A2 to the commented tensors before running the assertion 
    # assert torch.allclose(X1_brute, X1_path_dev)