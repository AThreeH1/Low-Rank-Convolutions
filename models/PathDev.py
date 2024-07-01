import torch
import torch.nn as nn

class PathDevelopmentNetwork(nn.Module):
    def __init__(self, d):
        super(PathDevelopmentNetwork, self).__init__()
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
    model = PathDevelopmentNetwork(d)

    # Get the output
    output = model(Z)
    print(output.shape)  # Expected output shape: (total_batches, seq_length, d, d)
