# HyenaVsMHA
An experiment designed to test Hyena Hierarchy with respect to the Multi-Head Attention Model. Training of Hyena is done with the target set as outputs from MHA(Multi-Head Attention) at different orders to see at what parameters the Hyena performs similar to MHA, given that MHA is a quadratic time computation while Hyena is semi-quadratic.

To run the code provided in the repository please follow the following steps:
1. Run Imports.py
2. Run MultiHeadAttention.py
3. Run HyenaLightning.py
4. Run Analysis.py by changing the appropriate values of hyperparameters such as order, learning rate, optimizer, etc. Log into wandb account to log gradients, losses, errors, etc.

For hyena convolution over low-rank functions: \
Download the data here - https://drive.google.com/file/d/1amypY8KfLQP5Fu2X2pTBnKUtip1akUT3/view?usp=sharing \
Use the script LowRankFunction.py with the appropriate path for the data
