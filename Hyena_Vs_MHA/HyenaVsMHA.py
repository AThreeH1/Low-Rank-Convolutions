#!/usr/bin/env python
# coding: utf-8

# In[ ]:


SeqD = 512
InpD = OutD = SeqD
KernalS = 3
Bytes = 4
Order = 128
BatchS = 20

Memory = (InpD*OutD*KernalS + OutD)*Bytes*Order*BatchS
print('Approx ' + str(Memory/1000000000) + ' GB should used')


# In[ ]:


# Assigning Values
device = torch.device("cuda")
input_dim = 512
d_model = 512
num_heads = 8
batch_size = 1000
sequence_length = 128
train_ratio = 0.8

# Initialising random input
x = torch.randn( (batch_size, sequence_length, input_dim), requires_grad = True ).to(device)

# Generate training data
torch.manual_seed(123)
model_mha = MultiheadAttention(input_dim, d_model, num_heads).to(device)
out = model_mha.forward(x)


# In[ ]:

class MyReplicatorModule(pl.LightningModule):
    # TODO
    pass
    # model = HyenaOperator(
    #     d_model=d_model,
    #     l_max=sequence_length,
    #     order=i,
    #     filter_order=64
    # ).to(device)


for i in [2,4,8,16,32,64,128]:

    # Instantiating model
    model = HyenaOperator(
        d_model=d_model,
        l_max=sequence_length,
        order=i,
        filter_order=64
    ).to(device)

    # starting a new wandb run to track this script
    wandb.init(

        project="HyenaStats",

        config={
        "learning_rate": 0.001,
        "architecture": "Hyena",
        "dataset": "MHA Outputs",
        "epochs": 10,
        }
    )

    wandb.watch(model, log_freq=100)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Performing train-test split
    total_data = TensorDataset(x, out)
    train_size = int(0.8 * len(total_data))
    test_size = len(total_data) - train_size
    train_dataset, test_dataset = random_split(total_data, [train_size, test_size])

    # DataLoader for training and testing
    batch_size = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            wandb.log({"loss": loss, "epoch": epoch})

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

    # Testing the model
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            mse = criterion(outputs, targets)
            total_mse += mse.item()

        average_mse = total_mse / len(test_loader)
        wandb.log({"error": average_mse})
        print('Mean Squared Error on the test set: {:.4f}'.format(average_mse))

    wandb.finish()

