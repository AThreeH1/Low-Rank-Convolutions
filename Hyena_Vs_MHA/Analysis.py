#!/usr/bin/env python
# coding: utf-8

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
x = torch.randn((batch_size, sequence_length, input_dim), requires_grad=True).to(device)

# Generate training data
torch.manual_seed(123)
model_mha = MultiheadAttention(input_dim, d_model, num_heads).to(device)
out = model_mha.forward(x)
out = out.detach()
x = x.detach()

for i in [2,4,8,16,32,64,128]:

    # Initialize HyenaOperator model
    hyena_model = HyenaOperator(d_model=d_model, l_max=sequence_length, order=i, dropout=0.0, filter_dropout=0.0)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    # initiate wandb logger
    wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")

    # Initialize Trainer and start training
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=11,
        logger = wandb_logger,
        log_every_n_steps=1,
    )

    wandb_logger.watch(hyena_model, log="gradients", log_freq=50, log_graph=True)

    trainer.fit(hyena_model)
    trainer.test()

    wandb.finish()

