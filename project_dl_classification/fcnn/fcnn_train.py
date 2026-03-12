
import pandas as pd  

from fcnn_dataset import PartialDischargeDataset, collate_fn
from fcnn_model import FCNN
from fcnn_hyperparameters import in_channels, nr_featuremaps, kernel_size, batch_size, lr, epochs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


df = pd.read_parquet('data/processed/partial_discharge_data.parquet')

n1 = int(0.8 * len(df))
n2 = int(0.9 * len(df))

train_df = df.iloc[:n1].reset_index(drop=True)
val_df = df.iloc[n1:n2].reset_index(drop=True)
test_df = df.iloc[n2:].reset_index(drop=True)

dataset_train = PartialDischargeDataset(train_df)
dataset_val = PartialDischargeDataset(val_df)
dataset_test = PartialDischargeDataset(test_df)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = FCNN(in_channels=in_channels, nr_featuremaps=nr_featuremaps, kernel_size=kernel_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


val_iter = iter(val_loader)

lossi = []
tr_loss = []
vl_loss = []
running_loss = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to CUDA if available

model.train()
for epoch in range(epochs):
    for i, (x_tr, y_tr) in enumerate(train_loader):
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)  # Move data to CUDA if available
        optimizer.zero_grad()

        # Forward pass
        output = model(x_tr)

        # Compute loss
        loss = criterion(output, y_tr)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # track train and validation loss
        lossi.append(loss.item())
        running_loss += loss.item()

        if i % 100 == 99:
            tr_loss.append(running_loss / 100)
            model.eval()
            with torch.no_grad():
                # Check validation loss for batches
                # xval, yval = next(val_iter)
                # xval, yval = xval.to(device), yval.to(device)

                # Check validation loss for specific class
                xval, yval = dataset_val.get_sample_for_id(0)
                xval, yval = xval.view(1, 1, -1).to(device), yval.view(1, -1).to(device)

                output = model(xval)
                vl_loss.append((criterion(output, yval).item(), dataset_val.last_rdm_idx))

            print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {tr_loss[-1]:.3f} === val_loss: {vl_loss[-1][0]:.3f} for idx {vl_loss[-1][1]}')
            running_loss = 0.0
            model.train()

# Plot Losses
vl_loss = np.array(vl_loss)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(tr_loss)), y=tr_loss, name='Train loss'))
fig.add_trace(go.Scatter(x=np.arange(len(vl_loss)), y=vl_loss[:, 0], name='Val Loss',
                         text=['@ idx: ' + str(val) for val in vl_loss[:, 1]]))
fig.show()


# Accuracy
t_loss = []
correct = 0
total = 0
model.to(device)
model.eval()
with torch.no_grad():
    # for i, (x_te, y_te) in enumerate(train_loader):
    for i, (x_te, y_te) in enumerate(test_loader):
        x_te, y_te = x_te.to(device), y_te.to(device)
        total += y_te.size(0)
        output = model(x_te)
        t_loss.append(criterion(output, y_te).item())
        _, predicted = torch.max(output.data, 1)  # returns max values and indices along given dimension
        correct += (predicted == y_te).sum().item()
    val_accuracy = 100 * correct / total

print(f'Test Acc: {val_accuracy:.2f}%')

# Visualization

# Classification Activation Map - which parts where relevant for decision
model.cpu()
model.eval()

with torch.no_grad():
    x_te, y_te = dataset_test.get_sample_for_id(1)
    out, conv_out = model(x_te.view(1, 1, -1), True)
    weights = model.fcl.weight.data
    cam = (weights[torch.argmax(out)].reshape(-1, 1) * conv_out).sum(dim=1).flatten()
    print(F.softmax(out, dim=1))
    # print(out)
    print('gt:', y_te)
    print('pred:', torch.argmax(out))

fig = go.Figure()
fig.update_layout(title=f'Classification Activation Map: df gt: {y_te}; prediction: {torch.argmax(out)} for idx {dataset_test.last_rdm_idx}')
fig.add_trace(
    go.Scatter(
        x=np.arange(x_te.shape[0]),
        y=x_te,
        mode='lines+markers',
        marker=dict(color=cam, colorscale='Viridis', colorbar=dict(title='Contribution')),
        line=dict(color='rgba(0, 0, 0, .3)', width=2)
    )
)
fig.show()


# Kernels
n_conv_layers = 0
kernels = []

for layer in model.net:
    if isinstance(layer, nn.Conv1d):
        n_conv_layers += 1
        kernels += layer.weight.data  # .squeeze()

# Create subplots
fig = make_subplots(
    rows=n_conv_layers,
    cols=n_feature_maps,
    shared_xaxes=True,
    shared_yaxes=True,
    x_title='Kernels',
    y_title='Input channels'
)

# Populate subplots with Kernels
for i in range(layer.in_channels):
    for j in range(n_feature_maps):
        x_values = np.arange(kernel_size)
        y_values = kernels[i].flatten() if i == 0 else kernels[i][j].flatten()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                line=dict(color='rgba(0,0,0,.6)'),
                name=f'kernel {i}{j}'
            ),
            row=i + 1,
            col=j + 1
        )

fig.update_layout(title=f'Conv Layer {n_conv_layers}')
fig.show()


