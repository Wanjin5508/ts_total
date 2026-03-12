import torch 
import torch.nn as nn
import numpy as np
import plotly.graph_object as go


from pathlib import Path
from tqdm import tqdm
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from dataset_tcnae import TCNAEDataset
from model import TCNAE
from hyperparameters import hyperparameters

result_dir = Path(r"results")

model = TCNAE(**hyperparameters['model'])
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])

train_loader = DataLoader(
    TCNAEDataset(path_to_signal=hyperparameters["train_data_path"], **hyperparameters['dataloader']),
    batch_size=hyperparameters['batch_size']
)

test_loader = DataLoader(
    TCNAEDataset(path_to_signal=hyperparameters['test_data_path'], **hyperparameters['dataloader']),
    batch_size=hyperparameters['batch_size']
)

log_interval = 2

train_losses = []
train_counter = []


train_run = 0

window_size = hyperparameters['dataloader']['window_size']
window_shift = hyperparameters['dataloader']['window_shift']

def train(epoch):
    global train_run
    model.train()
    for batch_idx, (input, target) in enumerate(tqdm(train_loader, leave=False)):
        optimizer.zero_grad()
        pred = model(input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            tqdm.write(f"Train Epoch: {epoch} [[{batch_idx * len(input)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%]) \tLoss: {loss.item()}]]")
            train_losses.append(loss.item())
            train_counter.append(train_run)
        train_run += 1
        
# Train
try:
    for epoch in tqdm(range(1, hyperparameters['n_epochs'] + 1)):
        train(epoch)
    torch.save(model.state_dict(), result_dir/f"TCNAE_{hyperparameters['version']}.pth")
    # torch.save(model.optimizer.state_dict(), result_dir/f"TCNAE_optimizer_{hyperparameters['version']}.pth")
    
    train_loss_elements = len(train_losses) // 10
    train_losses_plot = torch.tensor(train_losses[0: train_loss_elements*10]).view(-1, 10).mean(1)
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=np.arange(train_losses_plot.shape[0]), y=train_losses_plot, name='signal'))
    fig.update_layout(height=800, width=1920, 
                      title_text="Training_loss")
    fig.write_html(result_dir/f"TCNAE_{hyperparameters['version'].html}")
    fig.show()
    
except KeyboardInterrupt:
    train_loss_elements = len(train_losses) // 10
    train_losses_plot = torch.tensor(train_losses[0: train_loss_elements*10]).view(-1, 10).mean(1)
    torch.save(model.state_dict(), result_dir/f"TCNAE_{hyperparameters['version']}.pth")
    # torch.save(model.optimizer.state_dict(), result_dir/f"TCNAE_optimizer_{hyperparameters['version']}.pth")
    
    train_loss_elements = len(train_losses) // 10
    train_losses_plot = torch.tensor(train_losses[0: train_loss_elements*10]).view(-1, 10).mean(1)
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=np.arange(train_losses_plot.shape[0]), y=train_losses_plot, name='signal'))
    fig.update_layout(height=800, width=1920, 
                      title_text="Training_loss")
    fig.write_html(result_dir/f"TCNAE_{hyperparameters['version'].html}")
    fig.show()
    
# Test
batch_size = hyperparameters['batch_size']
length_raw_data = len(test_loader)
noise_prediction = np.zeros(shape=(length_raw_data * batch_size, 1, window_size))
mse_per_window = np.zeros(shape=(length_raw_data * batch_size, 1, window_size))
e_per_window = np.zeros(shape=(length_raw_data * batch_size, 1, window_size))

model.eval()

with torch.no_grad():
    for i, (sample, _) in enumerate(tqdm(test_loader), start=1):
        
        try:
            y_pred = model(sample)
            noise_prediction[(i-1)*batch_size: (i-1)*batch_size+y_pred.shape[0], :] = y_pred.numpy()
            mse_per_window[(i-1)*batch_size: (i-1)*batch_size+y_pred.shape[0]: ] = criterion(y_pred, sample).item()
            e_per_window[(i-1)*batch_size: (i-1)*batch_size+y_pred.shape[0]: ] = sample.numpy()**2
        except Exception as e:
            print(e)
            break
        
        
        
noise_prediction = noise_prediction[::window_size//window_shift].flatten()
log_mse_per_window = 20 * np.log(mse_per_window[::window_size//window_shift].flatten())
log_e_per_window = 20 * np.log(e_per_window[::window_size//window_shift].flatten())

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True)
fig.append_trace(go.Scatter(x=np.arange(len(test_loader.dataset.raw_data)), y=test_loader.dataset.raw_data, name='orig'), row=1, col=1)

fig.append_trace(go.Scatter(x=np.arange(len(noise_prediction)), y=noise_prediction, name='pred'), row=1, col=1)

fig.append_trace(go.Scatter(x=np.arange(len(test_loader.dataset.raw_data)), y=test_loader.dataset.raw_data - noise_prediction[0:len(test_loader.dataset.raw_data)], name='residual'), row=2, col=1)

fig.append_trace(x=go.Scatter(x=np.arange(0, len(0, noise_prediction), window_shift), y=log_mse_per_window - np.mean(log_e_per_window), name='20*log(MSE)'), row=3, col=1)

fig.append_trace(go.Scatter(x=np.arange(0, len(noise_prediction), window_shift), y=log_e_per_window - np.mean(log_e_per_window), name='20*log(signalenergy)'), row=3, col=1)

fig.update_layout(height=1080, width=1920)
fig.write_html(result_dir / f"TCNAE_{hyperparameters['version']}.html")

