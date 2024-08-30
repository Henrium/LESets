#%% Make mol graph
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import numpy as np
from models import LESets, LESetsAtt
from data_utils import GraphSetDataset, graph_set_collate
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import random
from sklearn.metrics import r2_score, mean_absolute_error

n_node_feature = 19
batch_size = 128
use_attention = True
response = 'youngs'

hyperpars = {
    # Architecture
    'gnn_dim': 32,
    'lesets_dim': 32,
    'n_conv_layers': 2,
    'conv': 'GraphConv',
    'after_readout': 'tanh',
    # Training
    'max_ep': 100,
    'es_patience': 10,
    'max_ep_wo_improv': 20,
    # Learning rate
    'lr': 0.001,
    'lrsch_patience': 10,
    'lrsch_factor': 0.5,
    # Regularization
    'weight_decay': 0.0001,
    'norm': None
}

best_model = None
SEED = 74
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = GraphSetDataset('data/hea_{}.pkl'.format(response))
train_data, val_data, test_data = random_split(dataset, (0.6, 0.2, 0.2))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_loader.collate_fn = graph_set_collate
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
val_loader.collate_fn = graph_set_collate


# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_attention:
    model = LESetsAtt(n_node_features=n_node_feature, gnn_dim=hyperpars['gnn_dim'], lesets_dim=hyperpars['lesets_dim'], conv=hyperpars['conv'], n_conv_layers=hyperpars['n_conv_layers'], after_readout=hyperpars['after_readout'], norm=hyperpars['norm']).to(device)
else:
    model = LESets(n_node_features=n_node_feature, gnn_dim=hyperpars['gnn_dim'], lesets_dim=hyperpars['lesets_dim'], conv=hyperpars['conv'], n_conv_layers=hyperpars['n_conv_layers'], after_readout=hyperpars['after_readout'], norm=hyperpars['norm']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperpars['lr'], weight_decay=hyperpars['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=hyperpars['lrsch_factor'], patience=hyperpars['lrsch_patience'])
loss_fn = torch.nn.MSELoss()

#%%
def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for sample in loader:
        # "inputs" are a batch of graphs sets
        inputs, targets = sample
        sample_size = len(targets)
        outs = torch.empty((sample_size, 1)).to(device)        
        targets = torch.tensor(targets).to(device)
        for j in range(sample_size):   
            graph_set = inputs[j].to(device)
            optimizer.zero_grad()
            outs[j] = model(graph_set)
        loss = criterion(outs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sample in loader:
            inputs, targets = sample
            sample_size = len(targets)
            outs = torch.empty((sample_size, 1)).to(device)
            targets = torch.tensor(targets).to(device)
            for j in range(sample_size):
                graph_set = inputs[j].to(device)
                outs[j] = model(graph_set)
            loss = criterion(outs, targets)
            val_loss += loss.item()
    return val_loss / len(loader)

# Set early stopping criteria
best_val_loss = np.inf
epochs_wo_improv = 0
print(f'Total params: {sum(param.numel() for param in model.parameters())}')

# The training loop
for epoch in range(hyperpars['max_ep']):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, val_loader, loss_fn)    
    scheduler.step(val_loss)

    # Early stopping check
    if epoch > hyperpars['es_patience']:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_wo_improv = 0
        else:
            epochs_wo_improv += 1
        if epochs_wo_improv >= hyperpars['max_ep_wo_improv']:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'Epoch {epoch+1}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}')

#%% Plots
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
model_name = '{}_{}_h{}_{}'.format(hyperpars['conv'], hyperpars['n_conv_layers'], hyperpars['lesets_dim'], response)

targets = []
predicted = []

if best_model is not None:
    model.load_state_dict(best_model)
    torch.save(best_model, 'results/checkpoints/{}.pt'.format(model_name))

model.eval()
with torch.no_grad():
    for sample in test_data:
        inputs, target = sample
        inputs = Batch.from_data_list(inputs).to(device)
        out = model(inputs)
        targets.append(target)
        predicted.append(out.cpu().numpy())

targets = dataset.get_orig(np.stack(targets).squeeze())
predicted = dataset.get_orig(np.stack(predicted).squeeze())
results = pd.DataFrame({'target': targets, 'predicted': predicted})

# spearman_r = spearmanr(targets, predicted)
# pearson_r = pearsonr(targets, predicted)

print('R2: {:.4f}, MAE: {:.4f}'.format(
    r2_score(targets, predicted),
    mean_absolute_error(targets, predicted)
))

fig, ax = plt.subplots()
ax.scatter(targets, predicted)
ax.set_aspect('equal')
plt.xlabel('Target {}'.format(response))
plt.ylabel('Predicted {}'.format(response))
plt.axline([0, 0], [1, 1], color='black')
plt.savefig('results/{}.png'.format(model_name))

results.to_csv('results/{}.csv'.format(model_name), index=False)
