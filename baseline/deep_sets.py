import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

### Define models

class PhiNet(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(n_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: batch_size * n_elements * n_features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class DeepSets(torch.nn.Module):
    def __init__(self, n_features, phi_dim, hidden_dim):
        super().__init__()

        self.phi = PhiNet(n_features, phi_dim)
       
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(phi_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_list):
        # embeddings: batch_size * n_elements * phi_dim
        embeddings = self.phi(x_list)
        # x: batch_size * phi_dim
        x = torch.sum(embeddings, dim=1)
        # out: batch_size
        out = self.rho(x)
        return out


### Dataset class
class XListDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        # shape: n_samples * n_elements * n_features
        self.x = np.stack([self.data_list[i][0] for i in range(len(self.data_list))])
        self.y = np.array([self.data_list[i][1] for i in range(len(self.data_list))])
        
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        # targets = np.array([self.data_list[i][-1] for i in range(len(self.data_list))])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        x = self.x[index]
        y_norm = (self.y[index] - self.y_mean) / self.y_std

        return torch.tensor(x, dtype=torch.float), torch.tensor(y_norm, dtype=torch.float)

    def get_orig(self, y):
        return y * np.std(self.y) + np.mean(self.y)

best_model = None
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

response = 'youngs'
dataset = XListDataset('./data/ds_hea_{}.pkl'.format(response))
train_data, val_data, test_data = random_split(dataset, (0.6, 0.2, 0.2))
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepSets(n_features=19, phi_dim=32, hidden_dim=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
loss_fn = torch.nn.MSELoss()


def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for sample in loader:
        # "inputs" are a batch of graphs sets
        inputs, targets = sample
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outs = model(inputs).squeeze()
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
            inputs = inputs.to(device)
            outs = model(inputs).squeeze()
            targets = targets.to(device)
            loss = criterion(outs, targets)
            val_loss += loss.item()
    return val_loss / len(loader)

best_val_loss = np.inf
epochs_wo_improv = 0
print(f'Total params: {sum(param.numel() for param in model.parameters())}')


for epoch in range(1000):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, val_loader, loss_fn)    
    scheduler.step(val_loss)

    # Early stopping check
    if epoch > 10:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_wo_improv = 0
        else:
            epochs_wo_improv += 1
        if epochs_wo_improv >= 20:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'Epoch {epoch+1}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}')


model_name = 'deepsets_' + response

targets = np.array([])
predicted = np.array([])

if best_model is not None:
    model.load_state_dict(best_model)
    torch.save(best_model, 'checkpoints/{}.pt'.format(model_name))

model.eval()
with torch.no_grad():
    for sample in test_loader:
        inputs, target = sample
        inputs = inputs.to(device)
        out = model(inputs).squeeze()
        targets = np.append(targets, target)
        predicted = np.append(predicted, out.cpu().numpy())

targets = dataset.get_orig(targets)
predicted = dataset.get_orig(predicted)

# targets = np.array(targets)
# predicted = np.concatenate(predicted)
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