# gpu, seed
import config
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from model.mewispool import Net3

args = config.parse()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

# dataset = my_load_Hyper_data(args, args.hyper_data, args.feature_noise)
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

train_dataset = dataset[:150]
test_dataset = dataset[150:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = Net3(input_dim=dataset.num_features, hidden_dim=args.hidden_dim, num_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       patience=args.schedule_patience,
                                                       factor=args.schedule_factor,
                                                       verbose=True)
nll_loss = torch.nn.NLLLoss()

#train process