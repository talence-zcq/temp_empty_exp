# gpu, seed
import numpy as np
import torch
import config
from util.convert_datasets_to_pygDataset import my_load_data
from util.preprocessing import ExtractV2E, expand_edge_index

args = config.parse()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset = my_load_data(args, args.data, args.feature_noise)
print(dataset)
print(dataset.data)
print(expand_edge_index(ExtractV2E(dataset.data)))
