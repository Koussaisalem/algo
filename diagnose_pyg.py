import torch_geometric
from torch_geometric.nn import conv
print(f'PyG version: {torch_geometric.__version__}')
available_convs = [name for name in dir(conv) if 'Conv' in name and not name.startswith('_')]
print(f'Available convolutions: {available_convs}')