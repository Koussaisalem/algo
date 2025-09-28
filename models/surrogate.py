import torch
import torch.nn as nn
import schnetpack as spk
from torch_geometric.data import Data

class Surrogate(nn.Module):
    def __init__(self, hidden_channels=64, num_interactions=3, cutoff=5.0):
        """
        The definitive, architecturally correct Surrogate model, built precisely
        according to the official SchNetPack "LEGO block" design philosophy.
        """
        super(Surrogate, self).__init__()

        # --- LEGO Block 1: The Representation (The Engine) ---
        # This is the core SchNet architecture that learns atom-wise features.
        # We use the official, pre-built representation module.
        schnet_representation = spk.representation.SchNet(
            hidden_channels=hidden_channels,
            n_interactions=num_interactions,
            cutoff=cutoff,
            n_gaussians=50 # A standard choice for the radial basis
        )

        # --- LEGO Block 2: The Output Module (The Predictor) ---
        # This module takes the atom-wise features from the representation
        # and predicts a single, atomic energy contribution for each.
        # It then sums them to get the total energy.
        atomwise_output = spk.atomistic.Atomwise(
            n_in=hidden_channels,
            # The key for the final predicted property
            output_key='energy'
        )

        # --- Assembly: The Final Model ---
        # The NeuralNetworkPotential class is the main "chassis" that assembles
        # the representation and output modules into a complete, working model.
        self.model = spk.model.NeuralNetworkPotential(
            representation=schnet_representation,
            output_modules=[atomwise_output]
        )

    def forward(self, data: Data):
        # The input to a SchNetPack model is a dictionary of tensors.
        # We must convert our PyTorch Geometric Data object into this format.
        inputs = {
            spk.properties.Z: data.z.long(),      # Atomic numbers
            spk.properties.R: data.pos,         # Atomic positions
            spk.properties.batch: data.batch,     # Batch indices
        }
        
        # The model handles everything internally and returns a dictionary of results.
        results = self.model(inputs)
        
        # We return the predicted energy.
        return results['energy']
