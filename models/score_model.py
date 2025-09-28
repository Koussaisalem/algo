import torch
import torch.nn as nn
import schnetpack as spk
from torch_geometric.data import Data

class ScoreModel(nn.Module):
    def __init__(self, hidden_channels=64, num_interactions=3, cutoff=5.0):
        """
        The definitive, architecturally correct ScoreModel, built using the
        official SchNetPack design philosophy for atom-wise vector predictions.
        """
        super(ScoreModel, self).__init__()

        # --- LEGO Block 1: The Representation (The Engine) ---
        schnet_representation = spk.representation.SchNet(
            hidden_channels=hidden_channels,
            n_interactions=num_interactions,
            cutoff=cutoff,
            n_gaussians=50
        )

        # --- LEGO Block 2: The Output Module (The Predictor) ---
        # For the score model, we need to predict a 3D vector for each atom.
        # We still use the Atomwise module, but configure its output shape.
        atomwise_output = spk.atomistic.Atomwise(
            n_in=hidden_channels,
            n_out=3, # CRITICAL: Predict a 3D vector, not a scalar
            output_key='score'
        )

        # --- Assembly: The Final Model ---
        self.model = spk.model.NeuralNetworkPotential(
            representation=schnet_representation,
            output_modules=[atomwise_output]
        )

    def forward(self, data: Data):
        # Convert the PyTorch Geometric Data object to the SchNetPack input format.
        inputs = {
            spk.properties.Z: data.z.long(),
            spk.properties.R: data.pos,
            spk.properties.batch: data.batch,
        }
        
        # The model returns a dictionary of results.
        results = self.model(inputs)
        
        # We return the predicted 3D score vectors for each atom.
        return results['score']