# %%
from pennylane import RZ, RY
import pennylane as qml
import torch
from src.ansatze import SimpleAnsatz1
from src.geometric_classifier import BasicClassifierTorch
from src.quanvolution import Quanvolution2DTorchLayer
import matplotlib.pyplot as plt
from examples.utils import SymmetricDataset

dataset = SymmetricDataset(100, 9)

# %%
# plt.imshow(dataset[1][0].reshape(9, 9))

# %%
quantum_circs = [BasicClassifierTorch(
    feature_map='RotEmbedding', ansatz=SimpleAnsatz1, size=9, measurement=qml.PauliZ(4))]
quantum_circs_properties = [{'n_layers': 1, 'embedding_pauli': RY}]
quanv_layer = Quanvolution2DTorchLayer(quantum_circs,
                                       quantum_circs_properties,
                                       (9, 9),
                                       3,
                                       3,
                                       [{'params': (1, 18)}]).to(torch.device('cpu'))

# %%
fake_data = torch.tensor(dataset[1][0].reshape(1, 1, 9, 9))


# %%
quanv_layer(fake_data)
