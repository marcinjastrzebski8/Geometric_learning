from src.quanvolution import Quanvolution2DTorchLayer
from src.geometric_classifier import BasicClassifierTorch
from src.ansatze import SimpleAnsatz1, MatchCallumAnsatz
import matplotlib.pyplot as plt
import torch
from pennylane import RX, RY, RZ

circs = [
    BasicClassifierTorch('RotEmbedding',
                         MatchCallumAnsatz,
                         4)
]
n_layers = 2
properties = [{
    'n_layers': n_layers, 'ansatz_block': [RY, RZ], 'embedding_pauli': RX}]
weights_shapes_list = [{'params': (1, n_layers, 4, 2)}]
my_image = torch.tensor([[[[1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]]], dtype=float)
print(my_image.shape)
shifted_image = torch.tensor([[[[0, 1, 0],
                              [0, 1, 0],
                              [0, 1, 0]]]], dtype=float)

my_quanv = Quanvolution2DTorchLayer(circs,
                                    properties,
                                    (3, 3),
                                    1,
                                    2,
                                    weights_shapes_list)

output = my_quanv.forward(my_image)
output_on_shifted = my_quanv.forward(shifted_image)
print(output)
print(output_on_shifted)

fig, ax = plt.subplots(2, 2)

ax[0][0].imshow(my_image.view(3, 3).detach().numpy())
ax[0][1].imshow(output.view(2, 2).detach().numpy())
ax[1][0].imshow(shifted_image.view(3, 3).detach().numpy())
ax[1][1].imshow(output_on_shifted.view(2, 2).detach().numpy())
plt.show()
