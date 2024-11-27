import matplotlib.pyplot as plt
from pennylane import RX, RY, RZ
import torch
from src.torch_architectures import ConvolutionalEQEQ
from examples.equivariant_quanvolution_study_with_trainer import prep_equiv_quant_classifier, prep_equiv_quanv_model

from data.datasets import MicrobooneTrainData


fake_json_config = {'image_size': 5}
fake_config = {'n_layers': 1,
               'n_reuploads': 1,
               'n_filters0': 1,
               'n_filters1': 1,
               'param_init_max_vals': 1,
               'classifier_n_layers': 1,
               'classifier_n_reuploads': 1,
               '1local_gates': [RX, RY, RZ, RX],
               '2local_gates': ['YY', 'YY', 'YY', 'YY'],
               '1local_placements': ['centre', 'corner'],
               '2local_placements': ['ring_third_neighbours_corner']}

architecture_config = {'quanv0': prep_equiv_quanv_model(fake_config, fake_json_config, True),
                       'quanv1': prep_equiv_quanv_model(fake_config, fake_json_config, False),
                       'quantum_classifier': prep_equiv_quant_classifier(fake_config)}

model = ConvolutionalEQEQ(architecture_config)
fig, ax = plt.subplots(1, 1)
# datapoint = MicrobooneTrainData()[0][0]
datapoint = torch.rand((1, 5, 5))
# ax.imshow(datapoint[0])
# plt.show(block=True)
datapoint_processed = model.forward(datapoint.view(
    datapoint.shape[0], 1, datapoint.shape[1], datapoint.shape[2]))
print(datapoint_processed)
