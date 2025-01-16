import matplotlib.pyplot as plt
from pennylane import RX, RY, RZ
import torch
from torchvision.transforms.functional import hflip
from src.torch_architectures import ConvolutionalEQEQ, ConvolutionalEQNEC
from examples.equivariant_quanvolution_study_with_trainer import prep_equiv_quant_classifier, prep_equiv_quanv_model

from data.datasets import MicrobooneTrainData


fake_json_config = {'image_size': 21,
                    'stride_quanv0': 1,
                    'stride_quanv1': 2,
                    'input_channel_side_len1': 10}
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
               '2local_placements': ['ring_third_neighbours_corner'],
               }

architecture_config = {'quanv0': prep_equiv_quanv_model(fake_config, fake_json_config, True),
                       'quanv1': prep_equiv_quanv_model(fake_config, fake_json_config, False),
                       'quantum_classifier': prep_equiv_quant_classifier(fake_config),
                       'pooling_kernels_size': [2, 3],
                       'pooling_strides': [2, 1]}

architecture_config1 = {'quanv0': prep_equiv_quanv_model(fake_config, fake_json_config, True),
                        'quanv1': prep_equiv_quanv_model(fake_config, fake_json_config, False),
                        'dense_units': [128, 32],
                        'image_size': 21,
                        'n_filters1': 1,
                        'use_dropout0': True,
                        'dropout0': 0.1,
                        'use_dropout1': True,
                        'dropout1': 0.1}

model = ConvolutionalEQEQ(architecture_config)
model1 = ConvolutionalEQNEC(architecture_config1)
datapoint = MicrobooneTrainData(21)[0][0]
datapoint1 = datapoint.rot90(1, (1, 2))
idx = torch.randperm(datapoint.nelement())
datapoint2 = datapoint.view(-1)[idx].view(datapoint.size())
# ax[0].imshow(datapoint[0])
# ax[1].imshow(datapoint1[0])
# ax[2].imshow(datapoint2[0])
# plt.savefig('datapoint_rotated', dpi=300)
datapoint_processed = model.forward(datapoint.view(
    datapoint.shape[0], 1, datapoint.shape[1], datapoint.shape[2]), 'filters_after_quanv_0')
datapoint_processed1 = model.forward(datapoint1.view(
    datapoint1.shape[0], 1, datapoint1.shape[1], datapoint1.shape[2]), 'filters_after_quanv_1')
datapoint_processed2 = model.forward(datapoint2.view(
    datapoint2.shape[0], 1, datapoint2.shape[1], datapoint2.shape[2]), 'filters_after_quanv_2')
print(datapoint_processed)
print(datapoint_processed1)
print(datapoint_processed2)
