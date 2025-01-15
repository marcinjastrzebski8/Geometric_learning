import matplotlib.pyplot as plt
from pennylane import RX, RY, RZ
import torch
from torchvision.transforms.functional import hflip
from src.torch_architectures import ConvolutionalEQEQ
from examples.equivariant_quanvolution_study_with_trainer import prep_equiv_quant_classifier, prep_equiv_quanv_model
import numpy as np

from data.datasets import MicrobooneTrainData


def crop_an_image(image, cropped_size):
    """
    For making fake visualisation of what a given datapoint looks like at different patch sizes.
    Can't do it naturally because the different datasets are generated randomly so an index i
    between two datasets does not correspond to the same pixel of an event.

    Assumes main image has size 49.
    """
    # middle pixel for image of size 49
    middle_pixel = 24
    half_size = int((cropped_size - 1)/2)
    index_low = middle_pixel-half_size
    index_high = middle_pixel+half_size+1

    return image[index_low:index_high, index_low:index_high]


imsizes = [5, 9, 21, 31, 49]
label_lookup = {0: 'Track', 1: 'Shower'}

fig, ax = plt.subplots(5, 10, figsize=(15, 7))
for col_id in range(10):
    for row_id, imsize in enumerate(imsizes):
        if row_id == 0:
            ax[row_id][col_id].set_title(
                label_lookup[int(MicrobooneTrainData(49)[col_id][1])], fontsize=16)
        datapoint = MicrobooneTrainData(49)[col_id][0]
        print(datapoint.shape)
        ax[row_id][col_id].imshow(crop_an_image(datapoint[0], imsize))
        ax[row_id][col_id].set_axis_off()
fig.tight_layout(pad=0.5)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.savefig('for_paper_show_mnist_datapoint', dpi=300)

for figid, imsize in enumerate(imsizes):
    fig0, ax0 = plt.subplots()
    ax0.imshow(crop_an_image(MicrobooneTrainData(49)
               [9][0][0], imsize), vmin=0, vmax=4)
    ax0.set_axis_off()
    print(MicrobooneTrainData(49)[9][1])
    # plt.savefig(f'for_paper_example_mnist_datapoint_{imsize}', dpi=300)


data = MicrobooneTrainData(21, 100)
print(len(data))
