from data.datasets import MicrobooneTrainData
import torch
import matplotlib.pyplot as plt

data_path = '/Users/marcinjastrzebski/Desktop/ACADEMIA/THIRD_YEAR/Geometric_classifier/data/microboone_from_callum'

size = 49

train_dataset = MicrobooneTrainData(size)
train_data = train_dataset.data
train_labels = train_dataset.labels
print(train_data.shape)
print(train_labels.shape)

"""
train_dataset = MicrobooneTrainData(21)
train_data = train_dataset.data
train_labels = train_dataset.labels
print(train_data.shape)
print(train_labels.shape)
"""
fig, ax = plt.subplots(10, 10, figsize=(20, 10))
plot_dataset = train_data[:100].view(10, 10, size, size)
plot_labels = train_labels[:100].view(10, 10)
label_dict = {0: 'shower', 1: 'track'}
for col_id in range(10):
    for row_id in range(10):
        ax[col_id][row_id].imshow(plot_dataset[col_id][row_id])
        ax[col_id][row_id].set_axis_off()
        ax[col_id][row_id].set_title(
            label_dict[int(plot_labels[col_id][row_id])])
plt.tight_layout()
plt.show()
