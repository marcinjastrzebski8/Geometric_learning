from data.datasets import RotatedMNISTTest, RotatedMNISTTrain, RotatedMNISTVal, MicrobooneTrainData
import torch
import matplotlib.pyplot as plt

data_path = '/Users/marcinjastrzebski/Desktop/ACADEMIA/THIRD_YEAR/Geometric_classifier/data/rot_MNIST'

# NOTE: the below will not work anymore as the datasets have been superseded by non-dynamic ones
"""
train_dataset = RotatedMNISTTrain()
val_dataset = RotatedMNISTVal()
test_dataset = RotatedMNISTTest()
torch.save(train_dataset.data, data_path+'/train_data.pt')
torch.save(val_dataset.data, data_path+'/val_data.pt')
torch.save(test_dataset.data, data_path+'/test_data.pt')
torch.save(train_dataset.labels, data_path+'/train_labels.pt')
torch.save(val_dataset.labels, data_path+'/val_labels.pt')
torch.save(test_dataset.labels, data_path+'/test_labels.pt')
"""
train_dataset = RotatedMNISTTrain()
train_data = train_dataset.data
train_labels = train_dataset.labels
val_dataset = RotatedMNISTVal()
val_data = val_dataset.data
val_labels = val_dataset.labels
test_dataset = RotatedMNISTTest()
test_data = test_dataset.data
test_labels = test_dataset.labels
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)
print('MICROBOONE IS ', MicrobooneTrainData().data.shape,
      MicrobooneTrainData().labels.shape)
for data, name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
    fig, ax = plt.subplots(10, 10)
    dataset = data[:100].view(
        10, 10, data.shape[2], data.shape[3])
    for row_id in range(10):
        for col_id in range(10):

            # Plot the image
            ax[row_id][col_id].imshow(dataset[row_id][col_id].detach().numpy())
            ax[row_id][col_id].axis('off')
    plt.savefig(f'rot_mnist_vis_{name}', dpi=300)
print(train_dataset[:2], MicrobooneTrainData()[:2])
