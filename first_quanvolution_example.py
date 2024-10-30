import sys
import traceback
try:

    from copy import deepcopy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    from torchvision.transforms import CenterCrop, Compose, ToTensor, Normalize

    from sklearn.metrics import accuracy_score

    import pennylane as qml
    from pennylane import RY

    from src.losses import binary_cross_entropy_loss
    from src.quanvolution import Quanvolution2DTorchLayer, EquivariantQuanvolution2DTorchLayer
    from src.geometric_classifier import BasicClassifierTorch
    from src.ansatze import SimpleAnsatz1
    from pqc_training.trainer import TorchQMLTrainer
    from utils import SymmetricDataset

    def mnist_prep():
        """
        Load MNIST.
        """
        transforms = Compose(
            [CenterCrop(27), ToTensor(), Normalize((0.5), (0.5))])
        # get data
        train_data = datasets.MNIST('data', True, transforms)
        test_data = datasets.MNIST('data', False, transforms)
        # keep only 0s and 1s
        mask_train = (train_data.targets == 0) | (train_data.targets == 1)
        mask_test = (test_data.targets == 0) | (test_data.targets == 1)

        train_data.data = train_data.data[mask_train]
        train_data.targets = train_data.targets[mask_train]

        test_data.data = test_data.data[mask_test]
        test_data.targets = test_data.targets[mask_test]
        # not sure deepcopy needed but being safe
        train_data = Subset(deepcopy(train_data), range(500))
        test_data = Subset(deepcopy(test_data), range(500))

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = DataLoader(
            train_data, batch_size=10, shuffle=True)
        validation_loader = DataLoader(
            test_data, batch_size=10, shuffle=False)

        # Report split sizes
        # print('Training set has {} instances'.format(len(train_data)))
        # print('Validation set has {} instances'.format(len(test_data)))
        return train_data, test_data, training_loader, validation_loader

    def prep_data():
        """
        toy data used for development of code
        """
        # get data
        data = SymmetricDataset(500, 9)
        # data comes stacked, conv wants it on a plane
        data.data = data.data.reshape(1000, 1, 9, 9)
        train_data = Subset(data, range(500))
        test_data = Subset(data, range(500, 1000))

        # Create data loaders for our datasets;
        training_loader = DataLoader(
            train_data, batch_size=10, shuffle=True)
        # feels silly for test loader to have batch sizes
        validation_loader = DataLoader(
            test_data, batch_size=10, shuffle=False)

        # Report split sizes
        print('Training set has {} instances'.format(len(train_data)))
        print('Validation set has {} instances'.format(len(test_data)))
        return train_data, test_data, training_loader, validation_loader

    def prep_quanv_model():
        """
        Prep the quanv layer
        """
        n_layers = 4
        input_channel_side_len = 27
        quantum_circs = [BasicClassifierTorch(
            feature_map='RotEmbedding', ansatz=SimpleAnsatz1, size=9, measurement=qml.PauliZ(4))]
        quantum_circs_properties = [
            {'n_layers': n_layers, 'embedding_pauli': RY}]
        quanv_layer = Quanvolution2DTorchLayer(quantum_circs,
                                               quantum_circs_properties,
                                               (input_channel_side_len,
                                                input_channel_side_len),
                                               3,
                                               3,
                                               [{'params': (n_layers, n_layers*18)}])
        return quanv_layer

    def prep_equiv_quanv_model(first_layer=True):
        """
        Prep the quanv layer
        """
        n_layers = 4

        patch_circuit = BasicClassifierTorch(
            feature_map='RotEmbedding', ansatz=SimpleAnsatz1, size=9, measurement=qml.PauliZ(4))
        patch_circuit_properties = {
            'n_layers': n_layers, 'embedding_pauli': RY}
        if first_layer:
            quantum_circs = [patch_circuit]
            quantum_circs_properties = [patch_circuit_properties]
            input_channel_side_len = 27
        else:
            # using the same circuit for each pose
            quantum_circs = [patch_circuit for i in range(4)]
            quantum_circs_properties = [patch_circuit_properties
                                        for i in range(4)]
            input_channel_side_len = 9

        # this is a placeholder for however I choose to define the group object - dict might work
        group = {'size': 4}
        quanv_layer = EquivariantQuanvolution2DTorchLayer(group,
                                                          first_layer,
                                                          quantum_circs,
                                                          quantum_circs_properties,
                                                          (input_channel_side_len,
                                                           input_channel_side_len),
                                                          3,
                                                          3,
                                                          [{'params': (n_layers, n_layers*18)}])

        return quanv_layer

    def train_model(model, train_loader, criterion, optimizer, epochs=5):
        """
        Stolen from Callum. Ideally I'd revisit my trainer.py module - it should encompass anything like this example.
        """
        model.train()
        for epoch in range(epochs):
            print('epoch: ', epoch)
            running_loss = 0.0
            for images, labels in train_loader:
                print('labels: ', labels)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    def evaluate_model(model, test_loader):
        model.eval()
        all_labels = []
        all_preds = []

        # TODO: FIGUREOUT WHAT DOES
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                # this outputs (vals, indecies) - we just want the indecies as they correspond to label
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Test accuracy: {accuracy:.4f}')
        print(all_labels)
        print(all_preds)

    class FullQuanvClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            self.quanv_layer = prep_quanv_model()
            self.conv_layer = nn.Conv2d(1, 1, 3, 3)
            self.fc1 = nn.Linear(81, 100)
            self.fc2 = nn.Linear(100, 20)
            self.fc3 = nn.Linear(20, 2)

        def forward(self, x):
            # (27,27) becomes (9*9)
            x = F.relu(self.quanv_layer(x))
            # x = F.relu(self.conv_layer(x))
            print('x after the quanv layer is:', x.shape)
            x = torch.flatten(x, 1)
            # (9*9) becomes (100)
            x = F.relu(self.fc1(x))
            # (100) becomes (20)
            x = F.relu(self.fc2(x))

            # (20) becomes (2) - output of model
            x = F.relu(self.fc3(x))

            return x

    class FullEquivQuanvClassifier(nn.Module):
        """
        NOTE: The full classifier is likely not going to be label-invariant because the fc layers 
        have not been made equivariant/invariant.
        For now just checking if equiv quanv works.
        """

        def __init__(self):
            super().__init__()

            self.quanv_layer = prep_equiv_quanv_model()
            self.quanv_layer_subsequent = prep_equiv_quanv_model(False)
            self.conv_layer = nn.Conv2d(1, 1, 3, 3)
            self.fc1 = nn.Linear(324, 100)
            self.fc1_after_subsequent = nn.Linear(36, 100)
            self.fc2 = nn.Linear(100, 20)
            self.fc3 = nn.Linear(20, 2)

        def forward(self, x):
            # (27,27) becomes (4*9*9)
            x = F.relu(self.quanv_layer(x))
            print('x after the quanv layer is:', x.shape)
            # becomes (4*3*3)
            x = F.relu(self.quanv_layer_subsequent(x))
            print('x after the second quanv layer is:', x.shape)
            x = x.permute(1, 0, 2, 3, 4)
            x = torch.flatten(x, 1)
            # (9*9) becomes (100)
            x = F.relu(self.fc1_after_subsequent(x))
            # (100) becomes (20)
            x = F.relu(self.fc2(x))

            # (20) becomes (2) - output of model
            x = F.relu(self.fc3(x))

            return x

    tr_data, te_data, tr_loader, te_loader = mnist_prep()
    my_model = FullEquivQuanvClassifier()
    my_criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(my_model.parameters(), lr=0.01)
    print('Training_model')

    train_model(my_model, tr_loader, my_criterion, optimiser, epochs=20)
    evaluate_model(my_model, te_loader)
except Exception as e:
    with open('error_log.txt', 'w') as f:
        traceback.print_exc(file=f)
