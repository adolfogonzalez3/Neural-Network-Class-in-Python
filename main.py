
from itertools import chain

from pynn.activations import Sigmoid
from pynn.layers import Dense
from pynn.datasets import MnistDataset
from pynn import Network, Matrix2d

from tqdm import tqdm, trange

class What:
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            print(idx, len(idx))
        elif isinstance(idx, slice):
            print('Slice:', idx)
        else:
            print(idx)

def main():
    train_sets = MnistDataset()

    network = Network()
    network.append(Dense(784, 10))
    network.append(Sigmoid())

    for iteration, (images, labels) in enumerate(train_sets):
        images = images / 255
        results = network.forward(images)
        labels_onehot = [[0]*10 for _ in range(results.rows)]
        for i in range(results.rows):
            labels_onehot[i][int(labels[i])] = 1
        labels_onehot = Matrix2d(
            chain.from_iterable(labels_onehot), results.rows, 10
        )
        gradient = labels_onehot - results
        gradients = network.gradients(gradient)
        gradients = [-(grad*1e-2)/32 for grad in gradients]
        network.update_add(gradients)
        if iteration % 100 == 0:
            cost = (sum(gradient**2))/2
            cost = cost / 32
            print("Cost:", cost)

    test_sets = MnistDataset(train=False)

    accuracies = []
    for images, labels in test_sets:
        images = images / 255
        results = network.forward(images)
        predictions = [list(row) for row in results.iter_rows()]
        #print([max(row) for row in predictions])
        predictions = Matrix2d(
            [row.index(max(row)) for row in predictions],
            labels.rows, 1
        )
        #print(predictions)
        #print(labels)
        #print(predictions == labels)
        #input()
        accuracies.append(sum(labels == predictions))
    print(f'Accuracy: {sum(accuracies) / len(test_sets.labels)}')




if __name__ == '__main__':
    main()