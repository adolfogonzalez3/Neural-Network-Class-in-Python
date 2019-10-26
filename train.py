'''Script used to train a neural network on MNIST.'''
import argparse
from time import time
from datetime import timedelta
from itertools import chain

from pynn.activations import Sigmoid
from pynn.layers import Dense
from pynn.datasets import MnistDataset
from pynn import Network, Matrix2d

class ProgPrinter:
    def __init__(self):
        self.data = ''

    def __call__(self, *args):
        self.clear()
        self.data = ' '.join([str(arg) for arg in args])
        print(self.data, end='')
        print('\b'*len(self.data), end='')

    def clear(self):
        #print('\b'*len(self.data), end='')
        print(' '*len(self.data), end='')
        print('\b'*len(self.data), end='')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.clear()

def progbar(iterable, total=None, outermost=False):
    '''Create a progress bar.'''
    begin = timedelta(seconds=time())
    cnt_step = 0
    dur = timedelta(seconds=time()) - begin
    msg_fmt = '[Current Step: {:9d} | {:02d}:{:02d}:{:02d}] '
    msg = msg_fmt.format(
        cnt_step, dur.seconds//3600, dur.seconds//60, dur.seconds
    )
    last_dur = dur
    print(msg, end='')
    for element in iterable:
        dur = timedelta(seconds=time()) - begin
        if (dur - last_dur).seconds >= 0.1:
            print('\b'*len(msg), end='')
            msg = msg_fmt.format(
                cnt_step, dur.seconds//3600, dur.seconds//60, dur.seconds
            )
            print(msg, end='')
        yield element
        cnt_step += 1
    print('\b'*len(msg), end='')
    print(' '*len(msg), end='')
    print('\b'*len(msg), end='')
    if outermost:
        print()


def train_mnist(network: Network, num_epochs=10, log=100) -> Network:
    '''Train the network on the MNIST data set.'''
    train_sets = MnistDataset()

    for _ in progbar(range(num_epochs), outermost=True):
        with ProgPrinter() as printer:
            for iteration, (images, labels) in progbar(enumerate(train_sets)):
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
                if iteration % log == 0:
                    cost = (sum(gradient**2))/2
                    cost = cost / 32
                    msg = f'Cost: {cost}'
                    printer(msg)
                    #print(msg, end='')
                    #print('\b'*len(msg), end='')
    return network


def evaluate_mnist(network: Network) -> float:
    '''Evaluate the network on the test set for MNIST.'''
    test_sets = MnistDataset(train=False)

    accuracies = []
    for images, labels in progbar(test_sets, outermost=True):
        images = images / 255
        results = network.forward(images)
        predictions = [list(row) for row in results.iter_rows()]
        predictions = Matrix2d(
            [row.index(max(row)) for row in predictions],
            labels.rows, 1
        )
        accuracies.append(sum(labels == predictions))
    return sum(accuracies) / len(test_sets.labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs', help='Number of epochs to run.', default=10, type=int
    )
    parser.add_argument(
        '--log', help='Frequency of logging.', default=100, type=int
    )
    args = parser.parse_args()
    network = Network()
    network.append(Dense(784, 10))
    network.append(Sigmoid())
    train_mnist(network, num_epochs=args.num_epochs, log=args.log)
    print(f'Accuracy: {evaluate_mnist(network)}')


if __name__ == '__main__':
    main()
