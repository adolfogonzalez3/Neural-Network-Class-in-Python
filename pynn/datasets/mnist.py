
import math
import shutil
import gzip
from pathlib import Path
from urllib.request import urlopen
from functools import partial
from array import array
from concurrent.futures import ThreadPoolExecutor

from pynn import Matrix2d
from collections.abc import Sequence

MNIST_URL = "http://yann.lecun.com/exdb/mnist"
MNIST_URLS = {
    "train_images": f"{MNIST_URL}/train-images-idx3-ubyte.gz",
    "train_labels": f"{MNIST_URL}/train-labels-idx1-ubyte.gz",
    "test_images": f"{MNIST_URL}/t10k-images-idx3-ubyte.gz",
    "test_labels": f"{MNIST_URL}/t10k-labels-idx1-ubyte.gz"
}


def download(file_name: Path, url_name: str):
    file_name = Path(file_name)
    with urlopen(url_name) as response, file_name.open('wb') as out_file:
        shutil.copyfileobj(response, out_file)


def download_mnist(path: Path):
    path.mkdir()
    paths, urls = zip(*[
        (path / f'{name}-ubyte.gz', url) for name, url in MNIST_URLS.items()
    ])
    with ThreadPoolExecutor() as executor:
        list(executor.map(download, paths, urls))
    # for name, url in MNIST_URLS.items():
    #    download(path / f'{name}-ubyte.gz', url)


def read_idx(bytefile) -> Matrix2d:
    '''Read in IDX files.'''
    bytefile.read(2)
    type_encoding = bytefile.read(1)
    if type_encoding == b'\x08':
        matrix = array('B')
    elif type_encoding == b'\x09':
        matrix = array('b')
    elif type_encoding == b'\x0B':
        matrix = array('h')
    elif type_encoding == b'\x0C':
        matrix = array('l')
    elif type_encoding == b'\x0D':
        matrix = array('f')
    elif type_encoding == b'\x0B':
        matrix = array('d')
    else:
        raise RuntimeError(f'Encoding not known: {type_encoding}')

    dimensions = int.from_bytes(bytefile.read(1), byteorder='big')
    shape = tuple([
        int.from_bytes(bytefile.read(4), byteorder='big')
        for _ in range(dimensions)
    ])
    size = 1
    for shpe in shape:
        size *= shpe
    matrix.fromfile(bytefile, size)
    return Matrix2d(matrix, shape[0], size//shape[0])


class MnistDataset(Sequence):
    def __init__(self, mnist_path=None, train=True, batch_size=32):
        self.batch_size = batch_size
        mnist_path = mnist_path or Path("mnist")
        if not mnist_path.exists():
            download_mnist(mnist_path)
        train = "train" if train else "test"
        images_path = mnist_path / f'{train}_images-ubyte.gz'
        labels_path = mnist_path / f'{train}_labels-ubyte.gz'
        with gzip.open(images_path, 'rb') as bytefile:
            self.images = read_idx(bytefile)
        with gzip.open(labels_path, 'rb') as bytefile:
            self.labels = read_idx(bytefile)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return math.ceil(self.images.rows / self.batch_size)

    def __getitem__(self, idx):
        return (
            self.images[idx*self.batch_size:(idx+1)*self.batch_size, :],
            self.labels[idx*self.batch_size:(idx+1)*self.batch_size, :]
        )
