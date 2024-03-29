import gzip
import math
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, download_url, check_integrity
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from urllib.error import URLError

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    path_label = os.path.join(root, 'train-labels-idx1-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28) # Shape: [60000, 28, 28]
    with gzip.open(path_label, 'rb') as f:
        mnist_label = np.frombuffer(f.read(), np.uint8, offset=8) # Shape: [60000,]

    return mnist, mnist_label


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
        'http://www.cs.toronto.edu/~nitish/unsupervised_video/',
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("mnist_test_seq.npy", "be083ec986bfe91a449d63653c411eb2"),
    ]
    val_dataset = 'mmnist_val.pt'
    def __init__(self, root, train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                static_prob=-1,
                download=False,
                transform=None,
                length=int(1e4),
                val=False):
        '''
        Args:
            `root`: Root directory of the dataset (mnist dataset and moving mnist test set)
            `num_objects`: a list of number of **possible** objects. 
            `train`: generate data when is True or `num_objects`!=2, otherwise load the standard test dataset. 
            `n_frames_input`: (>0) number of frames to input.
            `n_frames_output`: (>0) number of frames to output.

        Dataset size is 10k if generate data, otherwise ...

        Sample shape:
            tuple of:
            `labels`: [num_objects,]
            `input_frames`: [n_frames_input, 1, image_size, image_size]
            `output_frames`: [n_frames_output, 1, image_size, image_size]
        '''
        super(MovingMNIST, self).__init__()
        self.root = root
        self.is_train = train
        if not self.is_train:
            self.is_val = val
        else:
            self.is_val = False

        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.dataset = None
        if train:
            self.mnist, self.mnist_label = load_mnist(root)
        elif self.is_val:
            if num_objects[0] != 2:
                self.mnist, self.mnist_label = load_mnist(root)
            else:
                self.dataset = torch.load(os.path.join(root, self.val_dataset))
        else:
            if num_objects[0] != 2:
                self.mnist, self.mnist_label = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        if self.dataset is None:
            self.length = length
        elif self.is_val:
            self.length = len(self.dataset)
        else:
            self.length = self.dataset.shape[1]

        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1
        self.static_prob = static_prob

    def get_random_trajectory(self, seq_length, velocity_dev=0.05, static_prob=-1):
        ''' Generate a random sequence of a MNIST digit 
        
        Args:
            `seq_length`: length of the sequence
            `velocity_dev`: maximum deviation of the velocity, default 0.05
            `static_prob`: probability of a static position, default -1 (never)
        '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        if random.random() >= static_prob:
            v_mag = random.random() * velocity_dev + 1
        else:
            v_mag = 0.
        v_y = np.sin(theta)*v_mag
        v_x = np.cos(theta)*v_mag

        start_y = np.zeros(seq_length) # sequence of positions
        start_x = np.zeros(seq_length) # sequence of positions
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        ind_data = np.zeros((max(self.num_objects), self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        labels = np.ones((max(self.num_objects),), dtype=np.int64)*(-1) # default value 
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total, static_prob=self.static_prob)
            ind = random.randint(0, self.mnist.shape[0] - 1) # randomly select an index for a mnist digit image
            digit_image = self.mnist[ind] # the corresponding image, shape: [28, 28]
            digit_label = self.mnist_label[ind] # the corresponding label, shape: []
            labels[n] = digit_label
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image) # addition
                ind_data[n, i, top:bottom, left:right] = digit_image

        data = data[..., np.newaxis]
        ind_data = ind_data[..., np.newaxis]
        return data, ind_data, labels

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        labels = np.ones((max(self.num_objects),), dtype=np.int64)*(-1) # default value 
        ind_images = None
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images, ind_images, labels = self.generate_moving_mnist(num_digits)
        elif self.is_val:
            labels, input, output, ind_images = *self.dataset[idx],
            return labels, input, output, ind_images
        else:
            images = self.dataset[:, idx, ...]

        if self.transform is not None:
            images = self.transform(images)
            if ind_images is not None:
                ind_images = self.transform(ind_images)

        r = 1 # patch size (a 4 dans les PredRNN)
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        if ind_images is not None:
            ind_images = ind_images.transpose(0, 1, 4, 2, 3) # [N, T, C, H, W]
            ind_images = torch.from_numpy(ind_images / 255.0).contiguous().float()

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        labels = torch.from_numpy(labels).contiguous().long() # Shape: [num_digits]
        
        if ind_images is None:
            ind_images = torch.cat([input, output], dim=0)

        out = [labels,input,output, ind_images]
        return out

    def __len__(self):
        return self.length

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.root, filename), md5=md5)
            for filename, md5 in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_url(url, root=self.root, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

def main():
    train_set = MovingMNIST(
        root='./data',
        train=True,
        n_frames_input=10,
        n_frames_output=10,
        num_objects=[2],
        static_prob=0.5,
        download=False
    )
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    for idx, samples in enumerate(tqdm(train_loader)):
        print(samples[0].shape, samples[1].shape, samples[2].shape, samples[3].shape)
        break
    labels, v_in, v_target = next(iter(train_set)) # v_in.size() = (10, 1, 64, 64)
    show = make_grid([*v_in, *v_target], nrow=10)
    print(labels)
    plt.imshow(show.numpy().transpose(1, 2, 0))
    plt.show()
    pass


if __name__ == '__main__':
    main()