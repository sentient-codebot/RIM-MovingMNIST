import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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
    def __init__(self, root, train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                static_prob=-1,
                download=False,
                transform=None):
        '''
        Args:
            `root`: Root directory of the dataset (mnist dataset and moving mnist test set)
            `num_objects`: a list of number of **possible** objects. 
            `train`: generate data when is True or `num_objects`!=2, otherwise load the standard test dataset. 
            `n_frames_input`: (>0) number of frames to input.
            `n_frames_output`: (>0) number of frames to output.

        Sample shape:
            tuple of:
            `labels`: [num_objects,]
            `input_frames`: [n_frames_input, 1, image_size, image_size]
            `output_frames`: [n_frames_output, 1, image_size, image_size]
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if train:
            self.mnist, self.mnist_label = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist, self.mnist_label = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = train
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
        labels = np.zeros((num_digits,), dtype=np.int64)
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
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data, labels

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        labels = None
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images, labels = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        if self.transform is not None:
            images = self.transform(images)

        r = 1 # patch size (a 4 dans les PredRNN)
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        labels = torch.from_numpy(labels).contiguous().long() # Shape: [num_digits]

        out = [labels,input,output]
        return out

    def __len__(self):
        return self.length

def main():
    train_set = MovingMNIST(
        root='./data/',
        train=True,
        n_frames_input=10,
        n_frames_output=10,
        num_objects=[2],
        static_prob=0.5
    )
    labels, v_in, v_target = next(iter(train_set)) # v_in.size() = (10, 1, 64, 64)
    show = make_grid([*v_in, *v_target], nrow=10)
    print(labels)
    plt.imshow(show.numpy().transpose(1, 2, 0))
    plt.show()
    pass


if __name__ == '__main__':
    main()