import numpy as np # linear algebra
import struct
from array import array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os.path  import join

class MnistDataloader(object):
    """
        MNIST Data Loader Class
    """

    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
    
    def downsample(self, data):
        downsample_factor = 4
        reshaped_arr = data.reshape(-1, downsample_factor)
        downsampled_x = reshaped_arr.mean(axis=1)
        reshaped_x = downsampled_x.reshape(14,14)
        return reshaped_x

    def load_data(self, downsample=False):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_train = [np.asarray(x_train[i]) for i in range(len(x_train))]
        x_test = [np.asarray(x_test[i]) for i in range(len(x_test))]

        if downsample:
            x_train  = [self.downsample(x) for x in x_train]
            x_test = [self.downsample(x) for x in x_test]
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=10000)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def show_images(self, images, title_texts):
        """
            Helper function to show a list of images with their relating titles
        """

        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1    
        for x in zip(images, title_texts):        
            image = x[0]        
            title_text = x[1]
            plt.subplot(rows, cols, index)        
            plt.imshow(image)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15);        
            index += 1
    
if __name__ == "__main__":
    input_path = './mnist'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = mnist_dataloader.load_data()
