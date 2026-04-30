'''
Concrete IO class for stage 3 image datasets (ORL, MNIST, CIFAR-10).
'''

from local_code.base_class.dataset import dataset
import os
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    use_single_channel_for_orl = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _normalize_orl_shape(self, image):
        if self.use_single_channel_for_orl and isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3:
            # ORL is grayscale duplicated in RGB channels; keep one channel.
            return image[:, :, 0]
        return image

    def load_file(self, file_name=None):
        print('loading data...')
        source_file_name = file_name if file_name is not None else self.dataset_source_file_name
        full_path = os.path.join(self.dataset_source_folder_path, source_file_name)
        with open(full_path, 'rb') as f:
            loaded_dataset = pickle.load(f)

        train_X, train_y = [], []
        test_X, test_y = [], []

        for instance in loaded_dataset['train']:
            image = np.asarray(instance['image'])
            image = self._normalize_orl_shape(image)
            train_X.append(image)
            train_y.append(int(instance['label']))

        for instance in loaded_dataset['test']:
            image = np.asarray(instance['image'])
            image = self._normalize_orl_shape(image)
            test_X.append(image)
            test_y.append(int(instance['label']))

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y}
        }

    def load(self):
        return self.load_file()
