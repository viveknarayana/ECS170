'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name=None):
        print('loading data...')
        X = []
        y = []
        source_file_name = file_name if file_name is not None else self.dataset_source_file_name
        f = open(self.dataset_source_folder_path + source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            y.append(elements[0])
            X.append(elements[1:])
        f.close()
        return {'X': X, 'y': y}

    def load(self):
        return self.load_file()
