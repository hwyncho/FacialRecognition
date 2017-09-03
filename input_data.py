def _has_path(path):
    import os

    return os.path.exists(path)


def load_dataset(path='./Datasets.zip'):
    """
    Load dataset and Return

    Parameters
    ==========
    path : str
        path of dataset file
    balanced : bool
        balance the number of data by class

    Returns
    ==========
    datasets : _Datasets
        loaded dataset
    """
    import json
    import random
    from random import shuffle
    import os
    import zipfile

    class Dataset:
        def __init__(self):
            self._idx = 0
            self._size = 0
            self.images = []
            self.labels = []

        def add_data(self, image, label):
            self._size += 1
            self.images.append(image)
            self.labels.append(label)

        def next_batch(self, batch_size):
            if self._idx >= self._size:
                self._idx = 0

            i = self._idx
            j = self._idx + batch_size
            self._idx += batch_size

            return self.images[i:j], self.labels[i:j]

    class Datasets:
        def __init__(self):
            self.train = Dataset()
            self.test = Dataset()

    if _has_path(path):
        par_dir = os.path.split(path)
        temp_dir = '{}/Datasets'.format(par_dir[0])

        datsets = Datasets()

        datasets_zip = zipfile.ZipFile(path)
        datasets_zip.extractall(temp_dir)
        datasets_zip.close()

        dataset_list = []
        for dataset_name in os.listdir(temp_dir):
            if dataset_name.endswith('.json'):
                dataset_list.append(dataset_name)
        dataset_list.sort()

        for dataset in dataset_list:
            with open('{0}/{1}'.format(temp_dir, dataset), 'r', encoding='utf-8') as f:
                loaded_json = json.load(f)

            random.seed(777)
            shuffle(loaded_json)
            if 'train' in dataset:
                for data in loaded_json:
                    datsets.train.add_data(
                        image=data['image'],
                        label=data['label']
                    )
            elif 'test' in dataset:
                for data in loaded_json:
                    datsets.test.add_data(
                        image=data['image'],
                        label=data['label']
                    )

            os.remove('{0}/{1}'.format(temp_dir, dataset))
        os.removedirs(temp_dir)

        print('Dataset Load complete!')

        return datsets
    else:
        print("The path '{}' does not exist.".format(path))
