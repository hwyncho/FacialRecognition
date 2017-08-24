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
    import os
    import zipfile
    from random import shuffle

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
        datsets = Datasets()

        datasets_zip = zipfile.ZipFile(path)
        datasets_zip.extractall('./')
        datasets_zip.close()

        dataset_list = []
        for dataset_name in os.listdir('./Datasets'):
            if dataset_name.endswith('.json'):
                dataset_list.append(dataset_name)
        dataset_list.sort()

        for dataset in dataset_list:
            with open('./Datasets/{}'.format(dataset), 'r', encoding='utf-8') as f:
                loaded_json = json.load(f)

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

            os.remove('./Datasets/{}'.format(dataset))
        os.removedirs('./Datasets')

        print('Dataset Load complete!')

        return datsets
    else:
        print("The path '{}' does not exist.".format(path))
