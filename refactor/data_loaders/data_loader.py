import pandas as pd

from base.base_data_loader import BaseDataLoader
from utils.factory import create
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader).__init__(config)

    def build_files_dataframe(self, distr):
        files, labels = [], []

        for folder, _, the_files in os.walk(os.path.join(self.config.data_loader.base_dir, distr)):
            if folder != self.config.data_loader.base_dir:
                label = folder.split('/')[-1]
                for f in the_files:
                    labels.append(label)
                    files.append(os.path.join(label,f))

        data = pd.DataFrame({'file':files, 'label':labels})
        return data


    def prune_files_list(self):
    """     
    Given a list of files, prune the list to contain 
    only the entries for which the label specified by
    label_col contains at least min_samples.
    
    :param data: Dataframe containing at least one column 
    called label_col
    :param label_col: The column used to specify class
    :param min_samples: The minimum number of samples a
    class needs to have to be kept.
    """
    return_cols = list(data.columns)    
    keep = data.groupby(label_col).transform(
        lambda x: len(x) > min_samples
    ).values
    return data.iloc[keep][return_cols]


class InMemoryDataLoader(DataLoader):
    def __init__(self, config):
        super(InMemoryDataLoader).__init__(config)
        
        if 'preprocessing_function' in self.config.data_loader.toDict():
            self.preprocess_func = create("tensorflow.keras.applications.{}".format(
                self.config.data_loader.preprocessing_function))
        else:
            self.preprocess_func = lambda x: x / 255.

        self.load()
        self.preprocess()


    def load(self):
        self.config.data_loder.base_dir
