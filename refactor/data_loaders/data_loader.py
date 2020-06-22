import logging
import numpy as np
import os
import pandas as pd

from base.base_data_loader import BaseDataLoader
from utils.factory import create
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

class DataLoader(BaseDataLoader):

    def __init__(self, config):
        super(DataLoader, self).__init__(config)

    def build_files_dataframe(self, distr):
        """ Search a directory tree to build a list of files 
        and their labels. """
        files, labels = [], []
        
        for folder, _, the_files in os.walk(os.path.join(self.config.data_loader.base_dir, distr)):
            if folder != self.config.data_loader.base_dir:
                label = folder.split('/')[-1]
                for f in the_files:
                    labels.append(label)
                    files.append(os.path.join(label,f))

        data = pd.DataFrame({'file':files, 'label':labels})
        return data


    def prune_files_list(self, data):
        """ Remove classes which contain less than min_samples. """
        return_cols = list(data.columns)    
        keep = data.groupby('label').transform(
            lambda x: len(x) > self.config.data_loader.min_samples
        ).values

        return data.iloc[keep][return_cols]


    def drop_other_classes(self, data, classes):
        """ Drop any other class from the dataframe 
        that does not exist in the classes list. """
        keep = data['label'].apply(
            lambda x: x in classes
        )
        return data.iloc[np.where(keep == True)[0]]

    
class InMemoryDataLoader(DataLoader):

    def __init__(self, config):
        super(InMemoryDataLoader, self).__init__(config)

        self.logger = logging.getLogger('train')
        
        if 'preprocessing_function' in self.config.data_loader.toDict():
            self.preprocess_func = create("tensorflow.keras.applications.{}".format(
                self.config.data_loader.preprocessing_function))
        else:
            self.preprocess_func = lambda x: x / 255.

        if 'augmentations' in self.config.data_loader.toDict():
            self.augmentations = self.config.data_loader.augmentations.toDict()
        else:
            self.augmentations = {}

        # Encoder for the labels..
        self.label_encoder = LabelEncoder()

        self.train_dataframe = self.build_files_dataframe('train')
        self.train_dataframe = self.prune_files_list(self.train_dataframe)        
        self.dev_dataframe = self.build_files_dataframe('dev')
        self.test_dataframe = self.build_files_dataframe('test')

        # Shuffle the ordering
        self.train_dataframe = self.train_dataframe.sample(frac=1).reset_index(drop=True)
        self.dev_dataframe = self.dev_dataframe.sample(frac=1).reset_index(drop=True)
        self.test_dataframe = self.test_dataframe.sample(frac=1).reset_index(drop=True)
        
        # Drop the classes that are not in the
        # training set from test and dev. 
        self.classes = np.unique(self.train_dataframe['label'])
        self.logger.debug('Training classes are {}'.format(self.classes))
        self.dev_dataframe = self.drop_other_classes(self.dev_dataframe, self.classes)
        self.test_dataframe = self.drop_other_classes(self.test_dataframe, self.classes)

        # If the user asked us to do some upsampling, we should
        # do that before loading the images. 
        if 'images_per_class' in self.config.data_loader.toDict():
            self.train_dataframe = self._resample(self.train_dataframe)
            self.logger.info('Resampled to size {}'.format(
                self.config.data_loader.images_per_class
            ))
            
            # Load and preprocess the dataset. 
        self.load()
        self.preprocess()

        # Setup generators and flows for training
        self.train_gen = ImageDataGenerator(**self.augmentations)
        self.train_flow = self.train_gen.flow(self.X_train, self.Y_train,
                                              batch_size=self.config.data_loader.batch_size)
        
        
    def load(self):
        """ Load all of the images, this is time/memory consuming. """
        self.X_train = np.zeros((len(self.train_dataframe), 224, 224, 3))
        self.X_dev = np.zeros((len(self.dev_dataframe), 224, 224, 3))
        self.X_test = np.zeros((len(self.test_dataframe), 224, 224, 3))
        self.Y_train = self.label_encoder.fit_transform(self.train_dataframe['label'])
        self.Y_dev = self.label_encoder.transform(self.dev_dataframe['label'])
        self.Y_test = self.label_encoder.transform(self.test_dataframe['label'])
        
        for i in range(len(self.train_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/train',
                                self.train_dataframe['file'].values[i])
            self.X_train[i, :, :, :] = load_img(path, target_size=(224,224))

        for i in range(len(self.dev_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/dev',
                                self.dev_dataframe['file'].values[i])
            self.X_dev[i, :, :, :] = load_img(path, target_size=(224,224))

        for i in range(len(self.test_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/test',
                                self.test_dataframe['file'].values[i])
            self.X_test[i, :, :, :] = load_img(path, target_size=(224,224))
            

    def preprocess(self):
        self.logger.info('Preprocessing train, dev, and test.')
        self.X_train = self.preprocess_func(self.X_train)
        self.X_dev = self.preprocess_func(self.X_dev)
        self.X_test = self.preprocess_func(self.X_test)
        

    def _resample(self, df):
        """ This method resamples all of the classes to a fixed 
        number of images.  It will upsample and downsample. """

        dataframes = []
        for c in self.classes:
            dataframes.append(
                resample(
                    df,
                    replace=True,
                    n_samples=self.config.data_loader.images_per_class
                )
            )

        # Replace the training dataframe by the resampled stuffs
        # here.
        return pd.concat(dataframes)
        
        
    @property
    def n_classes(self):
        return len(self.classes)

    def get_train_data(self):
        return (self.X_train, self.Y_train)

    def get_dev_data(self):
        return (self.X_dev, self.Y_dev)

    def get_test_data(self):
        return (self.X_test, self.Y_test)

    def get_train_flow(self):
        return self.train_flow
