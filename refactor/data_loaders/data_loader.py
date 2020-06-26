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
from tensorflow.keras.utils import to_categorical

class DataLoader(BaseDataLoader):
    """ 

    DataLoader is an abstract class, not to be
    instantiated directly.  The methods defined
    below are common to the other loaders.

    The primary task for loading this dataset is
    to crawl the folders and create a dataframe
    that specifies paths and labels.  

    """
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self.logger = logging.getLogger('train')
        
        self._setup_preprocessing()
        self._setup_augmentations()

        # Crawl the disk and create dataframes with the files
        # and labels.  Then drop any classes which do not have
        # at least min_samples training samples.
        self._create_dataframes()

        # Ensure that the dev and test data have the same classes
        # as the training data.  This means dropping the extras
        # with a low number of samples.
        self.classes = np.unique(self.train_dataframe['label'])
        self._drop_dev_test_classes()
        
        # If the user asked us to do some upsampling, we should
        # do that before loading the images. 
        if 'images_per_class' in self.config.data_loader.toDict():
            self.train_dataframe = self._resample(self.train_dataframe)
            self.logger.info('Resampled to size {}'.format(
                self.config.data_loader.images_per_class
            ))


    def _setup_preprocessing(self):
        """ Setup the preprocessing function specified in the
        configuration or use a 0-1 normalization if none is 
        present. """
        if 'preprocessing_func' in self.config.data_loader.toDict():
            self.preprocess_func = create("tensorflow.keras.applications.{}".format(
                self.config.data_loader.preprocessing_func))
            self.logger.debug("Loaded {} for preprocessing".format(
                self.config.data_loader.preprocessing_func))
        else:
            self.preprocess_func = lambda x: x / 255.
            self.logger.debug("Using standard 0-1 normalization as preprocessing.")

    def _setup_augmentations(self):
        """ Read augmentations from configuration or 
        load an empty dictionary. """
        if 'augmentations' in self.config.data_loader.toDict():
            self.augmentations = self.config.data_loader.augmentations.toDict()
        else:
            self.augmentations = {}
        self.logger.debug("Setup augmentations: {}".format(self.augmentations))


    def _create_dataframes(self):
        # Load the file names from disk and drop the
        # ones which do not have enough samples.
        self.train_dataframe = self._build_files_dataframe('train')
        self.train_dataframe = self._prune_files_list(self.train_dataframe)        
        self.dev_dataframe = self._build_files_dataframe('dev')
        self.test_dataframe = self._build_files_dataframe('test')

        # Shuffle the ordering
        self.train_dataframe = self.train_dataframe.sample(frac=1).reset_index(drop=True)
        self.dev_dataframe = self.dev_dataframe.sample(frac=1).reset_index(drop=True)
        self.test_dataframe = self.test_dataframe.sample(frac=1).reset_index(drop=True)
        
    def _build_files_dataframe(self, distr):
        """ Search a directory tree to build a list of files 
        and their labels. """
        files, labels = [], []

        source_dir = os.path.join(self.config.data_loader.base_dir, distr)
        for folder, _, the_files in os.walk(source_dir):
            if folder != source_dir:
                label = folder.split('/')[-1]
                for f in the_files:
                    labels.append(label)
                    files.append(os.path.join(label,f))

        data = pd.DataFrame({'file':files, 'label':labels})
        return data


    def _prune_files_list(self, data):
        """ Remove classes which contain less than min_samples. """
        keep = data.groupby('label').transform(
            lambda x: len(x) > self.config.data_loader.min_samples
        ).values
        return data.iloc[keep]


    def _drop_dev_test_classes(self):
        self.dev_dataframe = self._drop_other_classes(self.dev_dataframe, self.classes)
        self.test_dataframe = self._drop_other_classes(self.test_dataframe, self.classes)
        self.logger.debug('Training classes are {}'.format(self.classes))
        self.logger.debug('Dev classes are {}'.format(np.unique(self.dev_dataframe['label'])))
        self.logger.debug('Test classes are {}'.format(np.unique(self.test_dataframe['label'])))

        
    def _drop_other_classes(self, data, classes):
        """ Drop any other class from the dataframe 
        that does not exist in the classes list. """
        keep = data['label'].apply(
            lambda x: x in classes
        )
        return data.iloc[np.where(keep == True)[0]]

 
    def _resample(self, df):
        """ This method resamples all of the classes to a fixed 
        number of images.  It will upsample and downsample. """

        dataframes = []
        for c in self.classes:
            indices = np.where(df['label'] == c)[0]
            dataframes.append(
                resample(
                    df.iloc[indices],
                    replace=True,
                    n_samples=self.config.data_loader.images_per_class
                )
            )

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
 
    def get_dev_flow(self):
        return self.dev_flow

    def get_test_flow(self):
        return self.test_flow

    
class InMemoryDataLoader(DataLoader):

    def __init__(self, config):
        super(InMemoryDataLoader, self).__init__(config)            
        self.load()
        self.preprocess()

        self.train_gen = ImageDataGenerator(**self.augmentations)
        self.train_flow = self.train_gen.flow(self.X_train, self.Y_train,
                                              batch_size=self.config.data_loader.batch_size)
        
        dev_augs = {}
        self.dev_gen = ImageDataGenerator(**dev_augs)
        self.dev_flow = self.dev_gen.flow(self.X_dev, self.Y_dev,
                                          batch_size=self.config.data_loader.batch_size)

        self.test_gen = ImageDataGenerator(**dev_augs)
        self.test_flow = self.test_gen.flow(self.X_test, self.Y_test,
                                          batch_size=self.config.data_loader.batch_size)
        
    def load(self):
        """ Load all of the images, this is time/memory consuming. """
        self.X_train = np.zeros((len(self.train_dataframe), 224, 224, 3))
        self.X_dev = np.zeros((len(self.dev_dataframe), 224, 224, 3))
        self.X_test = np.zeros((len(self.test_dataframe), 224, 224, 3))

        self.label_encoder = LabelEncoder()
        self.Y_train = to_categorical(self.label_encoder.fit_transform(self.train_dataframe['label']))
        self.Y_dev = to_categorical(self.label_encoder.transform(self.dev_dataframe['label']))
        self.Y_test = to_categorical(self.label_encoder.transform(self.test_dataframe['label']))
        
        for i in range(len(self.train_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/train',
                                self.train_dataframe.iloc[i]['file'])
            self.X_train[i, :, :, :] = load_img(path, target_size=(224,224))

        for i in range(len(self.dev_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/dev',
                                self.dev_dataframe.iloc[i]['file'])
            self.X_dev[i, :, :, :] = load_img(path, target_size=(224,224))

        for i in range(len(self.test_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/test',
                                self.test_dataframe.iloc[i]['file'])
            self.X_test[i, :, :, :] = load_img(path, target_size=(224,224))

    def preprocess(self):
        self.logger.info('Preprocessing train, dev, and test.')
        self.X_train = self.preprocess_func(self.X_train)
        self.X_dev = self.preprocess_func(self.X_dev)
        self.X_test = self.preprocess_func(self.X_test)
            

class DiskDataLoader(DataLoader):

    def __init__(self, config):
        super(DiskDataLoader, self).__init__(config)
        self.load()
        self.preprocess()

        self.X_train = None
        self.augmentations['preprocessing_function'] = self.preprocess_func
        self.train_gen = ImageDataGenerator(**self.augmentations)
        self.train_flow = self.train_gen.flow_from_dataframe(
            dataframe=self.train_dataframe,
            directory=os.path.join(self.config.data_loader.base_dir, 'train'),
            batch_size=self.config.data_loader.batch_size,
            target_size=(224,224),
            shuffle=True,
            x_col='file',
            y_col='label',
            class_mode='categorical'
        )
        
        dev_augs = {}
        self.dev_gen = ImageDataGenerator(**dev_augs)
        self.dev_flow = self.dev_gen.flow(self.X_dev, self.Y_dev,
                                          batch_size=self.config.data_loader.batch_size)

        self.test_gen = ImageDataGenerator(**dev_augs)
        self.test_flow = self.test_gen.flow(self.X_test, self.Y_test,
                                          batch_size=self.config.data_loader.batch_size)

        
    def load(self):
        """ The dev and test dataset are always loaded into memory. """
        self.X_dev = np.zeros((len(self.dev_dataframe), 224, 224, 3))
        self.X_test = np.zeros((len(self.test_dataframe), 224, 224, 3))

        self.label_encoder = LabelEncoder()
        self.Y_train = to_categorical(self.label_encoder.fit_transform(self.train_dataframe['label']))
        self.Y_dev = to_categorical(self.label_encoder.transform(self.dev_dataframe['label']))
        self.Y_test = to_categorical(self.label_encoder.transform(self.test_dataframe['label']))
        
        for i in range(len(self.dev_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/dev',
                                self.dev_dataframe.iloc[i]['file'])
            self.X_dev[i, :, :, :] = load_img(path, target_size=(224,224))

        for i in range(len(self.test_dataframe)):
            path = os.path.join(self.config.data_loader.base_dir + '/test',
                                self.test_dataframe.iloc[i]['file'])
            self.X_test[i, :, :, :] = load_img(path, target_size=(224,224))
        

    def preprocess(self):
        self.logger.info('Preprocessing dev and test.')
        self.X_dev = self.preprocess_func(self.X_dev)
        self.X_test = self.preprocess_func(self.X_test)
        
