from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import logging
import sys
 

def main():

    try:
        args = get_args()
        config = process_config(args.config)

        logger = configure_logger(level=logging.DEBUG)
        logger.info('Starting train.py...')
        logger.debug(config)
        
        # create the experiments dirs
        create_dirs([config.callbacks.checkpoint_dir])

        logger.info('Creating the data generator.')
        data_loader = factory.create("data_loaders."+config.data_loader.name)(config)

        # For this project the number of classes is only known
        # at runtime so we add that to the configuration.
        config.n_classes = data_loader.n_classes
        logger.debug('Running with {} classes.'.format(config.n_classes))

        logger.info('Creating model.')
        model = factory.create("models."+config.model.name)(config)


        logging.info('Creating trainer')
        trainer = factory.create("trainers."+config.trainer.name)(model, data_loader, config)

        logging.info('Running trainer')
        trainer.train()
        
        logging.info('Loading evaluators')
        evaluators = []
        for evaluator in config.evaluators:
            evaluators.append(factory.create(
                "evaluators." + evaluator.name
            )(model, data_loader, evaluator))


        logging.info('Evaluating...')
        for evaluator in evaluators:
            evaluator.evaluate()
        
    except Exception as e:
        print(e)
        sys.exit(1)

        
def configure_logger(level):
    """ Logging, very basic. """
    logger = logging.getLogger('train')
    logger.setLevel(level)

    # Can be changed to output file
    console_log = logging.StreamHandler()
    console_log.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    console_log.setFormatter(formatter)

    logger.addHandler(console_log)
    return logger


if __name__ == '__main__':
    main()
