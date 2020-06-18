from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys

def main():

    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.checkpoint_dir])

        print('Create the data generator.')
        data_loader = factory.create("data_loaders."+config.data_loader.name)(config)

        print('Create the model.')
        model = factory.create("models."+config.model.name)(config)

        print('Create the trainer')
        trainer = factory.create("trainers."+config.trainer.name)(model, data_loader, config)
        
        print('Loading evaluators')
        evaluators = []
        for evaluator in config.evaluators:
            evaluators.append(factory.create(
                "evaluators." + evaluator.name
            )(model, data_loader, evaluator))

        print('Start training the model.')
        trainer.train()

        print('Evaluating...')
        for evaluator in evaluators:
            evaluator.evaluate()
        
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
