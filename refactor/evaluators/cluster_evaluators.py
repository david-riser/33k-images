import logging
import numpy as np

from base.base_evaluator import BaseEvaluator
from sklearn.utils.linear_assignment_ import linear_assignment

class LinearAssignmentMetricEvaluator(BaseEvaluator):
    """

    A simple linear assignment between assigned labels
    and ground truth labels.  Several metrics are calculated.

    Expected format: 
    {
        "name":"cluster_evaluators.LinearAssignmentMetricEvaluator",
        "metrics":["accuracy", "balanced_accuracy", "f1_score"]
    },

    """
    def __init__(self, model, data, config):
        super(LinearAssignmentMetricEvaluator, self).__init__(model, data, config)
        self.logger = logging.getLogger('train')
        self.metrics = { key:np.inf for key in self.config.metrics }
        
        
    def evaluate(self):

        if self.config.n_classes != self.config.model.n_clusters:
            self.logger.debug("Cannot run linear assignment with unequal number of classes/clusters.")
            return 


        self.logger.info("Starting linear assignment.")
        x_dev, y_dev = self.data.get_dev_data()
        y_pred = self.model.model.predict(x_dev)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        y_pred_reassigned = y_pred.copy()
        mappings = self._hungarian_method(y_dev, y_pred)

        for (source, dest) in mappings:
            indices = np.where(y_pred == source)[0]
            y_pred_reassigned[indices] = dest

        for metric in self.metrics:
            if metric == "accuracy":
                self.metrics[metric] = accuracy(y_dev, y_pred_reassigned)

            elif metric == "balanced_accuracy":
                self.metrics[metric] = balanced_accuracy(y_dev, y_pred_reassigned)

            elif metric == "f1_score":
                self.metrics[metric] = f1_score(y_dev, y_pred_reassigned)

        for metric, value in self.metrics.items():
            self.logger.info("{0} = {1:6.4f}".format(metric, value))
                
    def _hungarian_method(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert (y_pred.size == y_true.size)
        dim = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((dim,dim), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_assignment(w.max() - w)
        return ind
        
