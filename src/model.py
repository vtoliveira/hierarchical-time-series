import pandas as pd 
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator

from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts import RevisionMethod


class HierarchicalModel(BaseEstimator, TransformerMixin):
    """
    Wrapper class to fit and create predictions to be used by
    Revision class in scikit-hts library.
    """

    def __init__(
        self, 
        model,
        nodes  
    ):
        self.model = model
        self.nodes = nodes


    def fit(
        self, 
        y_train, 
        y_test, 
        cv=False, 
        model_params=None,
        forecaster_params=None
    ):
        self.prediction_by_node = list()
        self.prediction_in_sample = list()

        for node in self.nodes:
            y_train_node = y_train[node]
            fh = np.arange(len(y_test)) + 1

            regressor = self.model
            regressor.fit(y_train_node, fh=fh)
            y_pred = regressor.predict(fh)

            self.prediction_by_node.append(y_pred.to_frame(name=node))

    def predict(self):
        preds = pd.concat(self.prediction_by_node, axis=1)

        return preds

    def predict_yhat(self, columns):
        yhat = {}
        preds = self.predict()
        for col in columns:
            yhat[col] = (
                preds[col].to_frame(name='yhat').reset_index(drop=True)
            )

        return yhat

    def predict_revised(self, sum_mat, method, ht, yhat_columns):

        yhat = self.predict_yhat(yhat_columns)
        revision = RevisionMethod(
            sum_mat=sum_mat,
            transformer=None,
            name=method
        )

        pred_revised = revision.revise(forecasts=yhat, nodes=ht)

        return pred_revised

    

