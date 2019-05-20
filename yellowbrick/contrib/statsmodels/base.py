# yellowbrick.contrib.statsmodels.base
# A basic wrapper for statsmodels that emulates a scikit-learn estimator.
#
# Author:  Ian Ozsvald
# Created: Wed Jan 10 12:47:00 2018 -0500
#
# ID: base.py [] benjamin@bengfort.com $

"""
A basic wrapper for statsmodels that emulates a scikit-learn estimator.
"""

##########################################################################
## Imports
##########################################################################

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator


##########################################################################
## statsmodels Estimator
##########################################################################

class StatsModelsWrapper(BaseEstimator):
    """
    The StatsModelsWrapper wraps a statsmodels GLM as a scikit-learn (fake) BaseEstimator for YellowBrick.
  

    Parameters
    ----------
    glm_partial : a partial function
        A partial function that contains the statsmodel model and model family

    stated_estimator_type : string, default: regressor
        The feature name that corresponds to a column name or index postion
        in the matrix that will be plotted against the x-axis

    scorer : object, scikit-learn scoring metric, default: sklearn.metrics.r2_score
        A scikit-learn scoring function


    """
    def __init__(self, glm_partial, stated_estimator_type="regressor",
                 scorer=r2_score):

        # YellowBrick checks the attribute to see if it is a
        # regressor/clusterer/classifier
        self._estimator_type = stated_estimator_type

        # assume user passes in a partial which we can instantiate later
        self.glm_partial = glm_partial

        # needs a default scoring function, regression uses r^2 in sklearn
        self.scorer = scorer

    def fit(self, X, y):
        """
        Pretend to be a sklearn estimator, fit is called on creation

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values
        """

        # note that GLM takes endog (y) and then exog (X):
        # this is the reverse of sklearn's methods
        self.glm_model = self.glm_partial(y, X)
        self.glm_results = self.glm_model.fit()
        return self

    def predict(self, X):
        """
        Predict the labels of X

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values
        """

        return self.glm_results.predict(X)

    def score(self, X, y):
        """
        Scoring the the quality of predictions against a score function.
        
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        """
        return self.scorer(y, self.predict(X))


def decisionviz(model,
                X,
                y,
                family=None,
                colors=None,
                classes=None,
                features=None,
                show_scatter=True,
                step_size=0.0025,
                markers=None,
                pcolormesh_alpha=0.8,
                scatter_alpha=1.0,
                title=None,
                **kwargs):
    """


    """
    statsmodels_partial = partial(model, family=sm.families.Gaussian())
    statsmodel_estimator = StatsModelsWrapper(statsmodels_partial)

    viz = PredictionError(sm_est, title="General Linear Model")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)

    # Fit, draw and poof the visualizer
    viz.fit_draw_poof(X, y, **kwargs)

    # Return the axes object on the visualizer
    return viz.ax

