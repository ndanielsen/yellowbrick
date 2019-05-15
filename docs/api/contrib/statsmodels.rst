.. -*- mode: rst -*-

StatsModels Visualizers
=======================

The StatsModelsWrapper wraps a statsmodels GLM and gives it a facade interface (scikit-learn BaseEstimator) for use with Yellowbrick. 

.. note:: This wrapper is trivial and under development, options and extra things like weights
    are not currently handled.

.. plot::
    :context: close-figs
    :alt: StatsModelsWrapper with statsmodels GLM

    from functools import partial

    import numpy as np
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split

    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import PredictionError
    from yellowbrick.contrib.statsmodels import StatsModelsWrapper

    # Use Yellowbrick to load the concrete dataset
    X, y = load_concrete()

    # Create the train and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate a partial with the statsmodels API
    glm_gaussian_partial = partial(sm.GLM, family=sm.families.Gaussian())
    sm_est = StatsModelsWrapper(glm_gaussian_partial)

    # Create a Yellowbrick visualizer to visualize prediction error:
    viz = PredictionError(sm_est, title="General Linear Model")
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()

    # For statsmodels usage, calling .summary() etc:
    gaussian_model = glm_gaussian_partial(y_train, X_train)

.. automodule:: yellowbrick.contrib.statsmodels.base
    :members: StatsModelsWrapper
    :undoc-members:
    :show-inheritance:
