from unittest.mock import Mock
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.pipeline import Pipeline



def test_compare_base():
    # Compare the default tags value for _DEFAULT_TAGS with the one used in basecase
    expected_tags = {
    "non_deterministic": False,
    "requires_positive_X": False,
    "requires_positive_y": False,
    "X_types": ["2darray"],
    "poor_score": False,
    "no_validation": False,
    "multioutput": False,
    "allow_nan": False,
    "stateless": False,
    "multilabel": False,
    "_skip_test": False,
    "_xfail_checks": False,
    "multioutput_only": False,
    "binary_only": False,
    "requires_fit": True,
    "preserves_dtype": [np.float64],
    "requires_y": False,
    "pairwise": False,
    }
    base_estimator = BaseEstimator()

    base_estimator.__sklearn_tags__ = Mock(side_effect=base_estimator.__sklearn_tags__)
    tags_result = base_estimator.__sklearn_tags__()

    base_estimator.__sklearn_tags__.assert_called() 

    assert expected_tags == tags_result

def test_type_error_exception_glm():
    # Test throwing TypeError for glm for __sklearn_tags__ in the case that _get_loss throws an error because of
    # for instance TweedieRegressor
    expected_result = {"X_types": ["2darray"]}
    generalized_linear_regressor = _GeneralizedLinearRegressor()

    generalized_linear_regressor._get_loss = Mock(side_effect=TypeError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results

def test_value_error_exception_glm():
    # Test throwing ValueError for glm for __sklearn_tags__ in the case that _get_loss throws an error because of
    # for instance TweedieRegressor
    expected_result = {"X_types": ["2darray"]}
    generalized_linear_regressor = _GeneralizedLinearRegressor()

    generalized_linear_regressor._get_loss = Mock(side_effect=ValueError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results

def test_attribute_error_exception_glm():
    # Test throwing AttributeError for glm for __sklearn_tags__ in the case that _get_loss throws an error because of
    # for instance TweedieRegressor
    expected_result = {"X_types": ["2darray"]}
    generalized_linear_regressor = _GeneralizedLinearRegressor()

    generalized_linear_regressor._get_loss = Mock(side_effect=AttributeError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results

def test_type_error_exception_pipeline():
    # Test throwing TypeError for glm for __sklearn_tags__ in the case `steps` is not a list of (name, estimator)
    # tuples and `fit` is not called yet to validate the steps.
    expected_result = {"X_types": ["2darray"]}
    steps = 0
    generalized_linear_regressor = Pipeline(steps)

    generalized_linear_regressor._safe_tags = Mock(side_effect=TypeError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results

def test_value_error_exception_pipeline():
    # Test throwing ValueError for glm for __sklearn_tags__ in the case `steps` is not a list of (name, estimator)
    # tuples and `fit` is not called yet to validate the steps.
    expected_result = {"X_types": ["2darray"]}
    steps = 0
    generalized_linear_regressor = Pipeline(steps)

    generalized_linear_regressor._safe_tags = Mock(side_effect=ValueError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results

def test_attribute_error_exception_pipeline():
    # Test throwing AttributeError for glm for __sklearn_tags__ in `steps` is not a list of (name, estimator)
    # tuples and `fit` is not called yet to validate the steps.
    expected_result = {"X_types": ["2darray"]}
    steps = 0
    generalized_linear_regressor = Pipeline(0)

    generalized_linear_regressor._safe_tags = Mock(side_effect=AttributeError())
    tag_results = generalized_linear_regressor.__sklearn_tags__()
        
    assert expected_result == tag_results
