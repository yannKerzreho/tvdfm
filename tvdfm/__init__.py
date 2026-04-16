"""
tvdfm — Time-Varying Dynamic Factor Model
==========================================

Main entry point::

    from tvdfm import TVDFModel

    model = TVDFModel(n_factors=3, exposure="ncde")
    model.fit(df_obs, covariates=df_cov)
    predictions = model.predict()
    factors     = model.transform()
"""

from .model import TVDFModel
from .core import TVDFM
from .ssm import DFMStateSpace
from .training import LTVTrainingManager, loss_e2e, RegularisationConfig
from .exposure import (
    AbstractExposure,
    NCDEExposure,
    GRUExposure,
    make_ncde_path,
)
from .utils import (
    to_float_times,
    parse_dataframe,
    parse_covariates,
    extract_statsmodels_params,
)

__all__ = [
    # High-level
    "TVDFModel",
    # Core Equinox module
    "TVDFM",
    # State-space model
    "DFMStateSpace",
    # Training
    "LTVTrainingManager",
    "loss_e2e",
    "RegularisationConfig",
    # Exposure
    "AbstractExposure",
    "NCDEExposure",
    "GRUExposure",
    "make_ncde_path",
    # Utils
    "to_float_times",
    "parse_dataframe",
    "parse_covariates",
    "extract_statsmodels_params",
]

__version__ = "0.1.0"
