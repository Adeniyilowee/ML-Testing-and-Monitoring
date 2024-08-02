import logging
from lsp_model.config.core import config
from lsp_model.processing import data_management
from lsp_model import __version__ as _version

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = data_management.load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, test_data):
    """Make a prediction using a saved model pipeline."""

    predictions = _price_pipe.predict(X=test_data[config.model_config.features])
    _logger.info(f"Making predictions with model version: {_version} "
                 f"Predictions: {predictions}")

    return predictions
