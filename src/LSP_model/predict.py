import logging
from lsp_model.config.core import config, PACKAGE_ROOT
from lsp_model.processing import data_management

_logger = logging.getLogger(__name__)

with open(PACKAGE_ROOT / "VERSION") as version_file:
    _version_ = version_file.read().strip()

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version_}.pkl"
_price_pipe = data_management.load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, test_data):
    """Make a prediction using a saved model pipeline."""

    predictions = _price_pipe.predict(X=test_data[config.model_config.features])
    _logger.info(f"Making predictions with model version: {_version_} "
                 f"Predictions: {predictions}")

    return predictions
