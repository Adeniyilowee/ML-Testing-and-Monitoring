from sklearn.model_selection import train_test_split

import pipeline
from processing.data_management import (
    load_dataset,
    save_pipeline,
)
from config.core import config
from src import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # load training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # fit to entire pipeline
    pipeline.landscape_pipe.fit(data)

    _logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.landscape_pipe)


if __name__ == "__main__":
    run_training()
