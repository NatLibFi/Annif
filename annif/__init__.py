#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import os.path
from typing import TYPE_CHECKING

logging.basicConfig()
logger = logging.getLogger("annif")
logger.setLevel(level=logging.INFO)

import annif.backend  # noqa

if TYPE_CHECKING:
    from flask.app import Flask


def create_flask_app(config_name: str | None = None) -> Flask:
    """Create a Flask app to be used by the CLI."""
    from flask import Flask

    _set_tensorflow_loglevel()

    app = Flask(__name__)
    config_name = _get_config_name(config_name)
    logger.debug(f"creating flask app with configuration {config_name}")
    app.config.from_object(config_name)
    app.config.from_envvar("ANNIF_SETTINGS", silent=True)
    return app


def create_app(config_name: str | None = None) -> Flask:
    """Create a Connexion app to be used for the API."""
    # 'cxapp' here is the Connexion application that has a normal Flask app
    # as a property (cxapp.app)
    import connexion
    from flask_cors import CORS

    from annif.openapi.validation import CustomRequestBodyValidator

    specdir = os.path.join(os.path.dirname(__file__), "openapi")
    cxapp = connexion.App(__name__, specification_dir=specdir)
    config_name = _get_config_name(config_name)
    logger.debug(f"creating connexion app with configuration {config_name}")
    cxapp.app.config.from_object(config_name)
    cxapp.app.config.from_envvar("ANNIF_SETTINGS", silent=True)

    validator_map = {
        "body": CustomRequestBodyValidator,
    }
    cxapp.add_api("annif.yaml", validator_map=validator_map)

    # add CORS support
    CORS(cxapp.app)

    if cxapp.app.config["INITIALIZE_PROJECTS"]:
        annif.registry.initialize_projects(cxapp.app)
        logger.info("finished initializing projects")

    # register the views via blueprints
    from annif.views import bp

    cxapp.app.register_blueprint(bp)

    # return the Flask app
    return cxapp.app


def _get_config_name(config_name: str | None) -> str:
    if config_name is None:
        config_name = os.environ.get("ANNIF_CONFIG")
    if config_name is None:
        if os.environ.get("FLASK_RUN_FROM_CLI") == "true":  # pragma: no cover
            config_name = "annif.default_config.Config"
        else:
            config_name = "annif.default_config.ProductionConfig"  # pragma: no cover
    return config_name


def _set_tensorflow_loglevel():
    """Set TensorFlow log level based on Annif log level (--verbosity/-v
    option) using an environment variable. INFO messages by TF are shown only on
    DEBUG (or NOTSET) level of Annif."""
    annif_loglevel = logger.getEffectiveLevel()
    tf_loglevel_mapping = {
        0: "0",  # NOTSET
        10: "0",  # DEBUG
        20: "1",  # INFO
        30: "1",  # WARNING
        40: "2",  # ERROR
        50: "3",  # CRITICAL
    }
    tf_loglevel = tf_loglevel_mapping[annif_loglevel]
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", tf_loglevel)
