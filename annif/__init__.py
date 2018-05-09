#!/usr/bin/env python3

import os
import connexion
import logging

logger = logging.getLogger('annif')

import annif.backend  # noqa


def create_app(script_info=None, config_name=None):
    # 'cxapp' here is the Connexion application that has a normal Flask app
    # as a property (cxapp.app)

    cxapp = connexion.App(__name__, specification_dir='../swagger/')
    if config_name is None:
        config_name = os.environ.get('ANNIF_CONFIG')
    if config_name is None:
        if os.environ.get('FLASK_RUN_FROM_CLI') == 'true':
            config_name = 'config.Config'
        else:
            config_name = 'config.ProductionConfig'
    cxapp.app.config.from_object(config_name)
    cxapp.app.config.from_envvar('ANNIF_SETTINGS', silent=True)

    cxapp.add_api('annif.yaml')

    annif.project.init_projects(cxapp.app)

    # register the views via blueprints
    from annif.views import bp
    cxapp.app.register_blueprint(bp)

    # return the Flask app
    return cxapp.app
