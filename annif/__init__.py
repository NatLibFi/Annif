#!/usr/bin/env python3

import connexion
import logging

# 'cxapp' here is the Connexion application that has a normal Flask app as a
# property (cxapp.app)

cxapp = connexion.App(__name__, specification_dir='../swagger/')

cxapp.app.config.from_object('config.Config')

# make the Flask logger easily available to the rest of the app
logger = cxapp.app.logger

# initialize CLI commands
import annif.cli

cxapp.add_api('annif.yaml')

application = cxapp.app
