#!/usr/bin/env python3

import connexion

# 'cxapp' here is the Connexion application that has a normal Flask app as a
# property (cxapp.app)

cxapp = connexion.App(__name__, specification_dir='../swagger/')

cxapp.app.config.from_object('config.Config')

# initialize CLI commands
import annif.cli

cxapp.add_api('annif.yaml')

application = cxapp.app
