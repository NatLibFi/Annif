"""Unit tests for Annif REST API / Swagger spec"""

from swagger_tester import swagger_test
import time
import threading
import os
import requests


def test_swagger(app):

    # run a Flask/Connexion server in a background thread
    def run_app():
        # We need to set this env var to 'false' because otherwise Flask
        # thinks, due to the CLI tests that could have run before, that
        # it has been started via its  CLI and refuses to run.
        os.environ['FLASK_RUN_FROM_CLI'] = 'false'
        app.run(port=8000)

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    time.sleep(1)
    swagger_test(app_url='http://localhost:8000/v1')

    # test that the service supports CORS
    req = requests.get('http://localhost:8000/v1/projects')
    assert req.headers['access-control-allow-origin'] == '*'
