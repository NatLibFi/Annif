"""common fixtures for use by all test classes"""

import pytest
import annif


@pytest.fixture(scope='session')
def app():
    app = annif.create_app(config_name='config.TestingConfig')
    return app


@pytest.fixture(scope='module')
def app_with_initialize():
    app = annif.create_app(config_name='config.TestingInitializeConfig')
    return app
