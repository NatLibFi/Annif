#!/usr/bin/env python3

import pytest
import sys
import os

# Import annif-module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from annif import app

# A dummy test for setup purposes
def test_start():
    assert app.start() == "Started application"
