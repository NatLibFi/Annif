#!/usr/bin/env python3

import pytest
import sys
import os

# Import annif-module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import annif

def test_start():
    assert annif.start() == "Started application"
