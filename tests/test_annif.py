#!/usr/bin/env python3

import pytest
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from annif import annif


# A dummy test for setup purposes
def test_start():
    assert annif.start() == "Started application"


#def test_listprojects():
#    assert annif.listprojects() != "jou"
