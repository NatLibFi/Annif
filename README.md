# Annif

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.org/NatLibFi/Annif.svg?branch=master)](https://travis-ci.org/NatLibFi/Annif?branch=master)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/master/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Coverage Status](https://coveralls.io/repos/github/NatLibFi/Annif/badge.svg)](https://coveralls.io/github/NatLibFi/Annif)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=master)
[![codebeat badge](https://codebeat.co/badges/e496f151-93db-4f0e-9e30-bc3339e58ca4)](https://codebeat.co/projects/github-com-natlibfi-annif-master)

Annif is an automated subject indexing toolkit. It was originally created as
a statistical automated indexing tool that used metadata from the
[Finna.fi](https://finna.fi) discovery interface as a training corpus.

This repo contains a rewritten production version of Annif based on the
[prototype](https://github.com/osma/annif). It is a work in progress.

### Dependencies

Python 3.5+. Using `venv` for setting up an isolated environment is
encouraged. Other dependencies are listed in `requirements.txt`.

### Installation and setup

Clone the repository.

Switch into the repository directory.
Create a virtual environment by running:

    python3 -m venv annif_env
    source annif_env/bin/activate

Install dependencies:

    pip install -r requirements.txt

Set the `FLASK_APP` enviroment variable:

    export FLASK_APP=annif/__init__.py

Set a shell alias for the application: 

    alias annif=flask

Start up the application

    annif run

Run tests with the command `pytest`. To have the test watch for changes in code
and run automatically, use pytest-watch by running `ptw`.
