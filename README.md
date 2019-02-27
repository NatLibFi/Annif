# Annif

[![DOI](https://zenodo.org/badge/100936800.svg)](https://zenodo.org/badge/latestdoi/100936800)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.org/NatLibFi/Annif.svg?branch=master)](https://travis-ci.org/NatLibFi/Annif?branch=master)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/master/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=master)
[![codebeat badge](https://codebeat.co/badges/e496f151-93db-4f0e-9e30-bc3339e58ca4)](https://codebeat.co/projects/github-com-natlibfi-annif-master)
[![BCH compliance](https://bettercodehub.com/edge/badge/NatLibFi/Annif?branch=master)](https://bettercodehub.com/)
[![LGTM: Python](https://img.shields.io/lgtm/grade/python/g/NatLibFi/Annif.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/NatLibFi/Annif/context:python)

Annif is an automated subject indexing toolkit. It was originally created as
a statistical automated indexing tool that used metadata from the
[Finna.fi](https://finna.fi) discovery interface as a training corpus.

This repo contains a rewritten production version of Annif based on the
[prototype](https://github.com/osma/annif). It is a work in progress, but
already functional for many common tasks.

# Basic install

You will need Python 3.5+ to install Annif.

The recommended way is to install Annif from
[PyPI](https://pypi.org/project/annif/) into a virtual environment.

    python3 -m venv annif-venv
    source annif-venv/bin/activate
    pip install annif

You will also need NLTK data files:

    python -m nltk.downloader punkt

Start up the application:

    annif

See [Getting Started](https://github.com/NatLibFi/Annif/wiki/Getting-started)
in the wiki for more details.

# Development install

A development version of Annif can be installed by cloning the [GitHub
repository](https://github.com/NatLibFi/Annif).
[Pipenv](https://docs.pipenv.org/) is used for managing dependencies for the
development version.

## Installation and setup

Clone the repository.

Switch into the repository directory.
Install pipenv if you don't have it:

    pip install pipenv  # or pip3 install pipenv

Install dependencies and download NLTK data:

    pipenv install  # use --dev if you want to run tests etc.

Enter the virtual environment:

    pipenv shell

You will also need NLTK data files:

    python -m nltk.downloader punkt

Start up the application:

    annif

## Unit tests

Run `pipenv shell` to enter the virtual environment and then run `pytest`.
To have the test suite watch for changes in code and run automatically, use
pytest-watch by running `ptw`.

# License

The code in this repository is licensed under Apache License 2.0, except for the
dependencies included under `annif/static/css` and `annif/static/js`,
which have their own licenses. See the file headers for details.
