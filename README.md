# Annif

[![Build Status](https://travis-ci.org/NatLibFi/Annif.svg?branch=master)](https://travis-ci.org/NatLibFi/Annif?branch=master)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/master/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Coverage Status](https://coveralls.io/repos/github/NatLibFi/Annif/badge.svg?branch=master)](https://coveralls.io/github/NatLibFi/Annif?branch=master)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=master)
[![codebeat badge](https://codebeat.co/badges/e496f151-93db-4f0e-9e30-bc3339e58ca4)](https://codebeat.co/projects/github-com-natlibfi-annif-master)

*ANNotation Infrastructure using Finna: an automatic subject indexing tool using Finna as corpus*.

This repo contains a rewritten production version of Annif based on the [prototype](https://github.com/osma/annif).
A work in progress.

### Dependencies

Python 3.5+ and a locally installed [Elasticsearch](https://www.elastic.co/products/elasticsearch) instance. Using virtualenv for setting up an isolated environment is encouraged. Other dependencies are listed in requirements.txt.

### Installation and setup

Clone the repository.

Switch into the repository directory and create a virtualenv environment by running `virtualenv -p python3 annif_env && source annif_env/bin/activate`.

Install dependencies by running `pip install -r requirements.txt`.

Set the `FLASK_APP` enviroment variable: `export FLASK_APP=annif/annif.py`.

Set a shell alias for the applicaton: `alias annif=flask`.

Run with the command `annif run`.

Run tests with the command `pytest`.
