# Annif architecture

## Basic concepts

An installation of Annif may contain multiple independent **projects**, each
of which specifies a set of settings (e.g. language and indexing
vocabulary). Projects are limited to a single language; multilingual
indexing can be performed by defining multiple projects, one per language.

Each project defines a (typically large) number of **subjects**, which
reflect concepts from an indexing vocabulary. Subjects are typically created
from a corpus extracted from existing metadata records and/or indexed
documents.

Further, there can be several independent **backends** that provide analysis
functionality. Backends can be integrated into Annif itself, or external
services queried via APIs. A project can make use of several backends and
combine their analysis results.

## Software architecture

Annif has a core application (using Flask), which provides both a REST API
(when run as a web application) and a command line interface. 

In addition, there are separate, independent command line utilities for
corpus management, quality evaluation and other auxiliary functions.
