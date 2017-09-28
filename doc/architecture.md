# Annif architecture

## Basic concepts

An installation of Annif may contain multiple independent **projects**, each
of which specifies a set of settings (e.g. language and indexing
vocabulary). Projects are limited to a single language; multilingual
indexing can be performed by defining multiple projects, one per language.

Each project defines a (typically large) number of **subjects**, which
reflect concepts from an indexing vocabulary. Annif maintains associations
between subjects and natural language words that appear in documents.
Subjects are typically created from a corpus extracted from existing
metadata records and/or indexed documents.

## Software architecture

An installation of Annif requires an Elasticsearch index, which is used to
store all volatile data.

Annif has a core application (using Flask), which provides both a REST API
(when run as a web application) and a command line interface. All accesses
to the Elasticsearch index are performed through this application.

In addition, there are separate, independent command line utilities for
corpus management, quality evaluation and other auxiliary functions that
don't need to access the Elasticsearch index directly.
