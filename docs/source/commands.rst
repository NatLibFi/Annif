############
CLI commands
############

These are the command-line interface commands of Annif, with REST API
equivalents when applicable.

To reference a vocabulary or a project, most of the commands take either a
``VOCAB_ID`` or a ``PROJECT_ID`` parameter, which are alphanumeric strings
``(A-Za-z0-9_-)``. Common options of the commands are ``--projects`` for
setting a (non-default) path to a `project configuration file
<https://github.com/NatLibFi/Annif/wiki/Project-configuration>`_ and
``--verbosity`` for selecting logging level.

Annif supports tab-key completion in bash, zsh and fish shells for commands and options
and project id, vocabulary id and path parameters. See `README.md
<https://github.com/NatLibFi/Annif#shell-completions>`_ for instructions on how to
enable the support.

.. contents::
   :local:
   :backlinks: none

*************************
Vocabulary administration
*************************

.. click:: annif.cli:run_load_vocab
   :prog: annif load-vocab

**REST equivalent**

   N/A

.. click:: annif.cli:run_list_vocabs
   :prog: annif list-vocabs

**REST equivalent**

   N/A

**********************
Project administration
**********************

.. click:: annif.cli:run_list_projects
   :prog: annif list-projects

**REST equivalent**
::

   GET /projects/

.. click:: annif.cli:run_show_project
   :prog: annif show-project

**REST equivalent**
::

   GET /projects/<PROJECT_ID>

.. click:: annif.cli:run_clear_project
   :prog: annif clear-project

**REST equivalent**

   N/A

.. click:: annif.cli:run_upload_projects
   :prog: annif upload-projects

**REST equivalent**

   N/A

****************************
Subject index administration
****************************

.. click:: annif.cli:run_train
   :prog: annif train

**REST equivalent**

   N/A

.. click:: annif.cli:run_learn
   :prog: annif learn

**REST equivalent**
::

   /projects/<PROJECT_ID>/learn

.. click:: annif.cli:run_suggest
   :prog: annif suggest

**REST equivalent**
::

   POST /projects/<PROJECT_ID>/suggest

.. click:: annif.cli:run_eval
   :prog: annif eval

**REST equivalent**

   N/A

.. click:: annif.cli:run_optimize
   :prog: annif optimize

**REST equivalent**

   N/A

.. click:: annif.cli:run_index
   :prog: annif index

**REST equivalent**

   N/A

.. click:: annif.cli:run_hyperopt
   :prog: annif hyperopt

**REST equivalent**

   N/A

.. click:: flask.cli:run_command
   :prog: annif run

**REST equivalent**

   N/A

*****
Other
*****

.. click:: annif.cli:run_completion
   :prog: annif completion

**REST equivalent**

   N/A
