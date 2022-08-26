############
CLI commands
############

These are the command-line interface commands of Annif, with REST API
equivalents when applicable.

To reference a project most of the commands take a ``PROJECT_ID`` parameter,
which is an alphanumeric string ``(A-Za-z0-9_-)``. Common options of the
commands are ``--projects`` for setting a (non-default) path to a `project
configuration file
<https://github.com/NatLibFi/Annif/wiki/Project-configuration>`_ and
``--verbosity`` for selecting logging level.

.. contents::
   :local:
   :backlinks: none

**********************
Project administration
**********************

.. click:: annif.cli:run_loadvoc
   :prog: annif loadvoc

**REST equivalent**

   N/A

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
