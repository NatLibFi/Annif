############
CLI commands
############

These are the command line interface commands of Annif, with REST API equivalents when applicable.

Most of these methods take a ``PROJECT_ID`` parameter. Projects are identified by alphanumeric strings ``(A-Za-z0-9_-)``.

.. contents::
   :local:
   :backlinks: none

**********************
Project administration
**********************

.. click:: annif.cli:run_loadvoc
   :prog: annif loadvoc

.. click:: annif.cli:run_list_projects
   :prog: annif list-projects

.. click:: annif.cli:run_show_project
   :prog: annif show-project

.. click:: annif.cli:run_clear_project
   :prog: annif clear-project

****************************
Subject index administration
****************************

.. click:: annif.cli:run_train
   :prog: annif train

.. click:: annif.cli:run_learn
   :prog: annif learn

.. click:: annif.cli:run_suggest
   :prog: annif suggest

.. click:: annif.cli:run_eval
   :prog: annif eval

.. click:: annif.cli:run_optimize
   :prog: annif optimize

.. click:: annif.cli:run_index
   :prog: annif index

.. click:: flask.cli:run_command
   :prog: annif run
