.. _my-reference-label:
###############################
Supported CLI commands in Annif
###############################

These are the command line commands of Annif, with REST API equivalents when applicable.

Most of these methods take a ``PROJECT_ID`` parameter. Projects are identified by alphanumeric strings ``(A-Za-z0-9_-)``.

.. contents::
   :local:
   :backlinks: none

**********************
Project administration
**********************

.. click:: annif.cli:run_loadvoc
   :prog: annif loadvoc

This will load the vocabulary to be used in subject indexing. Note that although ``PROJECT_ID`` is a parameter of the command, the vocabulary is shared by all the projects with the same vocab identifier in the project configuration, and the vocabulary only needs to be loaded for one of those projects.

If a vocabulary has already been loaded, reinvoking ``loadvoc`` with a new subject file will update the Annif's internal vocabulary: label names are updated and any subject not appearing in the new subject file is removed. Note that new subjects will not be suggested before the project is retrained with the updated vocabulary. The update behavior can be overridden with the ``--force`` option.

REST equivalent: N/A

.. click:: annif.cli:run_list_projects
   :prog: annif list-projects

Show a list of currently defined projects. Projects are defined in a configuration file, normally called projects.cfg. See Project configuration for details.

REST equivalent::

  GET /projects/

.. click:: annif.cli:run_show_project
   :prog: annif show-project

REST equivalent:::

   GET /projects/<PROJECT_ID>


.. click:: annif.cli:run_clear_project
   :prog: annif clear-project

REST equivalent: N/A


****************************
Subject index administration
****************************

.. click:: annif.cli:run_train
   :prog: annif train

This will train the project using all the documents from the given directory or TSV file in a single batch operation.

REST equivalent: N/A

.. click:: annif.cli:run_learn
   :prog: annif learn

This will continue training an already trained project using all the documents from the given directory or TSV file in a single batch operation. Not supported by all backends.

REST equivalent::

   POST /projects/<PROJECT_ID>/learn

.. click:: annif.cli:run_suggest
   :prog: annif suggest

This will read a text document from standard input and suggest subjects for it.

REST equivalent::

  POST /projects/<PROJECT_ID>/suggest

.. click:: annif.cli:run_eval
   :prog: annif eval

You need to supply the documents in one of the supported Document corpus formats, i.e. either as a directory or as a TSV file. It is possible to give multiple corpora (even mixing corpus formats), in which case they will all be processed in the same run.

The output is a list of statistical measures.

REST equivalent: N/A

.. click:: annif.cli:run_optimize
   :prog: annif optimize

As with eval, you need to supply the documents in one of the supported Document corpus formats. This command will read each document, assign subjects to it using different limit and threshold values, and compare the results with the gold standard subjects.

The output is a list of parameter combinations and their scores. From the output, you can determine the optimum limit and threshold parameters depending on which measure you want to target.

REST equivalent: N/A

.. click:: annif.cli:run_index
   :prog: annif index

.. click:: flask.cli:run_command
   :prog: annif run

