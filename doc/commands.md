# Supported CLI commands in Annif

These are the command line commands of Annif, with REST API equivalents when
applicable.

Most of These methods take a `projectid` parameter. Projects are
identified by alphanumeric strings (`A-Za-z0-9_-`).

## Project administration

### List available projects

    annif list-projects

REST equivalent: 

    GET /projects/

Show a list of currently defined projects. Projects are defined in a
configuration file, normally called `projects.cfg`.

### Show project information

    annif show-project <projectid>

REST equivalent:

    GET /projects/<projectid>

## Subject index administration

### Show all subjects for a project

    annif list-subjects <projectid>

REST equivalent:

    GET /projects/<projectid>/subjects

### Show information about a subject

    annif show-subject <projectid> <subjectid>

REST equivalent:

    GET /projects/<projectid>/subjects/<subjectid>

### Create a new subject, or update an existing one

    annif create-subject <projectid> <subjectid> <subject.txt

REST equivalent:

    PUT /projects/<projectid>/subjects/<subjectid>

This will create a subject from a text file in the corpus format.

If you try to create a subject that already exists, the new subject
definition will overwrite the existing one. However, training and tuning
data associated with the subject will be preserved.

### Load all subjects from a directory

    annif load <projectid> <directory> [--clear=CLEAR]

Parameters:
* `directory`: path to a directory containing text files in the corpus format
* `clear`: Boolean flag that indicates whether the existing subjects should be
  removed first. Defaults to false.

This will load all the subjects from the given directory in a single batch
operation. It is equivalent to executing `create-subject` on each file
separately.

REST equivalent: N/A

### Delete a subject

    annif drop-subject <projectid> <subjectid>

REST equivalent:

    DELETE /projects/<projectid>/subjects/<subjectid>

This will remove all information about a subject, including training and
tuning data.

## Automatic subject indexing

    annif analyze <projectid> [--maxhits=MAX] [--threshold=THRESHOLD] <document.txt

This will read a text document from standard input and suggest subjects for
it.

Parameters:
* `maxhits`: maximum number of subjects to return
* `threshold`: minimum score threshold, expressed as a fraction of highest
  score, below which results will not be returned

REST equivalent:

    POST /projects/<projectid>/analyze
