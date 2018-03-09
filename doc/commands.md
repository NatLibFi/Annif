# Supported CLI commands in Annif

These are the command line commands of Annif, with REST API equivalents when
applicable.

Most of These methods take a `projectid` parameter. Projects are
identified by alphanumeric strings (`A-Za-z0-9_-`).

## Initialization

### Creating the index structure

    annif init

This will initialize the Elasticsearch index needed for storing information
about projects.

## Project administration

### List available projects

    annif list-projects

REST equivalent: 

    GET /projects/

Show a list of currently defined projects.

### Show project information

    annif show-project <projectid>

REST equivalent:

    GET /projects/<projectid>

### Create a new project

    annif create-project <projectid> --language <lang> --analyzer <analyzer>

Parameters:
* `lang`: language of text, expressed as ISO 639-1 code, e.g. `en`
* `analyzer`: Elasticsearch analyzer to use, e.g. `english`

REST equivalent: 

    PUT /projects/<projectid>

If you try to create a project that already exists, the settings will be
compared with the existing project. Some settings (language and analyzer)
cannot be changed after the project has been created. If the settings are
compatible (i.e. the immutable settings have not changed), then the new
settings will be used. If the settings are incompatible, you will get an
error instead.

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
