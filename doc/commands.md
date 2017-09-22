# Supported CLI commands in Annif

with REST API equivalents, when applicable

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

### Create a new project

    annif create-project <projectid> --language <lang> --analyzer <analyzer>

Parameters:
* `lang`: language of text, expressed as ISO 639-1 code, e.g. `en`
* `analyzer`: Elasticsearch analyzer to use, e.g. `english`

REST equivalent: 

    PUT /projects/<projectid>

### Delete a project

    annif drop-project <projectid>

REST equivalent: 

    DELETE /projects/<projectid>

## Subject index administration

### Show all subjects for a project

    annif list-subjects <projectid>

REST equivalent:

    GET /projects/<projectid>/subjects

### Show information about a subject

    annif show-subject <projectid> <subjectid>

REST equivalent:

    GET /projects/<projectid>/subjects/<subjectid>

### Create a new subject

    annif create-subject <projectid> <subjectid>

REST equivalent:

    PUT /projects/<projectid>/subjects/<subjectid>

This command can also be used to recreate an existing subject. All extra
information about the subject, such as boost values and learned word
associations, will be discarded.

### Update a subject

    annif update-subject <projectid> <subjectid>

REST equivalent:

    POST /projects/<projectid>/subjects/<subjectid>



### Delete a subject

    annif drop-subject <projectid> <subjectid>

REST equivalent:

    DELETE /projects/<projectid>/subjects/<subjectid>



## Automatic subject indexing

