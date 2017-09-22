# Supported CLI commands in Annif

with REST API equivalents, when applicable

Most of These methods take a `projectid` parameter. Projects are
identified by alphanumeric strings (`A-Za-z0-9_-`).

## Project administration

### List available projects

    annif list-projects

REST equivalent: 

    GET /projects/

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

## Index administration



## Automatic subject indexing

