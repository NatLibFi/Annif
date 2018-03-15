"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import statistics
import sys
import click
import annif
import annif.corpus
import annif.eval
import annif.project


@annif.cxapp.app.cli.command('list-projects')
def run_list_projects():
    """
    List available projects.

    Usage: annif list-projects
    """

    template = "{0: <15}{1: <15}{2: <15}\n"

    formatted = template.format("Project ID", "Language", "Analyzer")
    formatted += str("-" * len(formatted) + "\n")

    for proj in annif.project.get_projects().values():
        formatted += template.format(proj.project_id, proj.language,
                                     proj.analyzer)

    print(formatted)


@annif.cxapp.app.cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Show project information.

    Usage: annif show-project <project_id>

    Outputs a human-readable string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    Analyzer       finglish
    """

    try:
        proj = annif.project.get_project(project_id)
    except ValueError:
        print("No projects found with id \'{0}\'.".format(project_id))
        sys.exit(1)

    formatted = ""
    template = "{0:<15}{1}\n"

    formatted = template.format('Project ID:', proj.project_id)
    formatted += template.format('Language:', proj.language)
    formatted += template.format('Analyzer', proj.analyzer)
    print(formatted)


@annif.cxapp.app.cli.command('load')
@click.argument('project_id')
@click.argument('directory')
@click.option('--clear', default=False)
def run_load(project_id, directory, clear):
    print("TODO")


@annif.cxapp.app.cli.command('list-subjects')
@click.argument('project_id')
def run_list_subjects():
    print("TODO")


@annif.cxapp.app.cli.command('show-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_show_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('create-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_create_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('drop-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_drop_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('analyze')
@click.argument('project_id')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_analyze(project_id, limit, threshold):
    """"
    Analyze a document.

    USAGE: annif analyze <project_id> [--limit=N] [--threshold=N] <document.txt
    """
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        print("No projects found with id \'{0}\'.".format(project_id))
        sys.exit(1)

    text = sys.stdin.read()

    hits = project.analyze(text, limit, threshold)
    for hit in hits:
        print("{}\t<{}>\t{}".format(hit.score, hit.uri, hit.label))


@annif.cxapp.app.cli.command('eval')
@click.argument('project_id')
@click.argument('subject_file')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_eval(project_id, subject_file, limit, threshold):
    """"
    Evaluate the analysis result for a document against a gold standard given in a subject file.

    USAGE: annif eval <project_id> <subject_file> [--limit=N] [--threshold=N] <document.txt
    """
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        print("No projects found with id \'{0}\'.".format(project_id))
        sys.exit(1)

    text = sys.stdin.read()
    hits = project.analyze(text, limit, threshold)
    with open(subject_file) as subjfile:
        gold_subjects = annif.corpus.SubjectSet(subjfile.read())

    if gold_subjects.has_uris():
        selected = set([hit.uri for hit in hits])
        gold_set = gold_subjects.subject_uris
    else:
        selected = set([hit.label for hit in hits])
        gold_set = gold_subjects.subject_labels

    template = "{0:<10}\t{1}"

    precision = annif.eval.precision(selected, gold_set)
    print(template.format("Precision:", precision))

    recall = annif.eval.recall(selected, gold_set)
    print(template.format("Recall:", recall))

    f_measure = annif.eval.f_measure(selected, gold_set)
    print(template.format("F-measure:", f_measure))


@annif.cxapp.app.cli.command('evaldir')
@click.argument('project_id')
@click.argument('directory')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_evaldir(project_id, directory, limit, threshold):
    """"
    Evaluate the analysis results for a directory with documents against a gold standard given in subject files.

    USAGE: annif evaldir <project_id> <directory> [--limit=N] [--threshold=N]
    """
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        print("No projects found with id \'{0}\'.".format(project_id))
        sys.exit(1)

    precisions = []
    recalls = []
    f_measures = []

    for docfilename, keyfilename in annif.corpus.DocumentDirectory(directory):
        print("evaluating", docfilename, keyfilename)
        if keyfilename is None:
            continue
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = project.analyze(text, limit, threshold)
        with open(keyfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())

        if gold_subjects.has_uris():
            selected = set([hit.uri for hit in hits])
            gold_set = gold_subjects.subject_uris
        else:
            selected = set([hit.label for hit in hits])
            gold_set = gold_subjects.subject_labels

        precisions.append(annif.eval.precision(selected, gold_set))
        recalls.append(annif.eval.recall(selected, gold_set))
        f_measures.append(annif.eval.f_measure(selected, gold_set))

    template = "{0:<10}\t{1}"
    print(template.format("Precision:", statistics.mean(precisions)))
    print(template.format("Recall:", statistics.mean(recalls)))
    print(template.format("F-measure:", statistics.mean(f_measures)))
