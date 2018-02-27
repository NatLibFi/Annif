import click
import annif
import annif.operations

##############################################################################
# COMMAND-LINE INTERFACE
# Here are the definitions for command-line (Click) commands for invoking
# the above functions and printing the results to console.
##############################################################################

@annif.cxapp.app.cli.command('init')
def run_init():
    print(init())


@annif.cxapp.app.cli.command('list-projects')
def run_list_projects():
    template = "{0: <15}{1: <15}{2: <15}\n"

    formatted = template.format("Project ID", "Language", "Analyzer")
    formatted += str("-" * len(formatted) + "\n")

    for proj in annif.operations.list_projects():
        formatted += template.format(proj['name'], proj['language'],
                                     proj['analyzer'])

    print(formatted)


@annif.cxapp.app.cli.command('create-project')
@click.argument('project_id')
@click.option('--language')
@click.option('--analyzer')
def run_create_project(project_id, language, analyzer):
    print(annif.operations.create_project(project_id, language, analyzer))


@annif.cxapp.app.cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Takes a dict containing project information as returned by ES client
    and returns a human-readable string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    Analyzer       finglish

    """

    res = annif.operations.show_project(project_id)
    if type(res) is not str:
        formatted = ""
        template = "{0:<15}{1}\n"

        content = res['hits'][0]
        formatted = template.format('Project ID:', content['_source']['name'])
        formatted += template.format('Language:', content['_source']['language'])
        formatted += template.format('Analyzer', content['_source']['analyzer'])
        print(formatted)
    else:
        print(res)


@annif.cxapp.app.cli.command('drop-project')
@click.argument('project_id')
def run_drop_project(project_id):
    print(annif.operations.drop_project(project_id))


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
@click.option('--maxhits', default=20)
@click.option('--threshold', default=0.9)  # TODO: Check this.
def run_analyze(project_id, maxhits, threshold):
    print(annif.operations.analyze(project_id, maxhits, threshold))

##############################################################################
