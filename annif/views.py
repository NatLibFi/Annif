from flask import Blueprint, render_template

bp = Blueprint('app', __name__)


@bp.route('/')
def root():
    return render_template('root.html')
