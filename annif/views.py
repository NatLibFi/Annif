from flask import Blueprint, render_template

bp = Blueprint('app', __name__)


@bp.route('/test')
def test():
    return render_template('test.html')
