from flask import render_template,url_for
from webapp import app


@app.route("/")
def authentication_page():
    return render_template('authentication_page.html')
