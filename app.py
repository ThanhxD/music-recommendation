from flask import Flask, render_template
from main import recommend

app = Flask(__name__)

@app.route("/")
def index():
    results = recommend()
    return render_template("index.html", results = results)

if __name__ == '__main__':
    app.run()