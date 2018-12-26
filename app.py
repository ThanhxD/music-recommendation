from flask import Flask, render_template, request
from flask_api import FlaskAPI
from main import recommend
from database import listSongs

app = FlaskAPI(__name__)

@app.route("/")
def index():
    artist = request.args.get('artist')
    results = listSongs()
    return render_template("index.html", results = results )

@app.route("/recommender", methods=['GET'])
def suggest():
    artist = request.args.get('artist')
    results = recommend(artist)
    # return render_template("index.html", results = results)
    return results


if __name__ == '__main__':
    app.run(debug=True)