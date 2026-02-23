from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")

# Clean data
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# Convert text → numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def recommend(title):
    if title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    recommended = [movies.iloc[i[0]].title for i in scores]
    return recommended


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)

    movie_list = movies['title'].head(500).tolist()

    return render_template("index.html",
                           movie_list=movie_list,
                           recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
