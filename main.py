from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Cho phép FE gọi API từ domain khác

# --- Load model và dữ liệu ---
def load_model_and_data():
    with open("svd_model.pkl", "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv("merged_movie_data.csv")
    data = data.drop_duplicates(subset=['movieId', 'title']).reset_index(drop=True)
    data['genres'] = data['genres'].fillna('')
    data['genres'] = data['genres'].str.replace(' ', '')

    indices = pd.Series(data.index.values, index=data['title']).to_dict()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return model, data, indices, cosine_sim

model, data, indices, cosine_sim = load_model_and_data()

# --- Hàm gợi ý ---
def hybrid_recommend(user_id, title, top_n=5):
    if title not in indices:
        return None, f"❌ Phim '{title}' không tồn tại trong dataset."
    
    idx = indices[title]

    # Content-based
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:50]

    movie_indices = [i[0] for i in sim_scores]
    similar_movies = data.iloc[movie_indices][['movieId', 'title', 'genres']].copy()

    # Collaborative Filtering
    similar_movies['est_rating'] = similar_movies['movieId'].apply(
        lambda x: model.predict(user_id, x).est
    )

    recommendations = similar_movies.sort_values('est_rating', ascending=False).head(top_n)
    return recommendations[['title', 'genres', 'est_rating']], None


# --- API Endpoint: trả JSON ---
@app.route("/api/recommendations", methods=["POST"])
def api_recommend():
    try:
        data_req = request.get_json()

        user_id = int(data_req.get("userId", 1))
        title = data_req.get("movieTitle", "")
        top_n = int(data_req.get("topN", 5))

        if not title:
            return jsonify({"error": "Thiếu tên phim"}), 400

        result, error = hybrid_recommend(user_id, title, top_n)
        if error:
            return jsonify({"error": error}), 404

        # Convert to JSON
        result_json = result.round(2).to_dict(orient="records")
        return jsonify({
            "userId": user_id,
            "movieTitle": title,
            "recommendations": result_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    # Trả về file templates/index.html
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
