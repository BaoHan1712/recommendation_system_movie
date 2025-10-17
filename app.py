import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model_and_data():
    # Load model hybrid
    with open("svd_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load dữ liệu movies từ merged_movie_data.csv
    data = pd.read_csv("merged_movie_data.csv")

    # Loại bỏ trùng lặp
    data = data.drop_duplicates(subset=['movieId', 'title']).reset_index(drop=True)

    # Điền giá trị trống trong 'genres' và chuẩn hóa chuỗi
    data['genres'] = data['genres'].fillna('')
    data['genres'] = data['genres'].str.replace(' ', '')

    indices = pd.Series(data.index.values, index=data['title']).to_dict()
    # Tính cosine similarity dựa trên 'genres'
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return model, data, indices, cosine_sim

model, data, indices, cosine_sim = load_model_and_data()

def hybrid_recommend(user_id, title, top_n=5):
    if title not in indices:
        return f"❌ Phim '{title}' không tồn tại trong dataset."
    
    idx = indices[title]

    # --- Content-based ---
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:50]  # loại bỏ chính phim
    
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = data.iloc[movie_indices][['movieId', 'title', 'genres']].copy()

    # --- Collaborative Filtering ---
    similar_movies['est_rating'] = similar_movies['movieId'].apply(
        lambda x: model.predict(user_id, x).est
    )

    # --- Trả kết quả ---
    recommendations = similar_movies.sort_values('est_rating', ascending=False).head(top_n)
    return recommendations[['title', 'genres', 'est_rating']]

st.set_page_config(
    page_title="🎬 Hybrid Movie Recommender",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.header("Thông tin người dùng")
user_id_input = st.sidebar.number_input("Nhập User ID:", min_value=1, step=1, value=2)
movie_input = st.sidebar.text_input("Nhập tên phim yêu thích:", value="Presto (2008)")
top_n = st.sidebar.slider("Số lượng gợi ý:", 1, 20, 5)

# Main
st.markdown("""
<style>
.main > div.block-container {padding-top: 2rem;}
h1 {color: #1F77B4;}
.stButton>button {background-color: #FFDD00; color: #000; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Hybrid Movie Recommender")
st.markdown("Nhập `User ID` và `Tên phim yêu thích` để nhận gợi ý phim phù hợp.")

if st.button("✅ Gợi ý phim"):
    with st.spinner("Đang phân tích và gợi ý..."):
        result = hybrid_recommend(user_id_input, movie_input, top_n)
        if isinstance(result, str):  # lỗi
            st.error(result)
        else:
            st.success(f"Gợi ý cho User {user_id_input} dựa trên phim '{movie_input}':")
            st.dataframe(result.style.format({"est_rating": "{:.2f}"}))
