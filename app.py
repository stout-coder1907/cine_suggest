import ast
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')

# ----------------------------
# Configuration & Assets
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_CSV = PROJECT_ROOT / 'tmdb_5000_movies.csv'
CREDITS_CSV = PROJECT_ROOT / 'tmdb_5000_credits.csv'
TMDB_API_KEY = '254ddecd1de46492bed4a759f654969c'

st.set_page_config(
    page_title='CineSuggest',
    page_icon='🍿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ----------------------------
# Custom CSS (Visual Excellence)
# ----------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #E0E0E0;
        }

        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #FFFFFF !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }

        /* Glassmorphism Containers */
        .glass-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
        }

        .hero-section {
            text-align: center;
            padding: 40px 20px;
            background: radial-gradient(circle at center, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        }

        .hero-title {
            font-size: 4rem;
            background: linear-gradient(to right, #818cf8, #c4b5fd, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: #94a3b8;
            max-width: 600px;
            margin: 0 auto;
        }

        /* Controls styling */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: rgba(30, 41, 59, 0.8);
            border-color: rgba(99, 102, 241, 0.3);
            color: white;
            border-radius: 12px;
        }

        /* Button Styling */
        div.stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.6);
        }

        /* Movie Card */
        .movie-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .movie-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            border-color: rgba(99, 102, 241, 0.3);
        }
        .poster-container {
            position: relative;
            width: 100%;
            padding-top: 150%; /* Aspect ratio for poster */
            background: #0f1116;
        }
        .poster-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .no-poster {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1e293b;
            color: #64748b;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .card-content {
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #f1f5f9;
            line-height: 1.3;
        }
        .card-meta {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rating-badge {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
        }
        .genre-tag {
            font-size: 0.75rem;
            color: #cbd5e1;
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 4px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Core Logic (Cached)
# ----------------------------

def _convert_list_of_names(obj_str: str) -> List[str]:
    try:
        return [item['name'] for item in ast.literal_eval(obj_str)]
    except Exception:
        return []

def _convert_top_cast(obj_str: str, top_n: int = 3) -> List[str]:
    try:
        return [item['name'] for item in ast.literal_eval(obj_str)[:top_n]]
    except Exception:
        return []

def _fetch_director(obj_str: str) -> List[str]:
    try:
        for item in ast.literal_eval(obj_str):
            if item.get('job') == 'Director':
                return [item.get('name')]
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False)
def load_data_and_models():
    """Loads data, builds training frame and vectorizer matrix."""
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    
    # Merge
    raw_df = movies.merge(credits, on='title')
    
    # Process for ML
    df = raw_df[['movie_id', 'keywords', 'title', 'genres', 'overview', 'cast', 'crew']]
    df.dropna(inplace=True)
    
    df['genres_list'] = df['genres'].apply(_convert_list_of_names)
    df['keywords'] = df['keywords'].apply(_convert_list_of_names)
    df['cast'] = df['cast'].apply(_convert_top_cast)
    df['director'] = df['crew'].apply(_fetch_director)
    
    df['overview'] = df['overview'].apply(lambda x: x.split())
    
    # Tags creation for Vectorizer
    # Remove spaces for tags
    tags_df = df.copy()
    tags_df['genres'] = tags_df['genres_list'].apply(lambda x: [i.replace(' ', '') for i in x])
    tags_df['keywords'] = tags_df['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
    tags_df['cast'] = tags_df['cast'].apply(lambda x: [i.replace(' ', '') for i in x])
    tags_df['director'] = tags_df['director'].apply(lambda x: [i.replace(' ', '') for i in x])
    
    tags_df['tags'] = tags_df['overview'] + tags_df['genres'] + tags_df['keywords'] + tags_df['cast'] + tags_df['director']
    tags_df['tags'] = tags_df['tags'].apply(lambda x: ' '.join(x).lower())
    
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    ps = PorterStemmer()
    tags_stemmed = tags_df['tags'].apply(lambda x: ' '.join([ps.stem(i) for i in x.split()]))
    vectors = cv.fit_transform(tags_stemmed).toarray()
    similarity = cosine_similarity(vectors)

    # Prepare detailed metadata frame for UI
    meta_df = raw_df.copy()
    meta_df['release_year'] = pd.to_datetime(meta_df['release_date'], errors='coerce').dt.year
    meta_df['genres_list'] = meta_df['genres'].apply(_convert_list_of_names)
    
    return tags_df, similarity, meta_df

# Session with retry strategy
def get_session():
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int):
    if not TMDB_API_KEY: 
        return None
    try:
        session = get_session()
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}'
        # Add User-Agent to avoid some blocking
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        resp = session.get(url, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'poster_path' in data and data['poster_path']:
                return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except Exception as e:
        # In production, we might log this: print(f"Error fetching poster for {movie_id}: {e}")
        pass
    return None

def get_recommendations_filtered(
    title: str, 
    similarity_matrix, 
    df: pd.DataFrame, 
    meta_df: pd.DataFrame,
    filters: Dict
) -> List[Dict]:
    
    if title not in df['title'].values:
        return []
        
    movie_index = df[df['title'] == title].index[0]
    distances = similarity_matrix[movie_index]
    
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    movie_list = movie_list[1:]
    
    recommendations = []
    
    count = 0
    target_count = filters.get('top_n', 10)
    
    # Pre-fetch session for batch (conceptually) - though fetch_poster is cached per ID.
    
    for i in movie_list:
        idx = i[0]
        movie_title = df.iloc[idx].title
        
        meta_row = meta_df.loc[meta_df['title'] == movie_title]
        if meta_row.empty: 
            continue
        meta_row = meta_row.iloc[0]
        
        # Apply Filters
        min_rating = filters.get('min_rating', 0)
        if meta_row['vote_average'] < min_rating:
            continue
            
        year_range = filters.get('year_range')
        r_year = meta_row['release_year']
        if year_range and (pd.isna(r_year) or not (year_range[0] <= r_year <= year_range[1])):
            continue

        selected_genres = filters.get('genres')
        if selected_genres:
            movie_genres = set(meta_row['genres_list'])
            if not movie_genres.intersection(set(selected_genres)):
                continue
        
        poster = fetch_poster(meta_row['movie_id'])
        
        recommendations.append({
            'title': movie_title,
            'year': int(r_year) if pd.notna(r_year) else 'N/A',
            'rating': round(meta_row['vote_average'], 1),
            'genres': meta_row['genres_list'][:2],
            'poster': poster
        })
        
        count += 1
        if count >= target_count:
            break
            
    return recommendations


# ----------------------------
# UI Layout
# ----------------------------

def main():
    inject_custom_css()
    
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">CineSuggest</div>
            <div class="hero-subtitle">
                Discover your next visual masterpiece with our AI-driven recommendation engine.
                Curated just for you.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Loading Data
    with st.spinner('Initializing AI Model...'):
        df, similarity, meta_df = load_data_and_models()

    # Sidebar: Filters
    with st.sidebar:
        st.header("⚙️ Preferences")
        
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.5)
        
        min_year = int(meta_df['release_year'].min())
        max_year = int(meta_df['release_year'].max())
        
        year_range = st.slider(
            "Release Year", 
            min_year, max_year, 
            (1990, max_year)
        )
        
        all_genres = sorted(list(set([g for sublist in meta_df['genres_list'] for g in sublist])))
        selected_genres = st.multiselect("Filter by Genre", all_genres)
        
        top_n = st.number_input("Number of Recommendations", min_value=1, max_value=30, value=8)

    # Main Control Area
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        movie_list = df['title'].values
        selected_movie = st.selectbox(
            "Select a movie you love",
            movie_list,
            index=list(movie_list).index("Inception") if "Inception" in movie_list else 0
        )
    
    with col2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True) # Spacer
        if st.button("✨ Recommend"):
            st.session_state['show_results'] = True
    st.markdown('</div>', unsafe_allow_html=True)

    # Results Display
    if st.session_state.get('show_results'):
        filters = {
            'min_rating': min_rating,
            'year_range': year_range,
            'genres': selected_genres,
            'top_n': top_n
        }
        
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            recs = get_recommendations_filtered(selected_movie, similarity, df, meta_df, filters)
        
        if not recs:
            st.warning("No movies found matching your filters. Try adjusting the rating or year range!")
        else:
            st.markdown(f"### Top Picks for '{selected_movie}'")
            
            # Grid Layout
            cols = st.columns(4)
            for idx, movie in enumerate(recs):
                col = cols[idx % 4]
                with col:
                    if movie['poster']:
                        poster_html = f'<img src="{movie["poster"]}" class="poster-img"/>'
                    else:
                        poster_html = '<div class="no-poster"><span>No Poster</span></div>'
                    
                    genres_html = "".join([f'<span class="genre-tag">{g}</span>' for g in movie['genres']])
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="poster-container">
                            {poster_html}
                        </div>
                        <div class="card-content">
                            <div class="card-title">{movie['title']}</div>
                            <div class="card-meta">
                                <span>{movie['year']}</span>
                                <span class="rating-badge">★ {movie['rating']}</span>
                            </div>
                            <div style="margin-top: 8px;">
                                {genres_html}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()



