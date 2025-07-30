import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:
    def __init__(self, job_data_path):
        self.job_data = pd.read_csv(job_data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_descriptions = self.vectorizer.fit_transform(self.job_data['Description'])

    def recommend_jobs(self, user_input, top_n=3):
        user_vec = self.vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_vec, self.job_descriptions).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommended_jobs = self.job_data.iloc[top_indices]
        return recommended_jobs