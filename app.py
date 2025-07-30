import streamlit as st
from recommender import JobRecommender
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("AI Job Recommendation System")

job_rec = JobRecommender("job_data.csv")

user_input = st.text_area("Enter your resume text or job interest:")

if st.button("Recommend Jobs"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = job_rec.recommend_jobs(user_input, top_n=3)
        st.subheader("Top Job Matches:")
        for _, row in results.iterrows():
            st.write(f"**{row['Title']}**")
            st.write(f"{row['Description']}")
            st.markdown("---")
