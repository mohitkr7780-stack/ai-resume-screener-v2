
import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.set_page_config(page_title="AI Resume Analyzer")

st.title("AI Resume Screening System")
st.write("Predict suitable job role from resume skills")

resume_text = st.text_area(
    "Paste Resume Skills Here"
)

if st.button("Predict Role"):

    vector = vectorizer.transform(
        [resume_text]
    )

    prediction = model.predict(vector)

    st.success(
        f"Predicted Role: {prediction[0]}"
    )
