# pip install stramlit
# pip install openai
# pip install pinecone-client

import os
from pinecone import Pinecone
import streamlit as st
from openai import OpenAI

# AI Functions
def generate_blog(key, topic, additional_text):
    prompt = f"""
        You are a copy writer with years of experience writing impactful blog that converge and help elevate brands.
        Your task is to write a blog on any topic system provides you with. Make sure to write in a format that works for Medium.
        Each blog should be separated into segments that have titles and subtitles. Each paragraph should be three sentences long.

        Topic: {topic}
        Additiona pointers: {additional_text}
    """
    client = OpenAI(api_key=key)

    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=1,
        max_tokens=300
    )

    return response

def generate_image(key, prompt, number_of_images):
    client = OpenAI(api_key=key)
    response = client.images.generate(
        prompt=prompt,
        n=number_of_images,
        size="256x256"
    )

    return response
# END AI Functions


st.set_page_config(page_title="OpenAI API Web App", page_icon="🧊", layout="centered", initial_sidebar_state="auto")

st.title("OpenAI API Web App")
st.sidebar.title("OpenAI API Apps")

ai_app = st.sidebar.radio("Choose an AI App", ["Blog Generator", "Image Generator", "Movie Recommender"])

if ai_app == "Blog Generator":
    st.header("Blog Generator")
    st.write("Input a topic to generate a blog about it using the OpenAI API")

    key = st.text_input("Inform your OpenAI API key")
    topic = st.text_input("Enter a topic")
    additional_text = st.text_area("Additional text", "Write something here")

    if st.button("Generate Blog") and topic != "":
        with st.spinner("Loading..."):
            response = generate_blog(key, topic, additional_text)
            st.text_area("Generated blog", value=response.choices[0].text, height=700)

elif ai_app == "Image Generator":
    st.header("Image Generator")
    st.write("Input a prompt to generate an image using the OpenAI API and DALL-E")

    key = st.text_input("Inform your OpenAI API key")
    prompt = st.text_input("Enter a prompt")
    number_of_images = st.slider("Number of images", 1, 5, 1)

    if st.button("Generate Image") and prompt != "":
        with st.spinner("Loading..."):
            response = generate_image(key, prompt, number_of_images)

            for image in response.data:
                st.image(image.url)

elif ai_app == "Movie Recommender":
    st.header("Movie Recommender")
    st.write("Describe a movie that you would like to see")

    key = st.text_input("Inform your OpenAI API key")
    pinecone_key = st.text_input("Inform your Pinecone API key")
    index = st.text_input("Inform your Pinecone Index name")
    movie_description = st.text_area("Enter a movie description")

    if st.button("Get Recommendations") and movie_description != "":
        with st.spinner("Loading..."):
            client = OpenAI(api_key=key)
            vector = client.embeddings.create(
                model = "text-embedding-ada-002",
                input = movie_description
            )

            result_vector = vector.data[0].embedding

            index = Pinecone(api_key=pinecone_key).Index(index)
            result = index.query(
                vector=result_vector, 
                top_k=5,
                include_metadata=True
            )

            for movie in result.matches:
                st.write(f"Title: {movie['metadata']['title']}")