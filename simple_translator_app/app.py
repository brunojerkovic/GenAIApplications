from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
load_dotenv()


# Load the answer
def load_answer(sentence, source_language, dest_language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.)
    question = f"Translate from {source_language} to {dest_language}. Translate the following sentence: {sentence}"
    answer = llm.invoke(question).content
    return answer

# App UI starts here
st.set_page_config(page_title="LangChain Translator", page_icon=":robot:")
st.header("LLM Translator")

# Dropdown for languages
languages = ["Croatian", "English", "Spanish", "French", "German", "Dutch"]
source_language = st.selectbox("Select source language", languages)
dest_language = st.selectbox("Select destination language", languages[::-1])

# Get the user text
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

# Get the user input
user_input = get_text()
response = load_answer(user_input, source_language, dest_language)

# Get the submit button
submit = st.button("Translate")
if submit:
    st.subheader("Translation:")
    st.write(response)
