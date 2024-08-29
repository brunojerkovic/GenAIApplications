import streamlit as st
from prompt_generation import CustomLLM
from utils import PAGE
from io import StringIO
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PAGE.MAIN
        st.session_state.llm = CustomLLM(temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AICodingAnalyst", page_icon="💻", layout="centered", initial_sidebar_state="collapsed")
    st.markdown("<h1 style='text-align: center;'>AI Coding Analyst</h1>", unsafe_allow_html=True)


def page_main():
    # Header
    st.header("Choose what to do with your code:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Translate to a new programming language"):
            st.session_state.current_page = PAGE.TRANSLATE
            st.rerun()
    with col2:
        if st.button("Review and optimize my code"):
            st.session_state.current_page = PAGE.REVIEW
            st.rerun()


def page_translate():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.rerun()
    st.header("Choose the source code and language")

    # Upload code file
    uploaded_file = st.file_uploader("Upload your code", accept_multiple_files=False)

    # Select a language to translate to
    language = st.text_input("Language", value="Powershell")

    # button
    button_translate = st.button("Translate")

    if button_translate:
        # Read code
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
        code = stringio.read()

        # Get the output
        with st.spinner("Translating the code..."):
            translated_code = st.session_state.llm.translate_code(code, language)
            st.write(translated_code)


def page_review():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.rerun()
    st.header("Code Reviewer")

    # Upload code file
    uploaded_file = st.file_uploader("Upload your code", accept_multiple_files=False)

    # Button
    button_review = st.button("Review")

    if button_review:
        # Read code
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
        code = stringio.read()

        # Ge the output
        with st.spinner("Reviewing the code..."):
            new_code, comments = st.session_state.llm.review_code(code)
            st.code(new_code)
            st.write(comments)


# Main structure
init()

# Page selector
match st.session_state.current_page:
    case PAGE.MAIN:
        page_main()
    case PAGE.TRANSLATE:
        page_translate()
    case PAGE.REVIEW:
        page_review()
