import streamlit as st
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.llm = OpenAILLM(model_name="gpt-4o",
                                         temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AIDatabaseQueryTranslator", page_icon="💾")
    st.markdown("<h1 style='text-align: center;'>AI Database Query Translator</h1>", unsafe_allow_html=True)


def page_create_story():
    # Search bar
    st.header("Enter your query in natural text:")

    with st.form(key='my_form', clear_on_submit=False):
        # User query
        user_query = st.text_area(
            "Enter the description of your SQL query:",
            height=20,
            placeholder="Retrieve all users whose location is in the US.",
        )

        # Story language
        language = st.selectbox("Query language:", ["SQL", "neo4j", "MongoDB", "Gremlin", "XQuery"])

        # Button to create the story
        create_button = st.form_submit_button(label='Translate')

    # On send button
    if create_button:
        # Generate story
        with st.spinner("Gathering magic to generate a perfect story..."):
            # Generate story text
            response = st.session_state.llm.translate(user_query, language)

            st.subheader("Code:")
            st.code(response)


# Main structure
init()

# Page opener
page_create_story()