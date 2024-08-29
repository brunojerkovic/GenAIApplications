import streamlit as st
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
from utils import read_file
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.llm = OpenAILLM(model_name="gpt-3.5-turbo-0125", temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AICVHelper", page_icon="🏢")
    st.markdown("<h1 style='text-align: center;'>AI CV / Cover Letter Helper</h1>", unsafe_allow_html=True)


def main_menu():
    # Select document type
    document_types = ["CV", "Cover Letter"]
    document_type = st.selectbox("What do you wish to optimize?", document_types)

    # Upload document
    document = None
    file = st.file_uploader("Upload documents", type=["pdf", "docx"])
    MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
    if file is not None:
        # Check the file size
        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            st.error(f"File size should not exceed {MAX_FILE_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
        else:
            st.success("File uploaded successfully!")

            # Read file based on its type
            document = read_file(file)
            if document is None:
                st.error("Unsupported file type.")

    # Job descriptions
    job_description = st.text_area("Paste job description here:",
                                   key='input',
                                   height=500)

    # Submit button
    button_rewrite = st.button("Rewrite", key='rewrite')

    # Results
    if button_rewrite:
        with st.spinner("Waiting for the model response..."):
            # Get model response
            score_current, model_comment, document_new, score_new = st.session_state.llm.get_response(document, job_description, document_type)

        # Current match score
        if score_current:
            st.write(f"**Current match score is estimated to be:** {score_current}")

        # Model comment
        st.write("**What to improve:**")
        st.write(model_comment)

        # New CV/cover letter
        st.write("**New document:**")
        st.write(document_new)

        # New match score
        if score_new:
            st.write(f"**New match score is estimated to be:** {score_new}")


# Page content
init()
main_menu()
