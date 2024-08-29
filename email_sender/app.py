import streamlit as st
from prompt_generation import CustomLLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.llm = CustomLLM(temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AIEmailWriter", page_icon="📧", layout="centered", initial_sidebar_state="collapsed")
    st.markdown("<h1 style='text-align: center;'>AI Email Writer</h1>", unsafe_allow_html=True)


def main():
    # Upload multiple files
    uploaded_files = st.file_uploader("Upload recorded .mp3 files", type=["mp3"], accept_multiple_files=True)

    # Add recepient
    recepient = st.text_input("Recipient", value="brunojerkovi@gmail.com")

    if uploaded_files and recepient:
        st.write("**Uploaded files:**")
        col1, col2, col3 = st.columns([0.1, 1, 2])

        # Display uploaded files and buttons in a tabular form
        for uploaded_file in uploaded_files:
            with col1:
                st.write("-")
            with col2:
                st.write(uploaded_file.name)
            with col3:
                send_button = st.button(f"Send Email to: {recepient}")

                if send_button:
                    st.session_state.llm.send_email(uploaded_file, recepient)
                    st.success(f"Email sent to: {recepient}")

# Main structure
init()

# Page opener
main()