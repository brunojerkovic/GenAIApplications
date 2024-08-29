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


def page_generate_emails():
    form_input = st.text_area("Enter the email topic", height=275)

    # Creating columns for the UI - to receive inputs from user
    col1, col2, col3 = st.columns([10, 10, 5])

    with col1:
        email_sender = st.text_input("Sender name")
    with col2:
        email_receiver = st.text_input("Receiver name")
    with col3:
        email_style = st.selectbox(
            "Writing style",
            ("Formal", "Appreciating", "Not Satisfied", "Neutral"),
            index=0
        )
    submit = st.button("Generate")

    if submit:
        # Generate email
        email_text = st.session_state.llm.translate(form_input, email_sender, email_receiver, email_style)
        st.write(email_text)

# Main structure
init()

# Page opener
page_generate_emails()