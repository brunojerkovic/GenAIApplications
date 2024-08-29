import pandas as pd
import streamlit as st
from streamlit_chat import message
from utils import PAGE, read_file
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PAGE.MAIN
        st.session_state.llm = OpenAILLM(model_name="gpt-3.5-turbo-0125", temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]
        st.session_state.chat_start = False

    # Setting page title and header
    st.set_page_config(page_title="AIStatistician", page_icon=":bar_chart:")
    st.markdown("<h1 style='text-align: center;'>AI Statistician</h1>", unsafe_allow_html=True)


def main_page():
    # Header
    st.header("Main Page")

    # Upload docs
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
            data: pd.DataFrame = read_file(file)
            if data is None:
                st.error("Unsupported file type.")
            else:
                st.session_state.llm.upload_df(data)

    # Display buttons if file is uploaded
    if st.session_state.llm.is_df_uploaded():
        if st.button("Chat"):
            st.session_state.current_page = PAGE.CHAT
            st.session_state.chat_start = True
            st.rerun()
        if st.button("Visualize"):
            st.session_state.current_page = PAGE.VISUAL
            st.session_state.chat_start = True
            st.rerun()
        if st.button("Generate Code for Stats"):
            st.session_state.current_page = PAGE.STATS
            st.rerun()


def chat_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.chat_start = False
        st.session_state.llm.empty_df()
        st.rerun()
    st.header("Ask questions about the data")

    # Response and user container
    user_container = st.container()
    response_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=False):
            user_input = st.text_area("Type your question:", key='input', height=100)
            send_button = st.form_submit_button(label='Ask')

            if send_button or st.session_state.chat_start:
                # Get the model response, and save it
                if st.session_state.chat_start:
                    model_response = st.session_state.llm.get_chat_start_message()
                    st.session_state.chat_start = False
                else:
                    model_response = st.session_state.llm.get_chat_response(user_input)

            # Display chat messages
            with response_container:
                message(model_response, avatar_style="pixel-art")


def visualize_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.chat_start = False
        st.session_state.llm.empty_df()
        st.rerun()
    st.header("Generate Visualizations")

    # Response and user container
    user_container = st.container()
    response_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=False):
            user_input = st.text_area("Describe your visualization:", key='input', height=100)
            send_button = st.form_submit_button(label='Generate visualization')

            if send_button or st.session_state.chat_start:
                # Get the model response, and save it
                if st.session_state.chat_start:
                    model_response = st.session_state.llm.get_visualize_start_message()
                    st.session_state.chat_start = False

                    # Display chat message
                    with response_container:
                        message(model_response, avatar_style="pixel-art")
                else:
                    fig, model_response = st.session_state.llm.get_visualization_response(user_input)

                    # Display message
                    with response_container:
                        message(model_response, avatar_style="pixel-art")

                    # Display figure
                    if fig:
                        st.pyplot(fig)


def statistics_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.chat_start = False
        st.session_state.llm.empty_df()
        st.rerun()
    st.header("Get Code for Stats")

    # Response and user container
    user_container = st.container()
    response_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=False):
            user_input = st.text_area("I want to know...", key='input', height=100, placeholder="The relation between person's age and their survival.")
            send_button = st.form_submit_button(label='Generate Code')

            if send_button or st.session_state.chat_start:
                # Get the model response, and save it
                if st.session_state.chat_start:
                    model_response = st.session_state.llm.get_stats_start_message()
                    st.session_state.chat_start = False

                    # Display chat message
                    with response_container:
                        message(model_response, avatar_style="pixel-art")
                else:
                    model_response_message, code = st.session_state.llm.get_stats_response(user_input)

                    # Display message
                    with response_container:
                        message(model_response_message, avatar_style="pixel-art")
                        if code:
                            message(code, avatar_style="pixel-art")


# Main structure
init()

# Page selector
match st.session_state.current_page:
    case PAGE.MAIN:
        main_page()
    case PAGE.CHAT:
        chat_page()
    case PAGE.VISUAL:
        visualize_page()
    case PAGE.STATS:
        statistics_page()
