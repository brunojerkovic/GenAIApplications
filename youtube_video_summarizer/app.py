import streamlit as st
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.llm = OpenAILLM(model_name="gpt-4o", temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AIVideoSummarizer", page_icon=":cinema:")
    st.markdown("<h1 style='text-align: center;'>AI YT Video Summarizer</h1>", unsafe_allow_html=True)


# Init
init()

# Search bar
with st.form(key='my_form', clear_on_submit=False):
    user_input = st.text_input("Enter YT video URL:")
    word_number = st.slider("Select the number of words in a summary", min_value=0, max_value=1_000, value=200, step=50)
    send_button = st.form_submit_button(label='Summarize')

if send_button:
    # Check if the link is from YouTTube
    if not user_input.startswith("https://www.youtube.com/watch"):
        st.error("The link must be a valid link from a YT video!")
    else:
        summary = st.session_state.llm.get_transcript(user_input, word_number)
        st.header("Summary")
        st.write(summary)
        st.download_button(
            label="Download",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
