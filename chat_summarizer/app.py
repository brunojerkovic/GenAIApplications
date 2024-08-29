import streamlit as st
from streamlit_chat import message
from prompt_generation import get_response


def init():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Setting page title and header
    st.set_page_config(page_title="Chat GPT with Summarization Transcript", page_icon=":book:")
    st.markdown("<h1 style='text-align: center;'>How can I assist you today? </h1>", unsafe_allow_html=True)


def side_bar():
    # Sidebar
    st.sidebar.title("Credentials")
    st.session_state['API_Key'] = st.sidebar.text_input("What's your API key?", type="password")
    summary_type = st.sidebar.selectbox("Summary type", ["Detailed", "Shorter"])
    # summarise_download = st.sidebar.button("Download conversation summary", key="summarise")
    # Create a download button for the .txt file
    if "conversation" in st.session_state:
        history = st.session_state.conversation.memory.buffer if summary_type == "Detailed" else st.session_state.summary_memory.buffer
        st.sidebar.download_button(
            label="Download Conversation Summary",
            data=history,
            file_name="conversation.txt",
            mime="text/plain"
        )


def chat():
    # Response and user container
    response_container = st.container()
    user_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Your question goes here:", key='input', height=100)
            send_button = st.form_submit_button(label='Send')

            if send_button:
                # To remove summary type check-box
                if "messages" not in st.session_state:
                    st.session_state['messages'] = []
                    st.rerun()

                # Get the model response, and save it
                model_response = get_response(user_input, st.session_state['API_Key'])
                st.session_state.messages += [user_input, model_response]

            with response_container:
                for i in range(len(st.session_state.messages)):
                    if i % 2:
                        message(st.session_state.messages[i], key=f'{str(i)}_AI', avatar_style="pixel-art")
                    else:
                        message(st.session_state.messages[i], is_user=True, key=f'{str(i)}_user', avatar_style="adventurer-neutral")


# Main structure
init()
side_bar()
chat()
