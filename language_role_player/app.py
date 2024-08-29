import streamlit as st
from streamlit_chat import message
from prompt_generation import get_response, get_response_init
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Setting page title and header
    st.set_page_config(page_title="Chat GPT with Summarization Transcript", page_icon=":tongue:")
    st.markdown("<h1 style='text-align: center;'>Language Role-playing</h1>", unsafe_allow_html=True)


def side_bar():
    # Sidebar
    st.sidebar.title("Settings")
    if "conversation" in st.session_state:
        summary_type = st.sidebar.selectbox("Summary type", ["Brief", "Detailed"])
        history = st.session_state.conversation.memory.buffer if summary_type == "Detailed" else st.session_state.summary_memory.buffer
        st.sidebar.download_button(
            label="Download Conversation Summary",
            data=history,
            file_name="conversation.txt",
            mime="text/plain"
        )
        if st.sidebar.button("REFRESH CONVERSATION"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    else:
        st.markdown("""
        **Instructions:**
        1. Create scenario in the **left (Settings) tab**.
        2. Let's have some chit-chat. AI will try to correct you if it notices that you made an error.
        3. Download **the brief conversation summary** (or the detailed one) that you can use to study later.
        """)
        # st.write("Create scenario in the **left (Settings) tab**. Then, we will start talking. Later, be sure to download **the brief conversation summary** that you can use to study.")
        st.session_state.language = st.sidebar.selectbox("Language", ["English", "Dutch", "Croatian", "Spanish"])
        st.session_state.role_ai = st.sidebar.text_input("AI will pretend to be:", placeholder="Cashier")
        st.session_state.role_human = st.sidebar.text_input("Human will pretend to be:", placeholder="Customer")
        st.session_state.scenario_description = st.sidebar.text_area("The scenario will be:", placeholder="The customer is just about to pay, but does not have enough money.")
        if st.sidebar.button("Start Role-Play"):
            st.session_state.in_scene = True
            st.session_state.start = True


def chat():
    # Response and user container
    response_container = st.container()
    user_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area(st.session_state.role_human+":", key='input', height=100)
            send_button = st.form_submit_button(label='Send')

            if send_button or st.session_state.start:
                # To remove summary type check-box
                if "messages" not in st.session_state:
                    st.session_state['messages'] = []

                # Get the model response, and save it
                if st.session_state.start:
                    user_input, model_response = get_response_init()
                    st.session_state.start = False
                else:
                    model_response = get_response(user_input)
                st.session_state.messages += [user_input, model_response]

            with response_container:
                for i in range(1, len(st.session_state.messages)):
                    if i % 2:
                        message(st.session_state.messages[i], key=f'{str(i)}_AI', avatar_style="pixel-art")
                    else:
                        message(st.session_state.messages[i], is_user=True, key=f'{str(i)}_user', avatar_style="adventurer-neutral")


# Main structure
init()
side_bar()

if "in_scene" in st.session_state and st.session_state.in_scene:
    chat()
