import streamlit as st
from dotenv import load_dotenv
import time
import random
from utils import get_words, check_words

load_dotenv()

# Set headers
st.set_page_config(page_title="LangConnector", page_icon="🔗")
st.title("LangConnector")

# Initialization settings
if "initialized" not in st.session_state:
    st.session_state.word_history = []
    st.session_state.initialized = True
    st.session_state.game_started = False
    st.session_state.first_loop = True


# Set up new game
def start_game_screen(language: str, topic: str):
    # Get words
    word_set, word_set_eng = get_words(language, topic)

    # Create word graph
    st.session_state.word_graph = {word1: word2 for (word1, word2) in zip(word_set, word_set_eng)}

    # Unpack words
    random.shuffle(word_set)
    random.shuffle(word_set_eng)
    st.session_state.word_set = word_set
    st.session_state.word_set_eng = word_set_eng

    # Save eng words to dict
    for word in word_set_eng:
        st.session_state.word_history.append(word)

    # Initialize the game variables
    st.session_state.game_started = True
    st.session_state.mistake_count = 0
    st.session_state.time = time.time()
    st.session_state.clicked_left = False
    st.session_state.clicked_right = False

    st.rerun()


# Render start screen
if not st.session_state.game_started:
    # Create language and topic dropdown
    languages = ["Dutch", "Croatian", "Spanish", "French", "German"]

    # Start screen
    with st.spinner("Generating game..."):
        with st.form(key="game_generation_form", clear_on_submit=True):
            language = st.selectbox("Select language", languages)
            topic = st.text_input("Topic: ", key="input", placeholder="History")
            start = st.form_submit_button("Start")
        if language and topic and start:
            start_game_screen(language, topic)
else:
    # Buttons setup
    buttons_left_labels = st.session_state.word_set
    buttons_right_labels = st.session_state.word_set_eng
    keys_left = [k+str(i) for i, k in enumerate(st.session_state.word_set)]
    keys_right = [k+str(i)+'eng' for i, k in enumerate(st.session_state.word_set_eng)]

    # Display buttons in these columns
    st.write("**Connect words to their translation!**")

    # Create two columns of words
    col1, col2 = st.columns(2)
    with col1:
        for key, label in zip(keys_left, buttons_left_labels):
            if st.button(label.replace('_', ' ').capitalize(), key=key):
                st.session_state.clicked_left = label
    with col2:
        for key, label in zip(keys_right, buttons_right_labels):
            if st.button(label.replace('_', ' ').capitalize(), key=key):
                st.session_state.clicked_right = label

    # Check if the words are correct
    if check_words(st.session_state.clicked_left, st.session_state.clicked_right):
        st.session_state.word_set.remove(st.session_state.clicked_left)
        st.session_state.word_set_eng.remove(st.session_state.clicked_right)
        st.session_state.clicked_left = False
        st.session_state.clicked_right = False
        st.session_state.first_loop = True
        st.rerun()

    # End game screen
    if not st.session_state.word_set and not st.session_state.word_set_eng:
        duration = round(time.time() - st.session_state.time, 3)
        st.write(f"**Time completion:** {duration} seconds")

    # Reload button
    st.write("\n"*3)
    reload_button = st.button("RELOAD")
    if reload_button:
        st.session_state.game_started = False
        st.rerun()
