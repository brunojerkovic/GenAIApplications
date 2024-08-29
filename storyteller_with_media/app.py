import streamlit as st
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.llm = OpenAILLM(model_name="gpt-4o",
                                         temperature=1.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AIStoryteller", page_icon="🌒")
    st.markdown("<h1 style='text-align: center;'>AI Storyteller</h1>", unsafe_allow_html=True)


def page_create_story():
    # Search bar
    st.header("Create a story on your own")

    with st.form(key='my_form', clear_on_submit=False):
        # User query
        user_query = st.text_area(
            "Enter the description of your story:",
            height=200,
            placeholder="Story about a man that became a spider...",
            value="Story about a man that became a spider."
        )

        # Story language
        language = st.selectbox("Story language:", ["English", "Croatian", "Spanish", "German", "Dutch"])

        # Number of words
        words_len = st.slider("Desired number of words in your story: ", min_value=0, max_value=300, value=500, step=50)

        # Button to create the story
        create_button = st.form_submit_button(label='Create')

    # On send button
    if create_button:
        # Generate story
        with st.spinner("Gathering magic to generate a perfect story..."):
            # Generate story text
            story_chunks, story_full = st.session_state.llm.create_story(user_query, words_len, language, chunk_size=5)

            # Generate story sound
            filename = st.session_state.llm.generate_tts(story_full)  # TODO: make the background shading follow text reading
            st.audio(filename, autoplay=True)

            # Show completion message
            # st.success("Your story has been generated. Enjoy!")

        # Show story chunks
        for story_chunk in story_chunks:
            # Display a story
            st.write(story_chunk)

            # Generate image
            url = st.session_state.llm.create_images(story_chunk)
            st.image(url, use_column_width=True)


# Main structure
init()

# Page opener
page_create_story()