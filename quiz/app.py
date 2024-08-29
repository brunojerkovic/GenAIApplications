import streamlit as st
from streamlit_chat import message
from utils import PAGE, read_pdf
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()


def init():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PAGE.MAIN
        st.session_state.mcq_question_number = 10
        st.session_state.mcq_false_answer_number = 3
        st.session_state.llm = OpenAILLM(mcq_question_number=st.session_state.mcq_question_number,
                                         mcq_false_answer_number=st.session_state.mcq_false_answer_number)
        st.session_state.chat_start = False
        st.session_state.chat_messages = []

    # Setting page title and header
    st.set_page_config(page_title="AILearningBuddy", page_icon=":book:")
    st.markdown("<h1 style='text-align: center;'>AI Learning Buddy</h1>", unsafe_allow_html=True)


def main_page():
    # Header
    st.header("Main Page")

    # Upload docs
    file = st.file_uploader("Upload documents", type=["pdf", "txt"])
    MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
    if file is not None:
        # Check the file size
        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            st.error(f"File size should not exceed {MAX_FILE_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
        else:
            st.success("File uploaded successfully!")

            # Read file based on its type
            if file.type == "application/pdf":
                text = read_pdf(file)
                st.session_state.llm.upload_text(text)
            elif file.type == "text/plain":
                text = file.read().decode("utf-8")
                st.session_state.llm.upload_text(text)
            else:
                st.error("Unsupported file type.")

    # Display buttons if file is uploaded
    if st.session_state.llm.is_text_uploaded():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<h4>LEARN</h4>", unsafe_allow_html=True)

            if st.button("Create summary", key="summary_button"):
                st.session_state.current_page = PAGE.SUMMARY
                st.rerun()
            if st.button("Chat about the file", key="chat_button"):
                st.session_state.current_page = PAGE.CHAT
                st.session_state.chat_start = True
                st.rerun()
        with col2:
            st.markdown("<h4>TEST</h4>", unsafe_allow_html=True)

            if st.button("Create quiz", key="mcq_button"):
                st.session_state.current_page = PAGE.MCQ
                st.session_state.current_question = 0
                st.rerun()


def summary_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.llm.empty_text()
        st.rerun()
    st.header("Summary")

    # Get the summary
    summary = st.session_state.llm.get_text_summary()

    # Write summary
    st.write(summary)


def chat_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.chat_start = False
        st.session_state.chat_messages = []
        st.session_state.llm.empty_text()
        st.rerun()
    st.header("Chat About the Document")

    # Response and user container
    response_container = st.container()
    user_container = st.container()

    with user_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Type here:", key='input', height=100)
            send_button = st.form_submit_button(label='Send')

            if send_button or st.session_state.chat_start:
                # Get the model response, and save it
                if st.session_state.chat_start:
                    user_input, model_response = st.session_state.llm.start_chat()
                    st.session_state.chat_start = False
                else:
                    model_response = st.session_state.llm.get_chat_response(user_input)
                st.session_state.chat_messages += [user_input, model_response]

            # Display chat messages
            with response_container:
                for i in range(1, len(st.session_state.chat_messages)):
                    if i % 2:
                        message(st.session_state.chat_messages[i], key=f'{str(i)}_AI', avatar_style="pixel-art")
                    else:
                        message(st.session_state.chat_messages[i], is_user=True, key=f'{str(i)}_user',
                                avatar_style="adventurer-neutral")


def mcq_page():
    # Header
    if st.button(":back: Main Page"):
        st.session_state.current_page = PAGE.MAIN
        st.session_state.current_question = 0
        st.session_state.llm.empty_text()
        st.rerun()

    # Setup MCQ
    if st.session_state.current_question == 0:
        # Start MCQ and get the first question and answer
        st.session_state.llm.start_mcq()
        st.session_state.question, st.session_state.answers = st.session_state.llm.get_mcq_question()
        st.session_state.current_question += 1

    # Handler when pressing next
    def increase_current_question():
        st.session_state.current_question = st.session_state.current_question + 1
        st.session_state.llm.mcq_record_answer(st.session_state.selected_answer)
        st.session_state.question, st.session_state.answers = st.session_state.llm.get_mcq_question()

    # For every MCQ question
    if st.session_state.current_question <= st.session_state.mcq_question_number:
        # QA header
        st.header(f"Question {st.session_state.current_question} / {st.session_state.mcq_question_number}")

        # QA form
        with st.form(key='my_form', clear_on_submit=True):
            st.session_state.selected_answer = st.radio(f"{st.session_state.question}:", st.session_state.answers)
            st.form_submit_button(label="Next", on_click=increase_current_question)
    else:
        # Results header
        st.header("Results")

        # For the last QA, show score
        # st.session_state.current_question += 1
        score, score_perc = st.session_state.llm.get_mcq_score()
        st.markdown("<h4>" + f"Score: {score} / {st.session_state.mcq_question_number} ({score_perc} %)" + "</h4>", unsafe_allow_html=True)

        # List your answers and the correct ones
        for i, qa in enumerate(st.session_state.llm.mcq_answer_sheet[:-1]):
            question, answer, user_answer = qa['question'], qa['answer'], qa['user_answer']
            st.write("---")
            st.write(f"**Question {i+1}/{st.session_state.mcq_question_number}:** {question}")
            st.write(f"**Correct answer:** {answer}")
            st.write(f"**User answer:** {user_answer}")


# Main structure
init()

# Page selector
match st.session_state.current_page:
    case PAGE.MAIN:
        main_page()
    case PAGE.SUMMARY:
        summary_page()
    case PAGE.CHAT:
        chat_page()
    case PAGE.MCQ:
        mcq_page()
