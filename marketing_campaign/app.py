import streamlit as st
from dotenv import load_dotenv
from prompt_generation import prompt_generator
from result_saver import save_result

load_dotenv()

# Init block
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.generated_text = ""

# Set headers
st.set_page_config(page_title="MarketingAssistant", page_icon="🔗", layout="centered", initial_sidebar_state="collapsed")
st.header("Hey, I am your Marketing Assistant that learns from you!")

# Input form
form_input = st.text_area("Enter a raw text to be rewritten:", height=275)

# Roles
task_types = ["Sales copy", "Tweet", "Product description"]
task_type_option = st.selectbox("Action to be performed:", task_types, key=1)

# Age groups
age_groups = ["Kid", "Adult", "Senior citizen"]
age_option = st.selectbox("Target audience age group:", age_groups, key=2)

token_number = st.slider("Input token limit", 1, 1000, 50)

# Submit button
submit = st.button("Generate")
if submit:
    st.session_state.generated_text = prompt_generator(text=form_input, task_type=task_type_option, age_group=age_option, token_number=token_number)
    st.write(st.session_state.generated_text)

# Feedback loop
save_result_options = ["Yes", "No", "Don't want to contribute"]
if st.session_state.generated_text:
    save_result_flag = st.selectbox("*Did you like this result? (please contribute by providing a feedback, so that we can improve)*", save_result_options, index=2)

    if save_result_options.index(save_result_flag) != 2:
        save_result(input_text=form_input, task_type=task_type_option, age_group=age_option, output_text=st.session_state.generated_text, save_result_option=save_result_options.index(save_result_flag))
        st.session_state.generated_text = ""
