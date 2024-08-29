import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

# Set headers
st.set_page_config(page_title="AI Dutch Teacher", page_icon="🇳🇱")
st.header("Online Nederlandse Leraar")

# Set session memory
if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [
        SystemMessage(content="You are a Dutch teacher. Only speak in Dutch!"),
        AIMessage(content="Hoi! Ik ben je AI Nederlandse leraar. Wat wil je leren?")
    ]

# Instantiate LLM
llm = ChatOpenAI(temperature=0.)


def load_answer(question):
    st.session_state.sessionMessages.append(HumanMessage(content=question))  # Save human message
    assistant_answer = llm.invoke(st.session_state.sessionMessages).content  # Get answer of chat
    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer))  # Memorize AI's message
    return assistant_answer


# Display chat history
for message in st.session_state.sessionMessages:
    if isinstance(message, HumanMessage):
        st.write(f"**Jij:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**AI:** {message.content}")

# New text form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Jij: ", key="input")
    submit = st.form_submit_button("Verzenden")

# Handle text
if user_input and submit:
    with st.spinner("Wachten op het antwoord..."):
        response = load_answer(user_input)
        st.rerun()
