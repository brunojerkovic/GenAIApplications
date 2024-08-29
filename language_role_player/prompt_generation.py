import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory


def get_response_init():
    # Initialize model
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4o'
    )

    # Initialize extra memory
    st.session_state.summary_memory = ConversationSummaryMemory(llm=llm)

    # Initialize model's memory
    st.session_state.conversation = ConversationChain(
        llm=llm,
        verbose=False,
        memory=ConversationBufferMemory()
    )

    # Set system message
    system_message = f"""
        Talk in language: {st.session_state.language}.
        I want you to pretend you are: {st.session_state.role_ai}.
        I will pretend that I am: {st.session_state.role_human}.
        This is the scenario description: {st.session_state.scenario_description}.
        
        If I make a grammatical mistake, then break the scene, correct me, and continue in the scene.
        OTHER THAN THAT, NEVER STEP OUT OF THE ROLE!
        You start first!
    """
    response = st.session_state.conversation.predict(input=system_message)
    st.session_state.summary_memory.save_context({"input": system_message}, {"output": response})  # Save in memory

    return system_message, response


def get_response(user_input):
    response = st.session_state.conversation.predict(input=user_input)  # Inference
    st.session_state.summary_memory.save_context({"input": user_input}, {"output": response})  # Save in memory
    return response
