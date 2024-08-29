import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory


def get_response(user_input, api_key):
    if 'conversation' not in st.session_state:
        llm = ChatOpenAI(
            temperature=0,
            api_key=api_key,
            model_name='gpt-4o'
        )
        st.session_state.summary_memory = ConversationSummaryMemory(llm=llm)

        st.session_state.conversation = ConversationChain(
            llm=llm,
            verbose=False,
            memory=ConversationBufferMemory()
        )

    response = st.session_state.conversation.predict(input=user_input)  # Inference
    st.session_state.summary_memory.save_context({"input": user_input}, {"output": response})  # Save in memory

    return response
