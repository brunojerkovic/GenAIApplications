import json
from openai import OpenAI
import streamlit as st


# Get the words from the LLM
def get_words(language: str, topic: str):
    client = OpenAI()
    message = {}

    # Get the response from LLM
    while 'word_set' not in message or 'word_set_eng' not in message:
        query_text = f"Generate 10 words in language:{language} on the topic of topic:{topic}. After that generate their translations in English."
        query_text += f"Make sure that the generated words are NOT one of the following English words: {st.session_state.word_history}"
        query_text += """Output the data in JSON format using the following structure: {'array': {'word_set': [], 'word_set_eng': []}}"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "user",
                 "content": query_text}
            ],
            response_format={"type": "json_object"},
            temperature=1.
        )
        message = json.loads(response.choices[0].message.content)['array']
    return message['word_set'], message['word_set_eng']


def check_words(word1_original: str, word_eng: str):
    if not word1_original or not word_eng:
        return False

    return True if st.session_state.word_graph[word1_original] == word_eng else False
