import streamlit as st
from prompt_generation import OpenAILLM
from dotenv import load_dotenv
from utils import PAGE
load_dotenv()


def init():
    if 'init' not in st.session_state:
        st.session_state.current_page = PAGE.SEARCH
        st.session_state.init = True
        st.session_state.llm = OpenAILLM(model_name="gpt-3.5-turbo-0125",
                                         temperature=0.)  # Other models: ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]

    # Setting page title and header
    st.set_page_config(page_title="AIStockNews", page_icon="💸")
    st.markdown("<h1 style='text-align: center;'>AI Stock News</h1>", unsafe_allow_html=True)


def page_search():
    # Switch to upload more news
    st.write("Click here if you want to upload more news:")
    if st.button("Upload more news"):
        st.session_state.current_page = PAGE.UPLOAD
        st.rerun()

    # Search bar
    st.header("Search database")
    with st.form(key='my_form', clear_on_submit=False):
        # User query
        user_query = st.text_input("Enter your question:")
        send_button = st.form_submit_button(label='Submit')

    # On send button
    if send_button:
        answer, related_articles, dates, links, source_types, price_tables, related_stocks = st.session_state.llm.get_news(user_query)

        # Provide answer
        st.markdown("## Answer:")
        st.write(answer)
        st.markdown("---")

        # Iterate results
        for i, (related_article, date, link, source_type, price_table, related_stocks_list) in enumerate(zip(related_articles, dates, links, source_types, price_tables, related_stocks)):
            st.markdown(f"### Related article {i + 1}:")

            # Provide metadata of an article
            st.write("**Link:** "+link)
            st.write("**Date:** "+date)
            st.write("**Source type:** "+source_type)

            # Provide the summary of an article
            st.write("**Text:**")
            st.write(related_article[6:])

            for stock in related_stocks_list:
                st.write("**" + stock + "**:")
                st.table(price_table[stock])
            st.markdown("---")


def page_upload():
    # Switch to serach
    st.write("Click here if you want to search the database:")
    if st.button("Search existing news"):
        st.session_state.current_page = PAGE.SEARCH
        st.rerun()

    # Add information
    st.header("Add new stock-related source:")
    with st.form(key='my_form', clear_on_submit=True):
        # Source
        sources = ["Tweet", "News article"]
        source_selected = st.selectbox("Select source: ", sources)

        # Date input
        selected_date = st.date_input("Select a date")

        # Link adding
        link = st.text_input("Link:", value="https://")

        # Text input
        text_input = st.text_area("Input source text here:",
                                  key='input',
                                  height=500)

        # Save button
        save_button = st.form_submit_button(label='Add to database')

    if save_button:
        with st.spinner("Saving..."):
            st.session_state.llm.save_source(source_selected, text_input, link, selected_date)
            st.success("Your source was uploaded successfully")


# Main structure
init()

# Page selector
match st.session_state.current_page:
    case PAGE.UPLOAD:
        page_upload()
    case PAGE.SEARCH:
        page_search()
