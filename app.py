import streamlit as st
import generic
import frontend

if __name__ == '__main__':
    # Initialize
    generic.init_session()
    generic.init_params()
    # init_nlp()

    # Sidebar
    # Data
    generic.read_data()
    pages = frontend.show_layout(type='page')
    # groups = frontend.display_sidebar()
    prev_page, next_page, update_status = frontend.display_texts(pages=pages)
    frontend.display_sidebar()

    # groups = frontend.display_sidebar()
    # if groups:
    #     pages = frontend.show_layout(type='page')
    #     prev_page, next_page, update_status = frontend.display_texts(pages=pages,groups=groups)
    #     frontend.display_sidebar()