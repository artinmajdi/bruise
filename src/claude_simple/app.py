import streamlit as st
from components.sidebar import create_sidebar
import importlib


def main():
    st.set_page_config(
        page_title="Bruise Detection Postdoc Interview Prep",
        page_icon="🏥",
        layout="wide",
    )

    page = create_sidebar()

    # Dynamically import and display the selected page
    try:
        page_module = importlib.import_module(f"pages.{page.lower().replace(' ', '_')}")
        page_module.show()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.info("Please select a valid page from the sidebar.")

if __name__ == "__main__":
    main()
