import streamlit as st

def create_sidebar():
    """
    Creates the sidebar navigation and returns the selected page.
    """
    with st.sidebar:
        st.title("Interview Prep Dashboard")
        
        # Try to display ALS diagram image if it exists
        try:
            st.image("assets/als_diagram.png", use_column_width=True)
        except:
            st.info("ALS diagram visualization")
        
        # Navigation options
        selected = st.radio(
            "Navigate to:",
            [
                "Project Overview",
                "Team Profiles",
                "Technical Stack",
                "Interview Prep",
                "Demo Concept",
                "Research Papers"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This dashboard helps prepare for the GMU Postdoc interview "
            "on the Equitable and Accessible Software for Injury Detection (EAS-ID) project."
        )
    
    return selected
