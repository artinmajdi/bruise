import streamlit as st
from components.visualizations import create_timeline

def show():
    """
    Display the project overview page.
    """
    st.title("EAS-ID Project Overview")
    
    st.markdown("""
    ## Equitable and Accessible Software for Injury Detection (EAS-ID)
    
    The EAS-ID project is developing a mobile AI tool that makes bruises visible across all skin tones, 
    addressing a critical gap in forensic nursing and intimate partner violence care.
    
    ### Key Facts:
    - Recently received $4.85 million in philanthropic funding
    - Additional NIH AIM-AHEAD supplement for equity in AI
    - Uses alternate light source (ALS) technology, which is 5 times more effective than white light
    - Employs deep learning to analyze multi-spectral images
    - Aims to address healthcare disparities in bruise detection across all skin tones
    """)
    
    # Create project timeline
    events = [
        {"event": "Initial Research", "start": 0, "duration": 2},
        {"event": "DOJ Grant", "start": 2, "duration": 1},
        {"event": "$4.85M Funding", "start": 3, "duration": 1},
        {"event": "EAS-ID Platform Development", "start": 4, "duration": 3},
        {"event": "Postdoc Position", "start": 5, "duration": 0.5}
    ]
    
    timeline_fig = create_timeline(events, "Project Timeline (Years)")
    st.pyplot(timeline_fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Goals:
        
        1. Develop a smartphone + cloud platform for multi-spectral imaging
        2. Create deep learning models to detect bruises across all Fitzpatrick skin types
        3. Build a secure image repository with over 50,000 images
        4. Implement a fairness dashboard to ensure equitable performance
        5. Design user interfaces for forensic nurses and healthcare providers
        """)
    
    with col2:
        st.markdown("""
        ### Technical Stack (in progress):
        
        - **Frontend**: Mobile application for capturing images
        - **Backend**: Cloud-based microservices architecture
        - **AI**: Python CV pipelines, deep neural networks
        - **Data**: Secure image repository (>50k images)
        - **Monitoring**: Fairness dashboard across skin tones
        """)
    
    st.markdown("""
    ### Social Impact:
    
    The bruise detection work sits at the intersection of nursing science, computer-vision research, 
    and equity in intimate-partner-violence care. Key impacts include:
    
    - Improved documentation for intimate partner violence cases
    - Better detection of injuries on darker skin tones
    - Enhanced legal evidence collection
    - Increased equity in healthcare
    - Support for forensic nursing practice
    
    This research has received national press coverage on NBC & NPR, highlighting its importance 
    for addressing healthcare disparities and improving care for victims of violence.
    """)
