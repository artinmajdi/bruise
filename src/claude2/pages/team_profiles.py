import streamlit as st
from components.visualizations import create_team_card

def show():
    """
    Display the team profiles page.
    """
    st.title("Research Team Profiles")
    
    st.markdown("""
    ## Core Research Team
    
    The EAS-ID project is led by an interdisciplinary team combining expertise in nursing science, 
    computer vision, and AI engineering. Understanding their backgrounds will help you connect with 
    them during the interview.
    """)
    
    create_team_card(
        "Dr. Katherine Scafide, PhD RN",
        "Principal Investigator, Associate Professor, School of Nursing",
        [
            "Forensic nursing leader",
            "American Academy of Forensic Sciences Fellow",
            "Former death investigator for Maryland's medical examiner's office",
            "Leading expert on using alternate light sources for bruise detection",
            "Published groundbreaking research showing ALS is 5× more effective than white light"
        ]
    )
    
    st.markdown("---")
    
    create_team_card(
        "Dr. Janusz Wojtusiak, PhD",
        "Co-PI, Associate Professor, Health Administration & Policy",
        [
            "Leads AI/data-science arm of the project",
            "Director of the Machine Learning and Inference Laboratory",
            "Leads NIH AIM-AHEAD equity grant activities",
            "Expert in machine learning for healthcare applications",
            "Focus on database architecture and bias metrics"
        ]
    )
    
    st.markdown("---")
    
    create_team_card(
        "Dr. David Lattanzi, PhD",
        "Co-PI, Associate Professor, Civil Engineering",
        [
            "Expert in imaging & light-physics",
            "Mobile-app prototype lead",
            "Background in computer vision pipelines",
            "Originally applied deep learning to structural engineering",
            "Now focusing on embedded/mobile deployment for bruise detection"
        ]
    )
    
    st.markdown("---")
    
    st.markdown("""
    ## Other Potential Search Committee Members
    
    These faculty may also be involved in the search committee based on their roles 
    and relevance to the project:
    """)
    
    create_team_card(
        "Dr. Terri Rebmann, PhD RN",
        "Divisional Dean, School of Nursing",
        [
            "New dean of the School of Nursing",
            "Disaster-preparedness scholar",
            "Oversees School of Nursing research vision",
            "Likely to be interested in how the postdoc fits into the School's strategic vision"
        ]
    )
    
    st.markdown("---")
    
    create_team_card(
        "Dr. Karen Trister Grace, PhD CNM",
        "Assistant Professor",
        [
            "Intimate-partner-violence (IPV) researcher",
            "Ensures project serves IPV & reproductive-health communities",
            "Interested in clinical applications and workflow integration"
        ]
    )
    
    st.markdown("""
    ## Interview Strategy
    
    When meeting with each team member:
    
    - **With Dr. Scafide**: Discuss clinical relevance, nursing pedagogy, and the injury-equity focus
    - **With Dr. Wojtusiak**: Be prepared for questions on model design, database architecture, and bias metrics
    - **With Dr. Lattanzi**: Focus on computer vision pipelines and mobile/embedded deployment
    - **With Dr. Rebmann**: Discuss how your work fits into the School's broader research vision
    - **With Dr. Grace**: Highlight understanding of intimate partner violence context and documentation needs
    """)
