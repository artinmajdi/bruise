import streamlit as st
import numpy as np
from components.visualizations import create_skill_radar
import matplotlib.pyplot as plt

def show():
    """
    Display the technical stack page.
    """
    st.title("Technical Requirements")
    
    st.markdown("""
    ## Required Technical Skills
    
    The postdoctoral position requires expertise in multiple technical domains:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create skill radar chart
        skills = [
            "Deep Learning", 
            "Computer Vision",
            "Database Design",
            "API Development",
            "Cloud Computing",
            "Mobile Development"
        ]
        
        values = [5, 5, 4, 4, 3, 3]  # Importance on scale of 1-5
        
        radar_fig = create_skill_radar(skills, values, "Key Technical Skills")
        st.pyplot(radar_fig)
    
    with col2:
        st.markdown("""
        ### Programming Languages:
        - **Python**: Deep learning, computer vision, server-side
        - **JavaScript**: Web and mobile interfaces
        
        ### Libraries & Frameworks:
        - **Deep Learning**: TensorFlow, PyTorch
        - **Computer Vision**: OpenCV, scikit-image
        - **Mobile**: React Native, Flutter
        - **Cloud**: AWS, Azure, Google Cloud
        
        ### Knowledge Areas:
        - Deep neural networks and computer vision
        - Database systems (SQL and NoSQL)
        - Server-based and cloud environments
        - Computer security and patient privacy
        - Mobile software and hardware
        """)
    
    st.markdown("""
    ## Technical Architecture
    
    The EAS-ID platform consists of several integrated components:
    """)
    
    # Create simple architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    
    # Component boxes
    components = [
        {"name": "Mobile App", "x": 0.1, "y": 0.7, "width": 0.25, "height": 0.2},
        {"name": "Cloud API", "x": 0.5, "y": 0.7, "width": 0.25, "height": 0.2},
        {"name": "Deep Learning Models", "x": 0.5, "y": 0.3, "width": 0.25, "height": 0.2},
        {"name": "Database", "x": 0.85, "y": 0.5, "width": 0.25, "height": 0.2}
    ]
    
    # Draw boxes
    for component in components:
        rect = plt.Rectangle(
            (component["x"], component["y"]), 
            component["width"], 
            component["height"],
            fill=True,
            alpha=0.1,
            edgecolor='black',
            facecolor='blue'
        )
        ax.add_patch(rect)
        ax.text(
            component["x"] + component["width"]/2,
            component["y"] + component["height"]/2,
            component["name"],
            ha='center',
            va='center'
        )
    
    # Add arrows
    arrows = [
        {"start": (0.35, 0.8), "end": (0.5, 0.8), "label": "Images"},
        {"start": (0.5, 0.7), "end": (0.5, 0.5), "label": "API Call"},
        {"start": (0.6, 0.5), "end": (0.85, 0.6), "label": "Store Results"},
        {"start": (0.6, 0.3), "end": (0.85, 0.5), "label": "Query Data"}
    ]
    
    for arrow in arrows:
        ax.annotate(
            "",
            xy=arrow["end"],
            xytext=arrow["start"],
            arrowprops=dict(arrowstyle="->", color="black")
        )
        # Add label
        midpoint = ((arrow["start"][0] + arrow["end"][0])/2, (arrow["start"][1] + arrow["end"][1])/2)
        ax.text(
            midpoint[0],
            midpoint[1],
            arrow["label"],
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Mobile Application:
        
        - Captures multi-spectral images (white light & ALS)
        - Provides user interface for forensic nurses
        - Handles secure data transmission
        - Implements local caching and offline mode
        - Cross-platform (iOS & Android)
        """)
        
        st.markdown("""
        ### Cloud Infrastructure:
        
        - RESTful APIs for data exchange
        - Secure image repository
        - Microservices architecture
        - HIPAA-compliant data storage
        - Scalable processing for model training
        """)
    
    with col2:
        st.markdown("""
        ### AI Components:
        
        - Deep learning models for bruise detection
        - Multi-spectral image processing
        - Model explainability for legal contexts
        - Fairness metrics across skin tones
        - Continuous model improvement pipeline
        """)
        
        st.markdown("""
        ### Security & Privacy:
        
        - End-to-end encryption
        - De-identification protocols
        - HIPAA and 21 CFR Part 11 compliance
        - Audit logging
        - Role-based access control
        """)
    
    st.markdown("""
    ## Current Technical Challenges
    
    Being prepared to discuss these challenges will demonstrate your understanding of the project:
    """)
    
    with st.expander("Data Acquisition & Labeling"):
        st.markdown("""
        - **Limited Diverse Training Data**: Collecting sufficient bruise images across all Fitzpatrick skin types
        - **Annotation Complexity**: Subtle bruises require expert annotation
        - **Privacy Concerns**: Working with sensitive medical images
        - **Data Augmentation**: Need for synthetic data generation techniques
        - **Class Imbalance**: Addressing the imbalance across skin tones and bruise types
        """)
    
    with st.expander("Computer Vision Challenges"):
        st.markdown("""
        - **Low-Contrast Detection**: Identifying bruises with minimal contrast from surrounding skin
        - **Multi-Spectral Processing**: Handling multiple image channels (white light + ALS)
        - **Lighting Variation**: Accounting for different lighting conditions
        - **Model Generalization**: Ensuring consistent performance across all skin tones
        - **Feature Extraction**: Identifying subtle bruise features in various skin tones
        """)
    
    with st.expander("Mobile Deployment"):
        st.markdown("""
        - **On-Device vs. Cloud**: Balancing local processing with cloud capabilities
        - **Camera Calibration**: Ensuring consistent image capture
        - **User Interface Design**: Creating intuitive interfaces for clinical settings
        - **Battery & Performance**: Optimizing for mobile hardware constraints
        - **Offline Capability**: Functioning in low-connectivity environments
        """)
    
    with st.expander("Clinical Integration"):
        st.markdown("""
        - **Workflow Integration**: Fitting into existing clinical documentation processes
        - **EHR Compatibility**: Connecting with electronic health record systems
        - **Legal Admissibility**: Meeting requirements for evidence in legal proceedings
        - **User Training**: Educating forensic nurses on proper use
        - **Result Interpretation**: Presenting findings in an actionable format
        """)
