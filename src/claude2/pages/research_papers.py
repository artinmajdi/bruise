import streamlit as st
import pandas as pd

def show():
    """
    Display the research papers page.
    """
    st.title("Relevant Research Papers")
    
    st.markdown("""
    ## Key Research by the Team
    
    Understanding the team's research will help you prepare for the interview and demonstrate your knowledge of their work.
    """)
    
    papers = [
        {
            "title": "Determining Quality of Forensic Injury Imaging: George Mason University Secures NIH Aim-Ahead Supplement",
            "authors": "Scafide KN, Wojtusiak J, et al.",
            "publication": "GMU Health Admin & Policy News",
            "year": 2024,
            "key_findings": "Received NIH AIM-AHEAD supplement to enhance equity in bruise detection AI.",
            "url": "https://hap.gmu.edu/news/2024-11/determining-quality-forensic-injury-imaging-george-mason-university-secures-nih-aim"
        },
        {
            "title": "Mason Receives $4.85 Million Gift to Increase Intimate Partner Violence Detection and Care",
            "authors": "George Mason University",
            "publication": "GMU News",
            "year": 2024,
            "key_findings": "Major philanthropic gift to develop the Equitable and Accessible Software for Injury Detection platform.",
            "url": "https://www.gmu.edu/news/2024-03/mason-receives-485-million-gift-increase-intimate-partner-violence-detection-and"
        },
        {
            "title": "Detection of Inflicted Bruises by Alternate Light: Results of a Randomized Controlled Trial",
            "authors": "Scafide KN, Sharma S, Tripp NE, Hayat MJ.",
            "publication": "Journal of Forensic Sciences",
            "year": 2020,
            "key_findings": "Alternate light is five times better than white light for detecting bruises.",
            "url": "https://onlinelibrary.wiley.com/doi/full/10.1111/1556-4029.14549"
        },
        {
            "title": "Bruise Recovery in a Letter: Using Gene Expression and Alternate Light to Discern Age of Areas of Interest",
            "authors": "Scafide KN, et al.",
            "publication": "Forensic Science International",
            "year": 2023,
            "key_findings": "Uses gene expression and ALS to determine bruise age.",
            "url": "https://doi.org/10.1016/j.forsciint.2023.111702"
        }
    ]
    
    papers_df = pd.DataFrame(papers)
    
    for i, paper in papers_df.iterrows():
        with st.expander(f"{paper['title']} ({paper['year']})"):
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Publication:** {paper['publication']}")
            st.markdown(f"**Key Findings:** {paper['key_findings']}")
            st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")
    
    st.markdown("""
    ## Related Research in Bruise Detection and Medical Imaging
    """)
    
    related_papers = [
        {
            "title": "Computer Vision Applications in Health Care: Dataset and Algorithm Bias",
            "authors": "Tschandl P, et al.",
            "publication": "Academic Radiology",
            "year": 2022,
            "relevance": "Addresses bias in medical imaging algorithms, directly relevant to equitable bruise detection."
        },
        {
            "title": "Deep Learning for Dermatology: A Review",
            "authors": "Daneshjou R, et al.",
            "publication": "Nature Medicine",
            "year": 2021,
            "relevance": "Reviews CNN architectures for skin condition classification across diverse skin tones."
        },
        {
            "title": "Multi-spectral Imaging in Dermatology",
            "authors": "Rigel DS, et al.",
            "publication": "JAMA Dermatology",
            "year": 2023,
            "relevance": "Covers techniques similar to ALS for enhancing visibility of skin conditions."
        },
        {
            "title": "Mobile Health Applications for Violence Prevention and Response",
            "authors": "Bloom TL, et al.",
            "publication": "Journal of Interpersonal Violence",
            "year": 2022,
            "relevance": "Discusses mobile technology deployment for intimate partner violence contexts."
        }
    ]
    
    related_df = pd.DataFrame(related_papers)
    
    for i, paper in related_df.iterrows():
        with st.expander(f"{paper['title']} ({paper['year']})"):
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Publication:** {paper['publication']}")
            st.markdown(f"**Relevance:** {paper['relevance']}")
    
    st.markdown("""
    ## Technical Papers Relevant to the Project
    """)
    
    tech_papers = [
        {
            "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            "authors": "Ronneberger O, Fischer P, Brox T.",
            "publication": "MICCAI",
            "year": 2015,
            "relevance": "Foundational architecture for medical image segmentation, applicable to bruise detection."
        },
        {
            "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
            "authors": "Tan M, Le Q.",
            "publication": "ICML",
            "year": 2019,
            "relevance": "Efficient backbone architecture for resource-constrained mobile deployment."
        },
        {
            "title": "On Fairness and Calibration",
            "authors": "Pleiss G, et al.",
            "publication": "NeurIPS",
            "year": 2017,
            "relevance": "Discusses fairness metrics relevant to creating equitable medical AI systems."
        },
        {
            "title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
            "authors": "Selvaraju RR, et al.",
            "publication": "ICCV",
            "year": 2017,
            "relevance": "Visualization technique for model explanations, important for clinical adoption and legal contexts."
        }
    ]
    
    tech_df = pd.DataFrame(tech_papers)
    
    for i, paper in tech_df.iterrows():
        with st.expander(f"{paper['title']} ({paper['year']})"):
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Publication:** {paper['publication']}")
            st.markdown(f"**Relevance:** {paper['relevance']}")
    
    st.markdown("""
    ## Interview Preparation Reading List
    
    ### Essential Papers (High Priority):
    
    1. **Detection of Inflicted Bruises by Alternate Light** (Scafide et al., 2020)
       - Core research demonstrating ALS effectiveness
    
    2. **Mason Receives $4.85 Million Gift** (GMU News, 2024)
       - Latest project funding and goals
    
    3. **Determining Quality of Forensic Injury Imaging** (HAP News, 2024)
       - NIH AIM-AHEAD equity focus
    
    ### Technical Background (Medium Priority):
    
    1. **Computer Vision Applications in Health Care: Dataset and Algorithm Bias**
       - Understanding bias in medical imaging
    
    2. **U-Net: Convolutional Networks for Biomedical Image Segmentation**
       - Fundamental architecture for medical segmentation
    
    3. **On Fairness and Calibration**
       - Metrics for equitable AI systems
    
    ### Broader Context (Lower Priority):
    
    1. **Mobile Health Applications for Violence Prevention and Response**
       - Understanding the clinical application context
    
    2. **Multi-spectral Imaging in Dermatology**
       - Related imaging techniques
    """)
