import streamlit as st

def show():
    """
    Display the interview preparation page.
    """
    st.title("Interview Preparation")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Potential Questions", 
        "Your Questions to Ask", 
        "Key Talking Points",
        "12-Month Vision"
    ])
    
    with tab1:
        st.markdown("""
        ## Potential Interview Questions
        
        Prepare for these likely technical and project-related questions:
        """)
        
        with st.expander("Computer Vision & Deep Learning"):
            st.markdown("""
            1. **Q: How would you segment a faint bruise on dark skin under ALS illumination?**
               - Discuss transfer learning from dermatology models
               - Mention U-Net or Mask R-CNN architectures
               - Explain data augmentation for low-contrast features
               - Address multi-spectral input handling
            
            2. **Q: What architectures have you worked with for medical image segmentation?**
               - Highlight experience with CNN architectures (U-Net, FCN, etc.)
               - Discuss experience with attention mechanisms
               - Mention any experience with Vision Transformers
            
            3. **Q: How would you handle the class imbalance in bruise detection datasets?**
               - Weighted loss functions
               - Data augmentation and synthetic data
               - Focal loss and other techniques
               - Transfer learning approaches
            """)
        
        with st.expander("Mobile & Cloud Development"):
            st.markdown("""
            1. **Q: Inference on-device or in the cloud — which would you recommend?**
               - Discuss trade-offs (latency vs. computational power)
               - Mention model optimization techniques (quantization, pruning)
               - Address privacy considerations
               - Propose a hybrid approach for different use cases
            
            2. **Q: How would you ensure HIPAA compliance in a mobile health application?**
               - Data encryption at rest and in transit
               - Authentication and authorization
               - Audit logging
               - De-identification strategies
            
            3. **Q: Describe your experience with APIs and microservices.**
               - RESTful API design principles
               - API security considerations
               - Experience with specific frameworks (Flask, Express, etc.)
               - Testing and documentation approaches
            """)
        
        with st.expander("Database & Data Management"):
            st.markdown("""
            1. **Q: How would you design a database schema for storing bruise images and metadata?**
               - DICOM format considerations
               - Relational vs. NoSQL approaches
               - Metadata structure (patient data, image parameters, annotations)
               - Security and access control
            
            2. **Q: We capture HL7-FHIR bundles — outline your database schema.**
               - FHIR resource structure
               - JSON document storage
               - Indexing strategies
               - Integration with EHR systems
            """)
        
        with st.expander("Equity & Fairness"):
            st.markdown("""
            1. **Q: Describe a metric you would report to show equitable performance.**
               - Demographic parity difference
               - Equalized odds
               - True positive rate parity across skin tones
               - Fairness visualization approaches
            
            2. **Q: How would you validate that your models work equally well on all skin tones?**
               - Stratified testing across Fitzpatrick scale
               - Balanced datasets for training and validation
               - Collaboration with domain experts
               - Continuous monitoring and feedback loops
            """)
        
        with st.expander("Leadership & Collaboration"):
            st.markdown("""
            1. **Q: Tell us about a time you coordinated a heterogeneous team.**
               - Experience managing diverse skill sets
               - Communication strategies across disciplines
               - Project management approaches
               - Conflict resolution examples
            
            2. **Q: How would you approach supervising graduate students with varying technical backgrounds?**
               - Tailored mentorship approaches
               - Skill assessment and development planning
               - Creating learning opportunities
               - Balancing guidance with autonomy
            """)
    
    with tab2:
        st.markdown("""
        ## Questions to Ask the Committee
        
        Asking thoughtful questions shows your engagement and critical thinking:
        """)
        
        with st.expander("Project-Specific Questions"):
            st.markdown("""
            1. "How is the team prioritizing external validation sites to ensure geographic and demographic diversity?"
            
            2. "Are you envisioning FHIR-based data exchange with partner hospitals, or a purpose-built secure bucket?"
            
            3. "What metrics would make this platform courtroom-admissible, and how can the post-doc help shape that?"
            
            4. "How do you see this technology being integrated into existing clinical workflows?"
            
            5. "What has been the most challenging aspect of the bruise detection project so far?"
            """)
        
        with st.expander("Team and Collaboration Questions"):
            st.markdown("""
            1. "How does the team balance the clinical, technical, and research aspects of the project?"
            
            2. "What collaboration structures exist between nursing, computer science, and engineering departments?"
            
            3. "How do you ensure that technical development meets the needs of forensic nursing practice?"
            
            4. "What opportunities exist for cross-disciplinary mentorship?"
            
            5. "How are decisions made about technical approaches and priorities?"
            """)
        
        with st.expander("Career Development Questions"):
            st.markdown("""
            1. "How do you see the post-doc's work feeding into upcoming R01 or philanthropy renewals?"
            
            2. "What publication opportunities align with this position?"
            
            3. "What professional development resources are available for postdoctoral researchers?"
            
            4. "What are the potential career paths that could emerge from this postdoc position?"
            
            5. "How does the team approach conference presentations and outreach activities?"
            """)
    
    with tab3:
        st.markdown("""
        ## Key Talking Points
        
        Prepare to discuss these topics to demonstrate your fit for the position:
        """)
        
        with st.expander("Technical Expertise"):
            st.markdown("""
            - **Deep Learning for Medical Imaging**
              - Experience with medical image segmentation
              - Transfer learning from larger datasets
              - Model optimization for mobile deployment
            
            - **Computer Vision Pipelines**
              - Multi-spectral image processing
              - Low-contrast feature detection
              - Explainable AI techniques
            
            - **Mobile and Cloud Development**
              - API design and implementation
              - Secure data transmission
              - User interface for clinical settings
            """)
        
        with st.expander("Project Understanding"):
            st.markdown("""
            - **ALS Technology**
              - Understanding of alternate light sources (400-520nm wavelength)
              - Fluorescence principles with orange filters (>520nm)
              - Multi-spectral imaging techniques
            
            - **Forensic Nursing**
              - Awareness of documentation requirements
              - Legal admissibility considerations
              - Patient care integration
            
            - **Health Equity**
              - Fitzpatrick skin type variations
              - Bias in medical imaging
              - Fairness metrics and monitoring
            """)
        
        with st.expander("Interdisciplinary Collaboration"):
            st.markdown("""
            - **Team Science**
              - Experience working with healthcare professionals
              - Technical translation for non-technical audiences
              - Collaborative research approaches
            
            - **Mentorship**
              - Experience supervising students
              - Teaching technical skills
              - Project management
            
            - **Communication**
              - Explaining technical concepts to clinical staff
              - Documentation for different audiences
              - Presentation skills for diverse stakeholders
            """)
    
    with tab4:
        st.markdown("""
        ## 12-Month Vision
        
        Present a clear plan for your first year in the position:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Quarter 1 (Months 1-3)
            
            - Build data-version-controlled pipeline (DVC + Git)
            - Evaluate existing datasets and annotations
            - Design initial model architecture
            - Review literature on bruise detection and ALS imaging
            - Establish collaboration workflow with team members
            """)
            
            st.markdown("""
            ### Quarter 2 (Months 4-6)
            
            - Develop prototype models for bruise detection
            - Deploy containerized API on Mason's computing infrastructure
            - Begin integration with mobile application
            - Draft initial manuscript on methodology
            - Start supervising graduate students
            """)
        
        with col2:
            st.markdown("""
            ### Quarter 3 (Months 7-9)
            
            - Implement fairness dashboard for model monitoring
            - Conduct validation testing across skin tones
            - Optimize models for mobile deployment
            - Prepare conference abstract submissions
            - Expand annotation protocols with clinical team
            """)
            
            st.markdown("""
            ### Quarter 4 (Months 10-12)
            
            - Author IEEE JBHI or similar manuscript
            - Submit NIH K99/R00 or NSF supplement proposal
            - Develop full integration with clinical workflow
            - Evaluate system with forensic nursing partners
            - Present results at relevant conferences
            """)
        
        st.markdown("""
        ### Key Deliverables
        
        - **Technical**: Working prototype of bruise detection system with fairness metrics
        - **Research**: 1-2 peer-reviewed publications
        - **Funding**: Contribution to grant proposals
        - **Education**: Mentorship of graduate students
        - **Clinical**: Evaluation with forensic nursing partners
        """)
