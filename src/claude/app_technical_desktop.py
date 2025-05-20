import stat
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add the core directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claude1.core.data_module import DatabaseSchema, FHIRDataModel, DataPipeline, create_synthetic_bruise_data, plot_synthetic_data_distribution, create_synthetic_image
from claude1.core.deployment_module import DeploymentComparison, display_deployment_options, display_scalability_info
from claude1.core.fairness_module import FairnessMetrics, display_fairness_metrics, display_bias_mitigation_techniques
from claude1.core.leadership_module import TeamManagement
from claude1.core.vision_module import BruiseDetectionModel, preprocess_image, apply_als_filter, display_problem_statement, display_solution_overview

st.set_page_config(
    page_title="Bruise Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

class Utils:

    PAGES_LIST = ["Project Overview", "Vision Module", "Data Module", "Deployment Options", "Fairness Analysis", "Team Management", "Interview Preparation"]

    @staticmethod
    def setup_page():

        st.session_state.bruise_model = BruiseDetectionModel()

        st.markdown(
            """
            <style>
            .main .block-container {
                padding-top: 2rem;
            }
            .sidebar .sidebar-content {
                padding-top: 1rem;
            }
            .dashboard-title {
                text-align: center;
                font-size: 2.5rem;
                margin-bottom: 2rem;
            }
            .section-header {
                margin-top: 1.5rem;
                margin-bottom: 1rem;
                padding-left: 0.5rem;
                border-left: 5px solid #FF4B4B;
            }
            .metric-container {
                background-color: #f0f2f6;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True, )

        st.markdown("<h1 class='dashboard-title'>Bruise Detection System</h1>", unsafe_allow_html=True)

        st.sidebar.title("Navigation")

    @staticmethod
    def interview_prep():
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

    @staticmethod
    def render_deployment_page():
        """
        Renders the Deployment Strategy page.
        Uses functions from deployment_module and adds interactive elements.
        """
        st.header("🚀 Deployment Strategy")
        st.markdown(
            """
            Considerations for deploying the Bruise Detection AI model.
            This includes potential platforms, scalability, and monitoring.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True): # Use border for visual grouping
                st.subheader("Deployment Options")
                display_deployment_options()

        with col2:
            with st.container(border=True):
                st.subheader("Scalability & Monitoring")
                display_scalability_info()

        st.divider()
        st.subheader("Interactive Mock Deployment Scenario Planner")
        st.markdown("Select options below to see a simplified, illustrative deployment plan.")

        cloud_provider = st.selectbox(
            "Select Cloud Provider:",
            ["Amazon Web Services (AWS)", "Google Cloud Platform (GCP)", "Microsoft Azure"],
            index=0,
            help="Choose the cloud platform for the mock deployment."
        )
        deployment_target = st.selectbox(
            "Select Deployment Target:",
            ["Mobile App Backend API", "Web Application API", "Edge Device (e.g., specialized medical scanner)"],
            index=0,
            help="Choose the target environment for the model."
        )

        real_time_requirement = st.checkbox("Real-time Processing Required?", value=True, help="Does the application need immediate results?")

        if st.button("Generate Mock Deployment Plan", help="Creates a sample plan based on your selections."):
            st.markdown(f"#### Mock Deployment Plan for: **{deployment_target} on {cloud_provider}**")

            plan_details = []
            plan_details.append(f"- **Target Platform:** {deployment_target}")
            plan_details.append(f"- **Cloud Provider:** {cloud_provider}")

            if "AWS" in cloud_provider:
                compute = "AWS SageMaker Endpoints / AWS Lambda with ECR" if real_time_requirement else "AWS Batch / EC2"
                storage = "AWS S3"
                database = "AWS DynamoDB / RDS"
                monitoring = "AWS CloudWatch"
            elif "GCP" in cloud_provider:
                compute = "Google AI Platform Prediction / Cloud Functions" if real_time_requirement else "Google Kubernetes Engine / Compute Engine"
                storage = "Google Cloud Storage"
                database = "Google Cloud SQL / Firestore"
                monitoring = "Google Cloud Monitoring"
            else: # Azure
                compute = "Azure Machine Learning Endpoints / Azure Functions" if real_time_requirement else "Azure Kubernetes Service / Virtual Machines"
                storage = "Azure Blob Storage"
                database = "Azure SQL Database / Cosmos DB"
                monitoring = "Azure Monitor"

            plan_details.append(f"- **Compute Service (Illustrative):** {compute} for model serving.")
            plan_details.append(f"- **Data Storage:** {storage} for images and model artifacts.")
            plan_details.append(f"- **Metadata/Results Database:** {database}.")
            plan_details.append(f"- **API Gateway:** Setup an API endpoint (e.g., Amazon API Gateway, Google Cloud Endpoints, Azure API Management) for secure access.")
            plan_details.append(f"- **Monitoring & Logging:** Implement {monitoring} for performance, errors, and model drift detection.")
            plan_details.append(f"- **CI/CD Pipeline:** Establish using tools like AWS CodePipeline, Google Cloud Build, or Azure DevOps for automated training, testing, and deployment.")

            if "Edge Device" in deployment_target:
                plan_details.append(f"- **Edge Specifics:** Model optimization (e.g., TensorFlow Lite, ONNX Runtime) for on-device inference. Secure OTA updates.")

            for detail in plan_details:
                st.markdown(detail)

            st.success("This is a simplified, illustrative plan. Actual deployment requires detailed architectural planning.")

    @staticmethod
    def overview():
        st.header("Project Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Problem Statement")
            display_problem_statement()

        with col2:
            st.subheader("Solution")
            display_solution_overview()

        st.markdown("---")
        st.subheader("System Architecture")
        st.image("https://mermaid.ink/img/pako:eNqNkk1PwzAMhv9KlBOIaZqAHZC2w06AwAUJIdQa14u2JiWJQBP7727b0XVsEtMltefx-9iJj7TQBDSjDe0dCMJnQE8j-FFgLexb9KzaEZYGV1-LxRKtNQN54S-C5DsUIBc_BAnNyUMhMecBn5YrYLTQ7R3tnZ3UuCPTVi0K7ItEXZfQWJR0AXdLrQRolB72CKtbKDH6G9Nw3u3eD2Eb-J1qKaEHQR5z7XB1PbFxHq2kCR7vT6g3R2ygK1XP9H_Lh6b3Q1rPjuBQLsfifKM6RqUNVnG2vKGMDTquxpC_E6S-RbMvRrXTKJTVBvnOETzfVxhHkEqDq_yS6nHJaCC80E2LRXQ5Zi7OvSNNjWGpkVVYRz-JNTVbtDDYmLlGo7c2RkqRVXCCkz90-0X_C7MjL74BBYTiZQ", width=700)

        st.markdown("---")
        st.subheader("Key Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            - **Multi-Spectral Imaging**: Combined analysis using white light and alternate light sources (ALS)
            - **Cross-Skin Tone Optimization**: Enhanced accuracy across Fitzpatrick skin types I-VI
            - **Bruise Age Estimation**: Color-based analysis to determine approximate bruise age
            - **FHIR Integration**: Standardized healthcare data format compatibility
            - **Mobile-First Design**: Optimized for field use in clinical and forensic settings
            """)

        with col2:
            st.markdown("""
            - **Privacy-Preserving Design**: Edge processing options to minimize data sharing
            - **Fairness Evaluation**: Comprehensive testing across demographic groups
            - **Explainable Results**: Visual overlays and confidence metrics for interpretability
            - **Multi-Platform Support**: Android, iOS, and web-based interfaces
            - **Clinical Workflow Integration**: Seamless integration with existing documentation systems
            """)

    @staticmethod
    def vision_module():
        st.header("Vision Module")

        st.subheader("Image Processing & Bruise Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Image Input")
            image_option = st.radio(
                "Select image source:",
                ["Upload Image", "Generate Synthetic Image"]
            )

            if image_option == "Upload Image":
                uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)
                else:
                    image = create_synthetic_image()
                    st.image(image, caption="Default Synthetic Image", width=300)
            else:
                skin_tone = st.slider("Skin tone background (lighter to darker)", 1, 6, 3)
                num_bruises = st.slider("Number of simulated bruises", 1, 3, 1)

                # Adjust background color based on skin tone (simple mapping)
                skin_colors = {
                    1: (245, 235, 220),  # Very light
                    2: (230, 220, 205),  # Light
                    3: (210, 195, 180),  # Medium light
                    4: (180, 160, 140),  # Medium
                    5: (150, 120, 100),  # Medium dark
                    6: (120, 90, 70)     # Dark
                }

                image = create_synthetic_image(background_color=skin_colors[skin_tone], num_shapes=num_bruises)
                st.image(image, caption=f"Synthetic Image (Fitzpatrick Type {skin_tone})", width=300)

        with col2:
            st.markdown("### Processing Options")
            light_source = st.selectbox(
                "Light source:",
                ["white", "als_415nm", "als_450nm"]
            )

            skin_tone_value = st.number_input("Fitzpatrick skin type (1-6):", min_value=1, max_value=6, value=3)

            if "image" in locals():
                # Convert PIL Image to numpy array for processing
                image_array = np.array(image)

                if light_source != "white":
                    # Apply ALS filter simulation
                    wavelength = int(light_source.split("_")[1].replace("nm", ""))
                    filter_color = "orange" if wavelength == 415 else "yellow"
                    processed_image = apply_als_filter(image_array, wavelength=wavelength, filter_color=filter_color)
                    st.image(processed_image, caption=f"Simulated {light_source} with {filter_color} filter", width=300)
                else:
                    # Apply regular preprocessing
                    processed_image = preprocess_image(image_array, light_source, skin_tone_value)
                    st.image(processed_image, caption="Preprocessed Image", width=300)

                # Run detection model
                if st.button("Detect Bruises"):
                    with st.spinner("Processing..."):
                        mask, confidence, metadata = st.session_state.bruise_model.detect_bruises(
                            image_array, light_source, skin_tone_value
                        )

                        # Create a color mask for visualization
                        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        color_mask[mask > 0] = [255, 0, 0]  # Red for detected bruises

                        # Blend with original image
                        alpha = 0.5
                        blended = (alpha * color_mask + (1-alpha) * image_array).astype(np.uint8)

                        st.image(blended, caption="Detection Results", width=300)

                        # Display detection results
                        st.markdown("### Detection Results")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence", f"{confidence:.2f}")
                        col2.metric("Processing Time", f"{metadata['processing_time_ms']} ms")
                        col3.metric("Bruise Area", f"{metadata['bruise_area_pixels']} px")

                        st.markdown("### Age Estimation")
                        age_estimate, age_confidence, color_features = st.session_state.bruise_model.estimate_bruise_age(
                            image_array, mask, skin_tone_value
                        )

                        if age_estimate is not None:
                            st.markdown(f"Estimated bruise age: **{age_estimate} hours** (Confidence: {age_confidence:.2f})")

                            # Show color features in a radar chart
                            rgb_means = color_features["mean_rgb"]
                            color_ratios = [
                                color_features["r_g_ratio"],
                                color_features["r_b_ratio"],
                                color_features["g_b_ratio"]
                            ]

                            fig = go.Figure()

                            fig.add_trace(go.Scatterpolar(
                                r=[rgb_means[0]/255, rgb_means[1]/255, rgb_means[2]/255,
                                color_ratios[0]/2, color_ratios[1]/2, color_ratios[2]/2],
                                theta=['Red', 'Green', 'Blue', 'R/G Ratio', 'R/B Ratio', 'G/B Ratio'],
                                fill='toself',
                                name='Color Features'
                            ))

                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                showlegend=False
                            )

                            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def data_module():
        st.header("Data Module")

        tab1, tab2, tab3 = st.tabs(["Database Schema", "Sample Data Analysis", "FHIR Integration"])

        with tab1:
            st.subheader("FHIR-Compliant Database Schema")

            # Initialize the DatabaseSchema class
            schema = DatabaseSchema()

            # Display schema overview
            st.markdown("### Core Collections")

            # Create a dropdown for selecting collections
            selected_collection = st.selectbox(
                "Select a collection to view its schema:",
                list(schema.schema["core_collections"].keys())
            )

            # Display the selected collection schema
            collection_schema = schema.schema["core_collections"][selected_collection]

            st.markdown(f"**Description**: {collection_schema['description']}")
            st.markdown(f"**FHIR Resource Type**: {collection_schema['fhir_resource_type']}")

            # Display fields in a more readable format
            st.markdown("### Fields")

            fields_df = pd.DataFrame(
                [(field, details.get("type", ""), "Yes" if field in collection_schema.get("required_fields", []) else "No")
                for field, details in collection_schema["fields"].items()],
                columns=["Field Name", "Type", "Required"]
            )

            st.dataframe(fields_df, use_container_width=True)

            # Display extensions if any
            if collection_schema.get("extension_fields"):
                st.markdown("### Extensions")
                for ext in collection_schema["extension_fields"]:
                    st.markdown(f"- **URL**: {ext['url']}")
                    for key, value in ext.items():
                        if key != "url":
                            st.markdown(f"  - **{key}**: {value}")

        with tab2:
            st.subheader("Bruise Data Analysis")

            # Use the synthetic data from session state
            df = create_synthetic_bruise_data(100)

            st.markdown("### Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            st.markdown("### Data Distribution")

            col1, col2 = st.columns(2)

            with col1:
                numeric_column = st.selectbox("Select numeric column for histogram:",
                                            [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])

                fig = plot_synthetic_data_distribution(df, numeric_column)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                categorical_column = st.selectbox("Select categorical column for bar chart:",
                                            [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])])

                category_counts = df[categorical_column].value_counts().reset_index()
                category_counts.columns = [categorical_column, 'Count']

                fig = px.bar(category_counts, x=categorical_column, y='Count',
                            title=f'Distribution of {categorical_column}')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Cross-Analysis")

            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("X-axis:", df.columns, index=2)  # bruise_age_days default

            with col2:
                y_axis = st.selectbox("Y-axis:", df.columns, index=4)  # bruise_area_cm2 default

            color_by = st.selectbox("Color by:", ["None"] + list(df.select_dtypes(include=['object']).columns))

            if color_by == "None":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                                title=f'{y_axis} vs {x_axis} by {color_by}')

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("FHIR Integration")

            # Initialize FHIR model
            fhir_model = FHIRDataModel()

            st.markdown("""
            The Bruise Detection System uses FHIR (Fast Healthcare Interoperability Resources)
            to ensure compatibility with existing healthcare systems. Below is an example of how
            bruise detection data is structured in FHIR format.
            """)

            # Sample patient data for example
            sample_patient = {
                "id": "patient123",
                "name": "Jane Smith",
                "gender": "female",
                "birthDate": "1985-06-15",
                "fitzpatrickSkinType": 4
            }

            # Create FHIR patient
            fhir_patient = fhir_model.create_patient(sample_patient)

            # Display FHIR resource
            st.json(fhir_patient)

            st.markdown("### FHIR Resource Pipeline")

            st.image("https://mermaid.ink/img/pako:eNqNks1uwjAMhF8lyjkgFdpSjpVgJw487MiJyrXRGpGfKklBReXdm6QUJKRdcorH4_HYSk5K14xkLIEyoGYfoM0EfhVYCacRHatGaFNj27qqNkhrRvLGPwXJVyRAXnwVpDm5PgdJuQ34NG2gjRLj1MnJjanxt7Np1VFAXyTpvoTOYknv0LXUKkCa0sMJoXsESdHfO83n0-VtCLvAb1RLCaMTZBSB0OnOzFh_iRbfcwRfrHu5q9-0mj6q8FPMfh3pLJl3MWr3FcqFrV_p1NKqoR3N-pfyoRl9Nba3I3iUzVmMwGLswyrF1Zq7mLTKGLXDH4bR_RqWH6r8lbOODopkJcxXhuDlvsYYQbKCnvJLKseyFNYvNL5vLv56vvkfUKBkLy5gUqcX", width=600)

            # Sample bruise documentation data
            st.markdown("### Bruise Documentation Bundle Example")

            sample_data = {
                "patient": sample_patient,
                "imagingStudy": {
                    "description": "Bruise documentation session",
                    "started": "2023-05-15T14:30:00",
                    "alsWavelength": 415,
                    "alsFilter": "orange"
                },
                "media": [
                    {
                        "contentUrl": "https://example.com/images/bruise1.jpg",
                        "contentType": "image/jpeg",
                        "bodySite": "Right forearm",
                        "height": 1024,
                        "width": 768
                    }
                ],
                "observations": [
                    {
                        "bodySite": "Right forearm",
                        "size": 4.5,
                        "color": "Purple-blue",
                        "pattern": "Diffuse",
                        "estimatedAge": 48,
                        "ageConfidence": 0.75
                    }
                ],
                "diagnosticReport": {
                    "conclusion": "Bruising consistent with blunt force trauma, approximately 2 days old"
                }
            }

            # Create bundle
            bundle = fhir_model.create_bruise_documentation_bundle(sample_data)

            # Create a dropdown to show specific parts of the bundle
            bundle_section = st.selectbox(
                "View bundle section:",
                ["Overview", "Patient", "ImagingStudy", "Media", "Observation", "DiagnosticReport"]
            )

            if bundle_section == "Overview":
                st.markdown(f"**Bundle Type**: {bundle['type']}")
                st.markdown(f"**Resources**: {len(bundle['entry'])} total resources")

                # Display resource types in the bundle
                resource_types = [entry['resource']['resourceType'] for entry in bundle['entry']]
                resource_counts = pd.Series(resource_types).value_counts().reset_index()
                resource_counts.columns = ['ResourceType', 'Count']

                st.dataframe(resource_counts)
            else:
                # Find and display the requested resource
                for entry in bundle['entry']:
                    if entry['resource']['resourceType'] == bundle_section:
                        st.json(entry['resource'])
                        break

    @staticmethod
    def fairness_analysis():
        st.header("Fairness Analysis")

        display_fairness_metrics()

        st.markdown("---")

        # Create simulated fairness evaluation data
        np.random.seed(42)

        # Create sample data across skin tones
        skin_tones = []
        predictions = []
        scores = []
        labels = []

        # For each skin tone 1-6
        for skin_tone in range(1, 7):
            n_samples = 100  # 100 samples per skin tone

            # Generate skin tone values
            skin_tones.extend([skin_tone] * n_samples)

            # Generate true labels (slightly imbalanced)
            if skin_tone <= 3:
                pos_rate = 0.4  # 40% positive rate for lighter skin
            else:
                pos_rate = 0.4 - (skin_tone - 3) * 0.05  # Decreasing positive rate for darker skin

            sample_labels = np.random.binomial(1, pos_rate, n_samples)
            labels.extend(sample_labels)

            # Generate prediction scores (with skin tone dependent bias)
            sample_scores = []
            for label in sample_labels:
                if label == 1:
                    # For positive cases, high scores but with skin-tone dependent decrease
                    base_score = np.random.beta(8, 2)  # High scores centered around 0.8
                    adjustment = (skin_tone - 1) * 0.03  # Gradually decrease by skin tone
                    score = max(0, min(1, base_score - adjustment))
                else:
                    # For negative cases, low scores but with more false positives for lighter skin
                    base_score = np.random.beta(2, 8)  # Low scores centered around 0.2
                    adjustment = (6 - skin_tone) * 0.02  # More false positives for lighter skin
                    score = max(0, min(1, base_score + adjustment))

                sample_scores.append(score)

            scores.extend(sample_scores)

            # Generate binary predictions using fixed threshold
            sample_preds = [1 if s >= 0.5 else 0 for s in sample_scores]
            predictions.extend(sample_preds)

        # Convert to numpy arrays
        skin_tones = np.array(skin_tones)
        predictions = np.array(predictions)
        scores = np.array(scores)
        labels = np.array(labels)

        # Run fairness analysis
        fairness = FairnessMetrics()

        # Group skin tones into three categories
        skin_tone_groups = np.zeros_like(skin_tones, dtype=object)
        skin_tone_groups[(skin_tones == 1) | (skin_tones == 2)] = "Type I-II"
        skin_tone_groups[(skin_tones == 3) | (skin_tones == 4)] = "Type III-IV"
        skin_tone_groups[(skin_tones == 5) | (skin_tones == 6)] = "Type V-VI"

        # Calculate fairness metrics
        metrics, group_performance = fairness.evaluate_all_metrics(
            predictions, labels, scores, skin_tone_groups
        )

        # Generate fairness report
        report = fairness.fairness_report(metrics, group_performance)

        # Display metrics overview
        st.subheader("Fairness Metrics Overview")

        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values()),
            'Assessment': [report['assessment'].get(metric, "Unknown") for metric in metrics.keys()],
            'Threshold': [report['fairness_thresholds'].get(metric, "N/A") for metric in metrics.keys()]
        })

        def highlight_assessment(val):
            if val == "Passed":
                return 'background-color: #CCFFCC'
            elif val == "Failed":
                return 'background-color: #FFCCCC'
            else:
                return ''

        st.dataframe(metrics_df.style.map(highlight_assessment, subset=['Assessment']), use_container_width=True)


        # Display overall assessment
        st.markdown(f"### Overall Assessment: {report['overall']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Passed", report['passed_count'])
        col2.metric("Failed", report['failed_count'])
        col3.metric("Unknown", report['unknown_count'])

        # Display group performance
        st.subheader("Performance Across Skin Tone Groups")

        # Create performance comparison table
        performance_table = report['performance_table']

        # Visualize key metrics across groups
        metrics_to_plot = ['balanced_accuracy', 'true_positive_rate', 'false_positive_rate']

        for metric in metrics_to_plot:
            metric_data = []
            for group in performance_table[metric]:
                metric_data.append({
                    'Group': group,
                    'Value': performance_table[metric][group]
                })

            metric_df = pd.DataFrame(metric_data)

            fig = px.bar(
                metric_df,
                x='Group',
                y='Value',
                color='Group',
                title=f"{metric.replace('_', ' ').title()} by Skin Tone Group",
                text='Value'
            )

            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

            st.plotly_chart(fig, use_container_width=True)

        # Display detailed performance metrics
        st.subheader("Detailed Group Performance")

        # Create a nicer table of performance metrics
        group_metrics = []

        for group in group_performance:
            metrics = group_performance[group]
            group_metrics.append({
                'Group': group,
                'Positive Rate': f"{metrics['positive_rate']:.3f}",
                'True Positive Rate': f"{metrics['true_positive_rate']:.3f}",
                'False Positive Rate': f"{metrics['false_positive_rate']:.3f}",
                'Positive Predictive Value': f"{metrics['positive_predictive_value']:.3f}",
                'Balanced Accuracy': f"{metrics['balanced_accuracy']:.3f}",
                'Optimal Threshold': f"{metrics['optimal_threshold']:.2f}"
            })

        group_metrics_df = pd.DataFrame(group_metrics)
        st.dataframe(group_metrics_df, use_container_width=True)

        # Bias mitigation strategies
        st.markdown("---")
        st.subheader("Bias Mitigation Strategies")

        display_bias_mitigation_techniques()

    @staticmethod
    def team_management():
        st.header("Team Management")

        # Get team management from session state
        team = TeamManagement()

        tab1, tab2, tab3, tab4 = st.tabs(["Team Structure", "Project Milestones", "Communication", "Publication Plan"])

        with tab1:
            st.subheader("Team Composition")

            # Get team roles
            roles = team.get_team_roles()

            # Create a DataFrame for display
            roles_df = pd.DataFrame([
                {
                    'Role': role['role'],
                    'Discipline': role['discipline'],
                    'Background': role['background'],
                    'Key Responsibilities': ", ".join(role['responsibilities'][:2]) + "..."
                }
                for role in roles
            ])

            st.dataframe(roles_df, use_container_width=True)

            # Show team composition by discipline
            composition = team.get_team_composition_summary()

            composition_df = pd.DataFrame({
                'Discipline': list(composition.keys()),
                'Count': list(composition.values())
            })

            fig = px.pie(
                composition_df,
                values='Count',
                names='Discipline',
                title="Team Composition by Discipline"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show team challenges
            st.subheader("Team Challenges & Mitigation")

            # Display challenges in expandable sections
            for i, challenge in enumerate(team.team_challenges):
                with st.expander(f"{challenge['challenge']} (Impact: {challenge['impact']})"):
                    st.markdown(f"**Description**: {challenge['description']}")

                    st.markdown("**Mitigation Strategies**:")
                    for strategy in challenge['strategies']:
                        st.markdown(f"- {strategy}")

                    st.markdown("**Metrics to Monitor**:")
                    for metric in challenge['metrics']:
                        st.markdown(f"- {metric}")

        with tab2:
            st.subheader("Project Milestones")

            # Get milestone data
            milestones = team.project_milestones

            # Create a DataFrame for the Gantt chart
            gantt_data = []

            milestone_names = []
            for milestone in milestones:
                milestone_date = pd.to_datetime(milestone['date'])

                # Estimate end date based on dependencies and next milestones
                # For simplicity, we'll use a fixed duration
                end_date = milestone_date + pd.Timedelta(days=30)

                gantt_data.append({
                    'Task': milestone['name'],
                    'Start': milestone_date,
                    'Finish': end_date,
                    'Status': milestone['status']
                })

                milestone_names.append(milestone['name'])

            gantt_df = pd.DataFrame(gantt_data)

            # Plot milestones as a Gantt chart
            fig = px.timeline(
                gantt_df,
                x_start='Start',
                x_end='Finish',
                y='Task',
                color='Status',
                color_discrete_map={
                    'Completed': 'green',
                    'In Progress': 'blue',
                    'Planned': 'gray'
                },
                title="Project Milestones Timeline"
            )

            fig.update_yaxes(categoryorder='array', categoryarray=milestone_names)

            st.plotly_chart(fig, use_container_width=True)

            # Show milestone dependencies
            st.subheader("Milestone Dependencies")

            dependencies = team.get_milestone_dependencies()

            # Create a mermaid chart for dependencies
            nodes = [f"{i}[{name}]" for i, name in enumerate(milestone_names)]
            links = []

            # Create a mapping from milestone name to index
            name_to_index = {name: i for i, name in enumerate(milestone_names)}

            for dep in dependencies:
                from_idx = name_to_index.get(dep['from'])
                to_idx = name_to_index.get(dep['to'])

                if from_idx is not None and to_idx is not None:
                    links.append(f"{from_idx} --> {to_idx}")

            mermaid_chart = "graph TD;\n" + ";\n".join(nodes) + ";\n" + ";\n".join(links)

            # Display the mermaid chart with st.graphviz_chart or through an iframe
            st.markdown("```mermaid\n " + mermaid_chart + " \n```")

            # Show upcoming milestones
            st.subheader("Upcoming Milestones")

            upcoming = team.get_upcoming_milestones(90)

            if upcoming:
                upcoming_df = pd.DataFrame([
                    {
                        'Milestone': m['name'],
                        'Date': m['date'],
                        'Days Until': m['days_until'],
                        'Description': m['description']
                    }
                    for m in upcoming
                ])

                st.dataframe(upcoming_df.sort_values('Days Until'), use_container_width=True)
            else:
                st.info("No upcoming milestones in the next 90 days.")

        with tab3:
            st.subheader("Team Communication")

            # Show meeting schedule
            st.markdown("### Meeting Schedule")

            meetings = team.meeting_schedule

            # Create a DataFrame for meetings
            meetings_df = pd.DataFrame([
                {
                    'Meeting': m['name'],
                    'Date': m['date'],
                    'Time': m['time'],
                    'Participants': m['participants'],
                    'Frequency': m['frequency']
                }
                for m in meetings
            ])

            st.dataframe(meetings_df.sort_values('Date'), use_container_width=True)

            # Show communication matrix
            st.markdown("### Communication Matrix")

            communication_matrix = team.generate_communication_matrix()

            # Create a heatmap of communication frequency
            communication_freq = {}

            for entry in communication_matrix:
                from_group = entry['from_group']
                to_group = entry['to_group']

                if from_group not in communication_freq:
                    communication_freq[from_group] = {}

                # Map frequency to numeric value
                if entry['frequency'] == "Daily":
                    freq_value = 5
                elif entry['frequency'] == "Weekly":
                    freq_value = 4
                elif entry['frequency'] == "Bi-weekly":
                    freq_value = 3
                elif entry['frequency'] == "Monthly":
                    freq_value = 2
                else:
                    freq_value = 1

                communication_freq[from_group][to_group] = freq_value

            # Convert to DataFrame for heatmap
            team_groups = list(team.team_structure.keys())

            comm_matrix_data = []

            for from_group in team_groups:
                for to_group in team_groups:
                    value = communication_freq.get(from_group, {}).get(to_group, 0)
                    comm_matrix_data.append({
                        'From': from_group,
                        'To': to_group,
                        'Frequency': value
                    })

            comm_df = pd.DataFrame(comm_matrix_data)
            comm_pivot = pd.pivot_table(
                comm_df,
                values='Frequency',
                index='From',
                columns='To'
            )

            # Convert frequency to labels
            freq_labels = {
                5: "Daily",
                4: "Weekly",
                3: "Bi-weekly",
                2: "Monthly",
                1: "As needed",
                0: "None"
            }

            # Create the heatmap with custom text
            fig = go.Figure(data=go.Heatmap(
                z=comm_pivot.values,
                x=comm_pivot.columns,
                y=comm_pivot.index,
                text=comm_pivot.map(lambda x: freq_labels.get(x, "")),  # Custom text
                texttemplate="%{text}",  # Use the text we provided
                textfont={"size": 12},
                colorscale=[[0, 'white'], [0.3, 'yellow'], [0.7, 'orange'], [1, 'red']],
                showscale=True
            ))

            fig.update_layout(
                title="Team Communication Frequency",
                xaxis_title="",
                yaxis_title="",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show onboarding process
            st.markdown("### Team Onboarding Process")

            onboarding = team.get_onboarding_process()

            # Display onboarding timeline
            with st.expander("View Onboarding Timeline"):
                st.markdown("#### Pre-Arrival")
                for step in onboarding['pre_arrival']:
                    st.markdown(f"- **{step['step']}** ({step['timeline']}): {step['description']}")

                st.markdown("#### First Day")
                for step in onboarding['first_day']:
                    st.markdown(f"- **{step['step']}** ({step['timeline']}): {step['description']}")

                st.markdown("#### First Week")
                for step in onboarding['first_week']:
                    st.markdown(f"- **{step['step']}** ({step['timeline']}): {step['description']}")

                st.markdown("#### First Month")
                for step in onboarding['first_month']:
                    st.markdown(f"- **{step['step']}** ({step['timeline']}): {step['description']}")

        with tab4:
            st.subheader("Publication Plan")

            # Get publication plan
            pub_plan = team.get_publication_plan()

            # Show publication strategy
            st.markdown("### Publication Strategy")

            st.markdown("#### Focus Areas")
            for area in pub_plan['strategy']['focus_areas']:
                st.markdown(f"- {area}")

            # Show planned publications
            st.markdown("### Planned Publications")

            # Create a DataFrame for publications
            pubs_df = pd.DataFrame([
                {
                    'Title': pub['title'],
                    'Venue': pub['target_venue'],
                    'Submission Date': pub['planned_submission'],
                    'Lead Author': pub['lead_author'],
                    'Status': pub['status']
                }
                for pub in pub_plan['planned_publications']
            ])

            st.dataframe(pubs_df.sort_values('Submission Date'), use_container_width=True)

            # Show publication timeline
            st.markdown("### Publication Timeline")

            # Create a timeline for publications
            pub_timeline = []

            for pub in pub_plan['planned_publications']:
                date = pd.to_datetime(pub['planned_submission'])

                pub_timeline.append({
                    'Title': pub['title'],
                    'Date': date,
                    'Type': 'Journal' if 'Journal' in pub['target_venue'] else 'Conference',
                    'Lead': pub['lead_author']
                })

            # Add conference deadlines
            for conf in pub_plan['conference_targets']:
                date = pd.to_datetime(conf['deadline'])

                pub_timeline.append({
                    'Title': f"Deadline: {conf['name']}",
                    'Date': date,
                    'Type': 'Deadline',
                    'Lead': 'N/A'
                })

            # Create DataFrame for timeline
            timeline_df = pd.DataFrame(pub_timeline)
            timeline_df = timeline_df.sort_values('Date')

            # Create timeline plot
            fig = px.timeline(
                timeline_df,
                x_start='Date',
                x_end='Date',  # For single-day events, start and end are the same
                y='Title',
                color='Type',
                hover_data=['Lead'],
                title="Publication Timeline"
            )

            # For better visualization of single-day events
            fig.update_traces(width=0.5)  # Make the bars thinner for single-day events
            fig.update_layout(height=600)

            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def deployment_options():
        st.header("Deployment Options")

        # Get deployment comparison from session state
        deployment = DeploymentComparison()

        st.markdown("""
        The Bruise Detection System can be deployed in several configurations, each with
        different trade-offs in terms of privacy, performance, and connectivity requirements.
        Explore the comparison below to understand which option might be best for your needs.
        """)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Comparison Matrix", "Use Case Analysis", "Cost Analysis", "Bandwidth Requirements", "Strategy"])

        with tab1:
            st.subheader("Deployment Options Comparison")

            # Get comparison table
            comparison_df = deployment.get_comparison_table()

            # Display as a heatmap
            fig = px.imshow(
                comparison_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Deployment Options Comparison (Scale: 1-10)",
                labels=dict(x="Deployment Option", y="Metric", color="Score")
            )
            fig.update_layout(height=600)

            st.plotly_chart(fig, use_container_width=True)

            # Show radar chart comparison
            st.subheader("Visual Comparison")

            radar_data, metrics = deployment.get_radar_data()

            fig = go.Figure()

            for option in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=radar_data[option],
                    theta=metrics,
                    fill='toself',
                    name=option
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Use Case Analysis")

            # Use case input
            use_case = st.text_area(
                "Describe your use case scenario:",
                "Forensic documentation in a rural medical clinic with limited internet connectivity."
            )

            # Custom weights
            st.markdown("### Customize Importance Weights")
            st.markdown("Adjust the importance of each factor for your specific use case (0.0-1.0)")

            col1, col2 = st.columns(2)

            with col1:
                latency_weight = st.slider("Latency", 0.0, 1.0, 0.7, 0.1)
                privacy_weight = st.slider("Privacy", 0.0, 1.0, 0.8, 0.1)
                bandwidth_weight = st.slider("Bandwidth", 0.0, 1.0, 0.9, 0.1)
                compute_weight = st.slider("Compute", 0.0, 1.0, 0.5, 0.1)
                battery_weight = st.slider("Battery", 0.0, 1.0, 0.7, 0.1)

            with col2:
                offline_weight = st.slider("Offline Capability", 0.0, 1.0, 0.9, 0.1)
                model_size_weight = st.slider("Model Size", 0.0, 1.0, 0.5, 0.1)
                update_freq_weight = st.slider("Update Frequency", 0.0, 1.0, 0.4, 0.1)
                security_weight = st.slider("Security", 0.0, 1.0, 0.8, 0.1)
                complexity_weight = st.slider("Implementation Complexity", 0.0, 1.0, 0.6, 0.1)

            # Create weights dictionary
            weights = {
                "latency": latency_weight,
                "privacy": privacy_weight,
                "bandwidth": bandwidth_weight,
                "compute": compute_weight,
                "battery": battery_weight,
                "offline_capability": offline_weight,
                "model_size": model_size_weight,
                "update_frequency": update_freq_weight,
                "security": security_weight,
                "implementation_complexity": complexity_weight
            }

            # Analyze use case
            if st.button("Analyze Use Case"):
                analysis = deployment.analyze_use_case(use_case, weights)

                st.markdown(f"### Recommendation: {analysis['best_option'].title()}")

                # Display scores
                scores = pd.DataFrame({
                    'Deployment Option': list(analysis['scores'].keys()),
                    'Score': list(analysis['scores'].values()),
                    'Relative Score': [f"{analysis['score_ratio'][opt]:.2f}" for opt in analysis['scores'].keys()]
                })

                fig = px.bar(
                    scores,
                    x='Deployment Option',
                    y='Score',
                    color='Deployment Option',
                    title="Deployment Option Scores",
                    text='Relative Score'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display strengths and weaknesses
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### Strengths of {analysis['best_option'].title()}")
                    strengths = analysis['strengths'][analysis['best_option']]
                    for strength in strengths:
                        st.markdown(f"- **{strength}**")

                with col2:
                    st.markdown(f"### Weaknesses of {analysis['best_option'].title()}")
                    weaknesses = analysis['weaknesses'][analysis['best_option']]
                    for weakness in weaknesses:
                        st.markdown(f"- **{weakness}**")

                # Get implementation details
                details = deployment.get_deployment_details(analysis['best_option'])

                st.markdown(f"### Implementation Details for {analysis['best_option'].title()}")

                st.markdown("#### Architecture")
                st.markdown(details["architecture"])

                st.markdown("#### Key Components")

                tab1, tab2 = st.tabs(["Client Components", "Server Components"])

                with tab1:
                    for component in details["client_components"]:
                        st.markdown(f"- {component}")

                with tab2:
                    for component in details["server_components"]:
                        st.markdown(f"- {component}")

                st.markdown("#### Implementation Challenges")
                for challenge in details["implementation_challenges"]:
                    st.markdown(f"- {challenge}")

        with tab3:
            st.subheader("Cost Analysis")

            col1, col2 = st.columns(2)

            with col1:
                deployment_option = st.selectbox(
                    "Select deployment option:",
                    deployment.deployment_options
                )

            with col2:
                user_count = st.number_input("Number of users/devices:", min_value=1, max_value=10000, value=100)
                image_count = st.number_input("Average images per user per month:", min_value=1, max_value=1000, value=50)

            # Calculate cost analysis
            cost_analysis = deployment.get_cost_analysis(deployment_option, user_count, image_count)

            # Display cost metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Initial Development", f"${cost_analysis['initial_development']:,.0f}")

            with col2:
                st.metric("Monthly Infrastructure", f"${cost_analysis['monthly_infrastructure']:,.0f}")

            with col3:
                st.metric("Total Monthly Cost", f"${cost_analysis['total_monthly']:,.0f}")

            # Display additional cost details
            st.markdown("### Cost Breakdown")

            costs_df = pd.DataFrame({
                'Cost Category': [
                    'Initial Development',
                    'Monthly Infrastructure',
                    'Monthly User Cost',
                    'Monthly Image Processing Cost',
                    'First Year Total',
                    'Subsequent Years (Annual)',
                    'Three-Year TCO',
                    'Per-User Cost (Annual)'
                ],
                'Amount': [
                    f"${cost_analysis['initial_development']:,.0f}",
                    f"${cost_analysis['monthly_infrastructure']:,.0f}",
                    f"${cost_analysis['monthly_user_cost']:,.0f}",
                    f"${cost_analysis['monthly_image_cost']:,.0f}",
                    f"${cost_analysis['first_year']:,.0f}",
                    f"${cost_analysis['subsequent_yearly']:,.0f}",
                    f"${cost_analysis['three_year_tco']:,.0f}",
                    f"${cost_analysis['per_user_yearly']:,.0f}"
                ]
            })

            st.dataframe(costs_df, use_container_width=True)

            # Create cost comparison chart
            st.markdown("### Cost Projection")

            years = list(range(1, 6))

            # Calculate costs over 5 years
            first_year_cost = cost_analysis['first_year']
            subsequent_yearly = cost_analysis['subsequent_yearly']

            cumulative_costs = [first_year_cost]
            for i in range(1, 5):
                cumulative_costs.append(cumulative_costs[-1] + subsequent_yearly)

            annual_costs = [first_year_cost] + [subsequent_yearly] * 4

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=years,
                y=annual_costs,
                name="Annual Cost"
            ))

            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_costs,
                mode="lines+markers",
                name="Cumulative Cost"
            ))

            fig.update_layout(
                title="5-Year Cost Projection",
                xaxis_title="Year",
                yaxis_title="Cost ($)",
                legend_title="Cost Type"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Bandwidth Requirements")

            col1, col2 = st.columns(2)

            with col1:
                bandwidth_option = st.selectbox(
                    "Select deployment option for bandwidth analysis:",
                    deployment.deployment_options
                )

            with col2:
                image_resolution = st.slider("Image resolution (megapixels):", 2.0, 20.0, 12.0, 0.5)
                images_per_session = st.slider("Images per session:", 1, 20, 5)

            # Calculate bandwidth analysis
            bandwidth = deployment.get_bandwidth_analysis(bandwidth_option, image_resolution, images_per_session)

            # Display bandwidth metrics
            st.markdown(f"### Total Session Data Size: {bandwidth['session_data_size_mb']:.1f} MB")

            # Create a transmission time comparison
            transmission_data = []

            for scenario, details in bandwidth['transmission_scenarios'].items():
                transmission_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Speed (Mbps)': details['speed_mbps'],
                    'Transmission Time (s)': details['transmission_time_seconds'],
                    'Feasibility': details['feasibility']
                })

            trans_df = pd.DataFrame(transmission_data)

            # Add color coding based on feasibility
            color_map = {
                'Good': 'green',
                'Marginal': 'orange',
                'Poor': 'red'
            }

            fig = px.bar(
                trans_df,
                x='Scenario',
                y='Transmission Time (s)',
                color='Feasibility',
                color_discrete_map=color_map,
                title="Transmission Time by Connection Type",
                text='Transmission Time (s)'
            )

            fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')

            st.plotly_chart(fig, use_container_width=True)

            # Display the full transmission details
            st.dataframe(trans_df, use_container_width=True)

            # Add contextual information
            st.markdown("""
            ### Bandwidth Considerations

            - **Good**: Under 10 seconds transmission time - excellent user experience
            - **Marginal**: 10-30 seconds transmission time - acceptable but may frustrate users
            - **Poor**: Over 30 seconds transmission time - likely to disrupt clinical workflow

            **Note**: The hybrid deployment option can optimize bandwidth use by processing some images locally and only uploading critical data when necessary.
            """)

        with tab5:
            Utils.render_deployment_page()

    @staticmethod
    def pages(section):
        if section == "Project Overview":
            Utils.overview()

        elif section == "Vision Module":
            Utils.vision_module()

        elif section == "Data Module":
            Utils.data_module()

        elif section == "Deployment Options":
            Utils.deployment_options()

        elif section == "Fairness Analysis":
            Utils.fairness_analysis()

        elif section == "Team Management":
            Utils.team_management()

        elif section == "Interview Preparation":
            Utils.interview_prep()

Utils.setup_page()

Utils.pages(st.sidebar.radio( "Go to", Utils.PAGES_LIST ))
