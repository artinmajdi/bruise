# Standard library imports
import os
from datetime import datetime

# Third-party imports
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import plotly.express as px
import cv2 # OpenCV for image processing

# Local application imports
from core import vision_module, data_module, deployment_module, fairness_module, leadership_module
from core.vision_module import BruiseDetectionModel, apply_als_filter, preprocess_image
from core.data_module import DatabaseSchema, FHIRDataModel
from core.deployment_module import DeploymentComparison
from core.fairness_module import FairnessMetrics, generate_fairness_report
from core.leadership_module import TeamManagement

# Import visualization tabs
from visualization_tabs.home_page_tab import HomePage
from visualization_tabs.computer_vision_tab import ComputerVisionPage
from visualization_tabs.fairness_page import FairnessPage
from visualization_tabs.data_engineering_tab import DataEngineeringPage
from visualization_tabs.mobile_deployment_tab import MobileDeploymentPage
from visualization_tabs.leadership_tab import LeadershipPage
from visualization_tabs.funding_tab import FundingPage

# Set page configuration
st.set_page_config(
    page_title="GMU Bruise Detection Postdoc Interview Prep",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InterviewPrepDashboard:
    def __init__(self):
        """Initialize dashboard with core modules"""
        self.vision_model = BruiseDetectionModel()
        self.fairness_metrics = FairnessMetrics()
        self.database_schema = DatabaseSchema()
        self.fhir_model = FHIRDataModel()
        self.deployment_comparison = DeploymentComparison()
        self.team_management = TeamManagement()

        # Initialize session state for navigation if not already set
        if 'page' not in st.session_state:
            st.session_state.page = "Home"

    def run(self):
        """Main method to run the dashboard"""
        # Apply custom CSS styling
        self.apply_custom_css()

        # Create sidebar for navigation
        self.create_sidebar()

        # Render main content based on selected page
        if st.session_state.page == "Home":
            HomePage().render()
        elif st.session_state.page == "Computer Vision":
            ComputerVisionPage().render()
        elif st.session_state.page == "Fairness":
            FairnessPage().render()
        elif st.session_state.page == "Data Engineering":
            DataEngineeringPage().render()
        elif st.session_state.page == "Mobile Deployment":
            MobileDeploymentPage().render()
        elif st.session_state.page == "Leadership":
            LeadershipPage().render()
        elif st.session_state.page == "Funding & Impact":
            FundingPage().render()

    def apply_custom_css(self):
        """Apply custom CSS styling to the dashboard"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #006633;  /* GMU green */
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #006633;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .section-header {
            font-size: 1.4rem;
            font-weight: 500;
            color: #006633;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .highlight-text {
            background-color: #f0f7f0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #006633;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #666;
            font-size: 0.8rem;
        }
        </style>
        """, unsafe_allow_html=True)

    def create_sidebar(self):
        """Create the sidebar navigation"""
        st.sidebar.markdown("## Navigation")

        # Navigation buttons
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("👁️ Computer Vision"):
            st.session_state.page = "Computer Vision"
        if st.sidebar.button("⚖️ Fairness"):
            st.session_state.page = "Fairness"
        if st.sidebar.button("🗄️ Data Engineering"):
            st.session_state.page = "Data Engineering"
        if st.sidebar.button("📱 Mobile Deployment"):
            st.session_state.page = "Mobile Deployment"
        if st.sidebar.button("👥 Leadership"):
            st.session_state.page = "Leadership"
        if st.sidebar.button("💰 Funding & Impact"):
            st.session_state.page = "Funding & Impact"

        st.sidebar.markdown("---")

        # Show project team information
        st.sidebar.markdown("## Project Team")
        st.sidebar.markdown("""
        - **Dr. Katherine Scafide**: Nursing/Forensics
        - **Dr. Janusz Wojtusiak**: AI/Health Informatics
        - **Dr. David Lattanzi**: Civil Eng/Imaging
        - **Dr. Terri Rebmann**: Nursing Dean
        - **Dr. Karen Trister Grace**: IPV Research
        """)

        # Add project stats from core modules
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Project Statistics")

        # Get stats from core modules
        model_version = self.vision_model.model_version
        supported_light_sources = ", ".join(self.vision_model.supported_light_sources)
        team_size = sum(len(members) for members in self.team_management.team_structure.values())

        # Display stats
        st.sidebar.markdown(f"**Model Version:** {model_version}")
        st.sidebar.markdown(f"**Supported Light Sources:** {supported_light_sources}")
        st.sidebar.markdown(f"**Team Size:** {team_size} members")

        # Show current date and interview preparation note
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Interview Preparation")
        st.sidebar.info("This dashboard demonstrates technical concepts and approaches relevant to the GMU Bruise Detection Postdoc position.")

        # Display current date and time
        now = datetime.now()
        st.sidebar.markdown(f"**Current Date:** {now.strftime('%B %d, %Y')}")




class DashboardTechnical:

	@staticmethod
	def mock_bruise_detection_cv(pil_image):
		"""
		Performs a mock bruise detection on a PIL Image using OpenCV.
		Converts PIL to OpenCV format, applies some basic image processing,
		draws mock "detected" contours, and returns the processed PIL Image
		along with the count of detected areas.
		"""
		try:
			# Convert PIL image to OpenCV format (RGB)
			image_cv = np.array(pil_image.convert('RGB'))

			# Convert to grayscale for processing
			gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

			# Apply Gaussian blur to reduce noise and improve contour detection
			blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

			# Use adaptive thresholding to segment potential bruise areas
			# This is a common technique for variable lighting conditions
			# THRESH_BINARY_INV makes bruises (darker areas) white for contour finding
			adaptive_thresh = cv2.adaptiveThreshold(
				blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
				cv2.THRESH_BINARY_INV, 11, 2  # Block size 11, Constant C=2
			)

			# Morphological operations to clean up the thresholded image
			# Opening removes small noise, Closing fills small holes in detected regions
			kernel = np.ones((5,5), np.uint8)
			opened_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
			closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=2)

			# Find contours of the potential bruise areas
			contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# Draw contours on the original color image
			output_image_cv = image_cv.copy()
			cv2.drawContours(output_image_cv, contours, -1, (0, 255, 0), 2)  # Draw contours in green

			# Convert the processed OpenCV image back to PIL format
			processed_pil_image = Image.fromarray(output_image_cv)

			return processed_pil_image, len(contours)

		except Exception as e:
			st.error(f"Error during mock bruise detection: {e}")
			return pil_image, 0 # Return original image and 0 contours on error


	def run(self):
		"""
		Main function to render the Streamlit application.
		"""
		st.sidebar.title("Navigation")
		app_mode = st.sidebar.radio(
			"Choose a section:",
			[
				"🏠 Home",
				"📊 Data Exploration & Input",
				"👁️ Bruise Detection Demo",
				"⚖️ Fairness & Bias Analysis",
				"🚀 Deployment Strategy",
				"👥 Project Overview & Team",
			],
			help="Select a section to explore different aspects of the Bruise Detection AI project."
		)

		if app_mode == "🏠 Home":
			DashboardTechnical.render_home_page()
		elif app_mode == "📊 Data Exploration & Input":
			DashboardTechnical.render_data_exploration_page()
		elif app_mode == "👁️ Bruise Detection Demo":
			DashboardTechnical.render_bruise_detection_page()
		elif app_mode == "⚖️ Fairness & Bias Analysis":
			DashboardTechnical.render_fairness_page()
		elif app_mode == "🚀 Deployment Strategy":
			DashboardTechnical.render_deployment_page()
		elif app_mode == "👥 Project Overview & Team":
			DashboardTechnical.render_project_overview_page()


	@staticmethod
	def render_home_page():
		"""
		Renders the Home page with a welcome message and project vision.
		"""
		st.title("🩹 Bruise Detection AI: Enhanced Interactive Dashboard")
		st.markdown(
			"""
			Welcome to the demonstration platform for the Bruise Detection AI project.
			This dashboard provides an interactive way to explore the project's key components,
			from data handling and model demonstration to fairness and deployment strategies.

			Use the **sidebar navigation** to explore different sections.
			"""
		)
		st.divider()

		# Display project vision using the vision_module
		st.subheader("Project Vision")
		col1, col2 = st.columns(2)
		with col1:
			vision_module.display_problem_statement() # Assumes this function uses st.write/markdown
		with col2:
			vision_module.display_solution_overview() # Assumes this function uses st.write/markdown


	@staticmethod
	def render_data_exploration_page():
		"""
		Renders the Data Exploration page for image input and synthetic data analysis.
		Utilizes functions from data_module.
		"""
		st.header("📊 Data Exploration & Input")
		st.markdown(
			"""
			Upload an image to be used in the detection demo, or load a sample.
			You can also generate and visualize synthetic tabular data related to bruise characteristics.
			"""
		)

		col1, col2 = st.columns([0.6, 0.4]) # Adjust column widths

		with col1:
			st.subheader("Image Input for Detection Demo")
			uploaded_file = st.file_uploader(
				"Upload an image of a skin area (potential bruise)",
				type=["png", "jpg", "jpeg"],
				help="This image will be used in the 'Bruise Detection Demo' section."
			)

			if uploaded_file:
				try:
					image = data_module.load_image_from_upload(uploaded_file)
					st.image(image, caption="Uploaded Image", use_column_width=True)
					st.session_state.current_image = image # Save for other pages
					st.success("Image uploaded successfully! Navigate to the 'Bruise Detection Demo' to process it.")
				except Exception as e:
					st.error(f"Error loading uploaded image: {e}")

			st.markdown("--- Or ---")

			if st.button("Load Sample Synthetic Image", help="Loads a synthetically generated image."):
				try:
					# Use create_synthetic_image from data_module
					image = data_module.create_synthetic_image(size=(300, 300))
					st.image(image, caption="Sample Synthetic Image", use_column_width=True)
					st.session_state.current_image = image
					st.info("Sample image loaded. Navigate to 'Bruise Detection Demo' to process it.")
				except Exception as e:
					st.error(f"Could not load sample image: {e}")
					if 'current_image' in st.session_state:
						del st.session_state.current_image # Clear if loading failed

		with col2:
			st.subheader("Synthetic Tabular Data")
			num_samples = st.slider(
				"Number of synthetic data samples to generate:",
				min_value=10, max_value=500, value=50, step=10,
				help="Adjust the slider to generate a different number of synthetic data points."
			)

			if st.button("Generate Synthetic Bruise Data Table"):
				try:
					synthetic_df = data_module.create_synthetic_bruise_data(num_samples)
					st.session_state.synthetic_data = synthetic_df
					st.dataframe(synthetic_df.head(), height=200) # Show head of the dataframe
				except Exception as e:
					st.error(f"Error generating synthetic data: {e}")

			if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
				df_to_plot = st.session_state.synthetic_data
				st.markdown("#### Plot Synthetic Data Distribution")

				# Allow user to select a column to plot
				# Filter for numeric columns suitable for histogram
				numeric_cols = df_to_plot.select_dtypes(include=np.number).columns.tolist()
				if not numeric_cols:
					st.warning("No numeric columns available in the synthetic data for plotting.")
				else:
					column_to_plot = st.selectbox(
						"Select a feature to plot its distribution:",
						options=numeric_cols,
						index=0 if 'bruise_severity' not in numeric_cols else numeric_cols.index('bruise_severity') if 'bruise_severity' in numeric_cols else 0, # Default to bruise_severity if exists
						help="Choose a numerical feature to see its distribution."
					)
					if column_to_plot:
						try:
							fig = data_module.plot_synthetic_data_distribution(df_to_plot, column_to_plot)
							if fig:
								st.plotly_chart(fig, use_container_width=True)
							else:
								st.info(f"Could not generate plot for '{column_to_plot}'. The column might be unsuitable or empty.")
						except Exception as e:
							st.error(f"Error plotting data: {e}")


	@staticmethod
	def render_bruise_detection_page():
		"""
		Renders the Bruise Detection Demo page.
		Uses an image from session_state and applies mock_bruise_detection_cv.
		"""
		st.header("👁️ Bruise Detection Demo")
		st.markdown(
			"""
			This section demonstrates a **mock** bruise detection process using basic computer vision techniques.
			Upload an image or load a sample in the 'Data Exploration & Input' section first.
			The green contours indicate areas the mock algorithm identified as potential bruises.
			"""
		)

		if 'current_image' not in st.session_state or st.session_state.current_image is None:
			st.warning("⚠️ Please upload or load a sample image in the '📊 Data Exploration & Input' section first.")
			if st.button("Go to Data Input"):
				# This is a bit of a hack to encourage navigation. Streamlit doesn't have direct page switching.
				st.info("Please navigate to '📊 Data Exploration & Input' from the sidebar.")
			return

		original_image = st.session_state.current_image

		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Original Image")
			st.image(original_image, caption="Original Image for Detection", use_column_width=True)

		with col2:
			st.subheader("Mock Detection Output")
			if st.button("Run Mock Bruise Detection", help="Applies a basic CV algorithm to find potential bruise areas."):
				with st.spinner("Processing image..."):
					processed_image, num_areas = mock_bruise_detection_cv(original_image)

				st.image(processed_image, caption=f"Processed Image - Detected {num_areas} potential areas.", use_column_width=True)
				st.success(f"Mock detection complete. Found **{num_areas}** potential bruise areas.")

				# Placeholder for mock model confidence/details
				st.markdown("#### Mock Model Output Details")
				st.json({
					"detected_potential_areas": num_areas,
					"mock_confidence_score": f"{np.random.uniform(0.65, 0.95):.2f}" if num_areas > 0 else "N/A",
					"processing_details": "Basic CV: Grayscale -> Blur -> Adaptive Threshold -> Contours"
				})
			else:
				st.info("Click the button above to run the mock detection.")


	@staticmethod
	def render_fairness_page():
		"""
		Renders the Fairness & Bias Analysis page.
		Uses functions from fairness_module and includes a simulated bias analysis.
		"""
		st.header("⚖️ Fairness & Bias Analysis")
		st.markdown(
			"""
			This section explores the critical aspect of fairness in AI models.
			It includes general information on fairness metrics and bias mitigation,
			as well as a simple interactive simulation to illustrate potential disparities.
			"""
		)

		with st.expander("Fairness Metrics Overview (from fairness_module)", expanded=True):
			fairness_module.display_fairness_metrics() # Assumes this function uses st.write/markdown

		st.subheader("Simulated Bias Analysis Demo")
		st.markdown(
			"""
			The following is a simplified simulation to demonstrate how model performance
			might vary across different demographic groups. **This is illustrative and uses mock data.**
			"""
		)

		# Mock data generation for bias simulation
		group_options = ['Group A', 'Group B', 'Group C']
		selected_groups = st.multiselect(
			"Select demographic groups for simulation:",
			options=group_options,
			default=group_options[:2],
			help="Choose groups to include in the mock bias simulation."
		)

		if st.button("Run Mock Bias Simulation", help="Generates mock data and shows performance differences."):
			if not selected_groups:
				st.warning("Please select at least one group for the simulation.")
			else:
				data_size_per_group = 100
				all_mock_bias_data = []
				for i, group in enumerate(selected_groups):
					# Introduce varying base accuracy and a slight bias for demonstration
					base_accuracy = 0.75 + (i * 0.05) # Slightly different base for each group
					accuracy_values = np.random.normal(loc=base_accuracy, scale=0.05, size=data_size_per_group)
					accuracy_values = np.clip(accuracy_values, 0.5, 1.0) # Clip values between 0.5 and 1.0

					group_data = pd.DataFrame({
						'group': group,
						'mock_accuracy': accuracy_values,
						'sample_id': range(data_size_per_group)
					})
					all_mock_bias_data.append(group_data)

				if all_mock_bias_data:
					mock_bias_df = pd.concat(all_mock_bias_data, ignore_index=True)

					st.markdown("##### Simulated Model Accuracy by Group")
					st.dataframe(mock_bias_df.groupby('group')['mock_accuracy'].agg(['mean', 'std', 'count']).round(3))

					fig = px.box(mock_bias_df, x='group', y='mock_accuracy',
								title='Simulated Model Accuracy Distribution by Group',
								labels={'mock_accuracy': 'Mock Accuracy Score', 'group': 'Demographic Group'},
								color='group')
					fig.update_layout(yaxis_title="Mock Accuracy Score")
					st.plotly_chart(fig, use_container_width=True)
					st.caption(
						"This plot shows a simulated disparity in model accuracy across different groups. "
						"In real-world scenarios, identifying and mitigating such biases is crucial."
					)
				else:
					st.info("No data generated for simulation.")


		with st.expander("Bias Mitigation Techniques (from fairness_module)"):
			fairness_module.display_bias_mitigation_techniques() # Assumes this function uses st.write/markdown


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
				deployment_module.display_deployment_options() # Assumes this function uses st.write/markdown

		with col2:
			with st.container(border=True):
				st.subheader("Scalability & Monitoring")
				deployment_module.display_scalability_info() # Assumes this function uses st.write/markdown

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
	def render_project_overview_page():
		"""
		Renders the Project Overview & Team page.
		Uses functions from leadership_module.
		"""
		st.header("👥 Project Overview & Team")
		st.markdown(
			"""
			Information about the project's leadership, team structure, and general timeline.
			"""
		)

		# Using expanders to keep the page clean
		with st.expander("Team Structure (from leadership_module)", expanded=True):
			leadership_module.display_team_structure() # Assumes this function uses st.write/markdown

		with st.expander("Project Timeline Highlights (from leadership_module)", expanded=False):
			leadership_module.display_project_timeline() # Assumes this function uses st.write/markdown

		st.info(
			"This section provides a high-level overview. For detailed project management artifacts, "
			"please refer to the project's internal documentation."
		)


if __name__ == "__main__":

	InterviewPrepDashboard().run()
	# DashboardTechnical().run()
