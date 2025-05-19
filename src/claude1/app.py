
import numpy as np
import pandas as pd
import io
import base64
import json
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our custom modules
from vision_module import BruiseDetectionModel, preprocess_image, apply_als_filter
from fairness_module import FairnessMetrics, generate_fairness_report
from data_module import DatabaseSchema, FHIRDataModel
from deployment_module import DeploymentComparison
from leadership_module import TeamManagement
# from funding_module import FundingStrategy



# Set page configuration
st.set_page_config(
	page_title="GMU Bruise Detection Postdoc Interview Prep",
	page_icon="🔬",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Main application class
class InterviewPrepDashboard:
	def __init__(self):
		self.vision_model = BruiseDetectionModel()
		self.fairness_metrics = FairnessMetrics()
		self.database_schema = DatabaseSchema()
		self.fhir_model = FHIRDataModel()
		self.deployment_comparison = DeploymentComparison()
		self.team_management = TeamManagement()
		# self.funding_strategy = FundingStrategy()

	def run(self):
		# Add custom CSS
		self.apply_custom_css()

		# Sidebar for navigation
		self.create_sidebar()

		# Main content
		if st.session_state.page == "Home":
			self.render_home_page()
		elif st.session_state.page == "Computer Vision":
			self.render_vision_page()
		elif st.session_state.page == "Fairness":
			self.render_fairness_page()
		elif st.session_state.page == "Data Engineering":
			self.render_data_engineering_page()
		elif st.session_state.page == "Mobile Deployment":
			self.render_deployment_page()
		elif st.session_state.page == "Leadership":
			self.render_leadership_page()
		elif st.session_state.page == "Funding & Impact":
			self.render_funding_page()

	def apply_custom_css(self):
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
		# Initialize session state for navigation
		if 'page' not in st.session_state:
			st.session_state.page = "Home"

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
		st.sidebar.markdown("## Project Team")
		st.sidebar.markdown("""
		- **Dr. Katherine Scafide**: Nursing/Forensics
		- **Dr. Janusz Wojtusiak**: AI/Health Informatics
		- **Dr. David Lattanzi**: Civil Eng/Imaging
		- **Dr. Terri Rebmann**: Nursing Dean
		- **Dr. Karen Trister Grace**: IPV Research
		""")

		st.sidebar.markdown("---")
		st.sidebar.markdown("### Interview Preparation")
		st.sidebar.info("This dashboard demonstrates technical concepts and approaches relevant to the GMU Bruise Detection Postdoc position.")

		# Display current date and time
		now = datetime.now()
		st.sidebar.markdown(f"**Current Date:** {now.strftime('%B %d, %Y')}")

	def render_home_page(self):
		st.markdown('<div class="main-header">George Mason University Postdoc Interview Preparation</div>', unsafe_allow_html=True)

		st.markdown("""
		Welcome to your interview preparation dashboard for the Postdoctoral Researcher position at George Mason University's College of Public Health. This position is part of the **Equitable and Accessible Software for Injury Detection (EAS-ID)** platform development team.

		This dashboard showcases your knowledge and approaches in key areas relevant to the project:
		""")

		# Create three columns for the main areas
		col1, col2, col3 = st.columns(3)

		with col1:
			st.markdown("### Technical Areas")
			st.markdown("""
			- **Computer Vision**: Deep learning approaches for bruise detection
			- **Fairness**: Equitable performance metrics across skin tones
			- **Data Engineering**: Secure, FHIR-compliant database design
			""")

		with col2:
			st.markdown("### Implementation")
			st.markdown("""
			- **Mobile Deployment**: On-device vs. cloud inference trade-offs
			- **Leadership**: Team coordination and mentoring approaches
			- **Funding & Impact**: Strategic vision for project sustainability
			""")

		with col3:
			st.markdown("### Project Context")
			st.markdown("""
			- $4.85M project to improve bruise detection across all skin tones
			- Multi-disciplinary team (nursing, computer science, engineering)
			- Focus on intimate partner violence documentation and evidence
			""")

		st.markdown("---")

		# Project overview
		st.markdown('<div class="sub-header">Project Overview: EAS-ID Platform</div>', unsafe_allow_html=True)

		st.markdown("""
		<div class="highlight-text">
		The EAS-ID platform aims to develop a mobile AI tool that makes bruises visible across all skin tones using:

		* Deep neural networks and computer vision
		* Alternate Light Source (ALS) imaging technology
		* Multi-spectral analysis techniques
		* Fairness-aware model development
		* Secure, HIPAA-compliant cloud architecture
		</div>
		""", unsafe_allow_html=True)

		# Display sample system architecture diagram
		st.markdown('<div class="section-header">Sample System Architecture</div>', unsafe_allow_html=True)

		# Create a simple architecture diagram with Plotly
		fig = go.Figure()

		# Add nodes
		nodes = {
			'Mobile': {'x': 0, 'y': 0, 'type': 'Client'},
			'API': {'x': 2, 'y': 0, 'type': 'Server'},
			'ML Model': {'x': 4, 'y': 0, 'type': 'Server'},
			'Database': {'x': 6, 'y': 0, 'type': 'Server'},
			'Web Dashboard': {'x': 2, 'y': 2, 'type': 'Client'},
			'Analytics': {'x': 4, 'y': 2, 'type': 'Server'},
			'Storage': {'x': 6, 'y': 2, 'type': 'Server'},
		}

		# Add edges
		edges = [
			('Mobile', 'API'),
			('API', 'ML Model'),
			('ML Model', 'Database'),
			('Web Dashboard', 'API'),
			('API', 'Analytics'),
			('Analytics', 'Storage'),
			('Database', 'Storage')
		]

		# Node colors
		colors = {
			'Client': '#4CAF50',  # Green
			'Server': '#2196F3'   # Blue
		}

		# Plot nodes
		for node, attrs in nodes.items():
			fig.add_trace(go.Scatter(
				x=[attrs['x']],
				y=[attrs['y']],
				mode='markers+text',
				marker=dict(size=30, color=colors[attrs['type']]),
				text=[node],
				textposition='bottom center',
				name=node
			))

		# Plot edges
		for edge in edges:
			fig.add_trace(go.Scatter(
				x=[nodes[edge[0]]['x'], nodes[edge[1]]['x']],
				y=[nodes[edge[0]]['y'], nodes[edge[1]]['y']],
				mode='lines',
				line=dict(width=2, color='gray'),
				showlegend=False
			))

		# Update layout
		fig.update_layout(
			title="EAS-ID System Architecture",
			showlegend=False,
			height=400,
			margin=dict(l=20, r=20, t=40, b=20),
			plot_bgcolor='white',
			xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
			yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
		)

		st.plotly_chart(fig, use_container_width=True)

		# Interview preparation tips
		st.markdown('<div class="sub-header">Interview Preparation Tips</div>', unsafe_allow_html=True)

		st.markdown("""
		* Review the key personnel's recent publications on bruise detection.
		* Understand the clinical workflow for forensic bruise documentation.
		* Prepare specific examples of your experience with computer vision and AI in healthcare.
		* Consider the ethical implications of AI in intimate partner violence documentation.
		* Develop thoughtful questions about the project's technical challenges and future direction.
		""")

		# Footer
		st.markdown("""
		<div class="footer">
		This dashboard was created to demonstrate technical knowledge and project understanding for the GMU Postdoc interview.
		</div>
		""", unsafe_allow_html=True)

	def render_vision_page(self):
		st.markdown('<div class="main-header">Computer Vision Approaches</div>', unsafe_allow_html=True)

		st.markdown("""
		This section demonstrates approaches to the interview question:

		> **"How would you segment a faint bruise on dark skin under ALS illumination?"**

		The interviewers are looking for depth of knowledge in handling noisy, low-contrast data and practical algorithmic choices.
		""")

		# Create tabs for different aspects of the vision approach
		cv_tab1, cv_tab2, cv_tab3 = st.tabs(["Image Processing Pipeline", "Model Architecture", "Implementation Demo"])

		with cv_tab1:
			st.markdown('<div class="sub-header">Proposed Image Processing Pipeline</div>', unsafe_allow_html=True)

			st.markdown("""
			For segmenting faint bruises on dark skin under ALS illumination, I propose a multi-stage pipeline:

			1. **Multi-spectral acquisition**: Capture images under different wavelengths (white light + 415nm and 450nm ALS) to enhance bruise visibility
			2. **Image pre-processing**: Apply noise reduction, contrast enhancement, and color channel manipulation specific to ALS imagery
			3. **Deep learning segmentation**: Use a specialized U-Net architecture trained on multi-channel inputs
			4. **Post-processing**: Apply CRF (Conditional Random Fields) to refine segmentation boundaries
			""")

			# Create a visual pipeline diagram
			pipeline_stages = [
				"Image\nAcquisition",
				"Noise\nReduction",
				"ALS Color\nEnhancement",
				"Multi-channel\nU-Net",
				"CRF\nRefinement",
				"Bruise\nSegmentation"
			]

			# Visualization using Plotly
			fig = go.Figure()

			# Add nodes
			x_positions = np.linspace(0, 10, len(pipeline_stages))
			y_position = 5

			# Add nodes and connections
			for i, stage in enumerate(pipeline_stages):
				# Add node
				fig.add_trace(go.Scatter(
					x=[x_positions[i]],
					y=[y_position],
					mode='markers+text',
					marker=dict(size=40, color='rgba(0, 102, 51, 0.8)'),
					text=[stage],
					textposition='middle center',
					textfont=dict(color='black', size=10),
					name=stage
				))

				# Add connection to next node
				if i < len(pipeline_stages) - 1:
					fig.add_trace(go.Scatter(
						x=[x_positions[i], x_positions[i+1]],
						y=[y_position, y_position],
						mode='lines+text',
						line=dict(width=2, color='gray'),
						text=[''],
						showlegend=False
					))

			# Update layout
			fig.update_layout(
				title="Bruise Segmentation Pipeline",
				showlegend=False,
				height=250,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor='white',
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10])
			)

			st.plotly_chart(fig, use_container_width=True)

			# Detailed explanation of each stage
			st.markdown('<div class="section-header">Detail: ALS Color Enhancement</div>', unsafe_allow_html=True)

			st.markdown("""
			When working with Alternate Light Source (ALS) images for bruise detection, specialized preprocessing is crucial:

			1. **Channel manipulation**: While RGB conversion is standard for white light images, ALS requires different processing:
			   - For 415nm (violet) light: Enhance blue-green channels
			   - For 450nm (blue) light: Enhance green channel

			2. **Contrast enhancement**: Apply adaptive histogram equalization specifically tuned to each channel

			3. **Edge-preserving smoothing**: Use bilateral filtering to reduce noise while preserving bruise boundaries

			4. **Background subtraction**: Use surrounding tissue as reference to highlight bruise-specific reflectance
			""")

		with cv_tab2:
			st.markdown('<div class="sub-header">Model Architecture Selection</div>', unsafe_allow_html=True)

			st.markdown("""
			For faint bruise segmentation under ALS illumination on dark skin, I would implement:

			### Multi-Channel Light-Adaptive U-Net

			This architecture builds on the foundation of U-Net (which excels at medical image segmentation) with specific modifications for the bruise detection task:
			""")

			# Create columns for comparison
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				#### Key Architectural Innovations

				1. **Multi-channel input handling**:
				   - Accepts both white light and ALS channels
				   - Early fusion of different light spectra

				2. **Attention mechanism**:
				   - Spatial attention gates in skip connections
				   - Focus on subtle bruise features

				3. **Task-specific blocks**:
				   - Contrast-sensitive convolutional blocks
				   - Fine-tuned for low-contrast features

				4. **Uncertainty awareness**:
				   - Outputs confidence maps alongside segmentation
				   - Crucial for forensic applications
				""")

			with col2:
				st.markdown("""
				#### Implementation Details

				1. **Encoder backbone**:
				   - EfficientNet-B2 (balance of accuracy and speed)
				   - Pre-trained on dermoscopy datasets, fine-tuned on bruise data

				2. **Decoder structure**:
				   - Transposed convolutions with residual connections
				   - Channel attention modules after each upsampling

				3. **Loss function**:
				   - Weighted combination of:
					 - Dice loss (address class imbalance)
					 - Boundary loss (emphasize edges)
					 - Perceptual loss (tissue similarity)

				4. **Regularization**:
				   - Dropout (0.3)
				   - Mixed-sample data augmentation
				""")

			# Architecture visualization
			st.markdown('<div class="section-header">Model Visualization</div>', unsafe_allow_html=True)

			# U-Net architecture visual using Plotly
			fig = go.Figure()

			# Define architecture components
			layers = [
				{'name': 'Input', 'depth': 0, 'width': 3},
				{'name': 'Conv1', 'depth': 1, 'width': 3},
				{'name': 'Pool1', 'depth': 2, 'width': 3},
				{'name': 'Conv2', 'depth': 3, 'width': 2},
				{'name': 'Pool2', 'depth': 4, 'width': 2},
				{'name': 'Conv3', 'depth': 5, 'width': 1},
				{'name': 'Bottleneck', 'depth': 6, 'width': 1},
				{'name': 'Upconv1', 'depth': 5, 'width': 1},
				{'name': 'Concat1', 'depth': 5, 'width': 2},
				{'name': 'Upconv2', 'depth': 3, 'width': 2},
				{'name': 'Concat2', 'depth': 3, 'width': 3},
				{'name': 'Output', 'depth': 1, 'width': 3},
			]

			# Scaling factors
			x_scale = 1
			y_scale = 1

			# Add nodes
			for layer in layers:
				x = layer['depth'] * x_scale
				y = layer['width'] * y_scale

				# Different colors based on layer type
				if 'Conv' in layer['name']:
					color = 'rgba(0, 102, 51, 0.8)'  # Green
				elif 'Pool' in layer['name']:
					color = 'rgba(30, 136, 229, 0.8)'  # Blue
				elif 'Up' in layer['name']:
					color = 'rgba(216, 67, 21, 0.8)'  # Red
				elif 'Concat' in layer['name']:
					color = 'rgba(255, 193, 7, 0.8)'  # Yellow
				else:
					color = 'rgba(158, 158, 158, 0.8)'  # Gray

				fig.add_trace(go.Scatter(
					x=[x],
					y=[y],
					mode='markers+text',
					marker=dict(size=30, color=color),
					text=[layer['name']],
					textposition='bottom center',
					name=layer['name']
				))

			# Add skip connections
			fig.add_trace(go.Scatter(
				x=[1, 5], y=[3, 2],
				mode='lines',
				line=dict(width=2, color='rgba(255, 193, 7, 0.8)', dash='dash'),
				showlegend=False
			))

			fig.add_trace(go.Scatter(
				x=[3, 3], y=[2, 3],
				mode='lines',
				line=dict(width=2, color='rgba(255, 193, 7, 0.8)', dash='dash'),
				showlegend=False
			))

			# Update layout
			fig.update_layout(
				title="Multi-Channel U-Net Architecture",
				showlegend=False,
				height=300,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor='white',
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
			)

			st.plotly_chart(fig, use_container_width=True)

			st.markdown("""
			### Handling Dark Skin Challenges

			For dark skin tones specifically, the model includes:

			1. **Skin-tone adaptive preprocessing**: Dynamically adjust contrast enhancement based on detected skin tone

			2. **Fitzpatrick-aware data augmentation**: Generate synthetic examples across skin tones using CycleGAN

			3. **Multi-spectral feature fusion**: Combine features from different ALS wavelengths before final segmentation

			4. **Uncertainty calibration**: Provide confidence scores calibrated specifically for different skin types
			""")

		with cv_tab3:
			st.markdown('<div class="sub-header">Implementation Demo</div>', unsafe_allow_html=True)

			st.markdown("""
			This interactive demo simulates bruise detection using different preprocessing techniques and ALS filters.
			""")

			# Create columns for the demo controls and results
			demo_col1, demo_col2 = st.columns([1, 2])

			with demo_col1:
				st.markdown("### Controls")

				# Demo controls
				skin_tone = st.select_slider(
					"Skin Tone (Fitzpatrick Scale)",
					options=["Type I-II (Light)", "Type III-IV (Medium)", "Type V-VI (Dark)"]
				)

				bruise_visibility = st.slider("Bruise Visibility", 10, 100, 30)

				light_source = st.radio(
					"Light Source",
					["White Light", "ALS 415nm (Violet)", "ALS 450nm (Blue)"]
				)

				# Processing options
				st.markdown("### Processing Options")

				apply_noise_reduction = st.checkbox("Apply Noise Reduction", True)
				apply_contrast_enhancement = st.checkbox("Apply Contrast Enhancement", True)
				apply_channel_enhancement = st.checkbox("Apply Channel Enhancement", True)
				apply_segmentation = st.checkbox("Apply AI Segmentation", True)

			with demo_col2:
				# Generate simulated bruise image based on parameters
				st.markdown("### Image Processing Results")

				# Generate synthetic base image
				np.random.seed(42)  # For reproducibility

				# Map skin tone to background color
				if skin_tone == "Type I-II (Light)":
					bg_color = [230, 200, 180]
				elif skin_tone == "Type III-IV (Medium)":
					bg_color = [180, 140, 120]
				else:  # Type V-VI (Dark)
					bg_color = [90, 60, 50]

				# Create base image with skin tone
				img_size = 300
				base_img = np.ones((img_size, img_size, 3), dtype=np.uint8)
				for i in range(3):
					base_img[:, :, i] = bg_color[i]

				# Add some natural skin texture
				texture = np.random.normal(0, 10, (img_size, img_size, 3))
				base_img = np.clip(base_img + texture, 0, 255).astype(np.uint8)

				# Add a synthetic bruise
				center_x, center_y = img_size // 2, img_size // 2
				radius = 60

				bruise_mask = np.zeros((img_size, img_size))
				for i in range(img_size):
					for j in range(img_size):
						dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
						if dist < radius:
							# Gradient falloff for natural appearance
							bruise_mask[i, j] = max(0, 1 - (dist / radius) ** 2)

				# Make bruise less visible for dark skin
				if skin_tone == "Type V-VI (Dark)":
					visibility_factor = bruise_visibility * 0.5 / 100
				else:
					visibility_factor = bruise_visibility / 100

				# Apply bruise coloration based on light source
				bruised_img = base_img.copy()

				if light_source == "White Light":
					# Purple-brown bruise under white light
					bruised_img[:, :, 0] = np.clip(base_img[:, :, 0] - 30 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # Less red
					bruised_img[:, :, 1] = np.clip(base_img[:, :, 1] - 40 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # Less green
					bruised_img[:, :, 2] = np.clip(base_img[:, :, 2] - 10 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # Less blue reduction

				elif light_source == "ALS 415nm (Violet)":
					# Bruise fluorescence under 415nm
					bruised_img[:, :, 0] = np.clip(base_img[:, :, 0] - 40 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # Less red
					bruised_img[:, :, 1] = np.clip(base_img[:, :, 1] + 20 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # More green
					bruised_img[:, :, 2] = np.clip(base_img[:, :, 2] + 60 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # More blue

				else:  # ALS 450nm (Blue)
					# Bruise fluorescence under 450nm
					bruised_img[:, :, 0] = np.clip(base_img[:, :, 0] - 40 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # Less red
					bruised_img[:, :, 1] = np.clip(base_img[:, :, 1] + 40 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # More green
					bruised_img[:, :, 2] = np.clip(base_img[:, :, 2] + 20 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)  # More blue

				# Create PIL image for processing
				pil_img = Image.fromarray(bruised_img)

				# Apply processing steps
				processed_img = pil_img.copy()

				if apply_noise_reduction:
					# Simulate bilateral filtering
					processed_img = processed_img.filter(ImageFilter.GaussianBlur(0.5))

				if apply_contrast_enhancement:
					# Enhance contrast
					enhancer = ImageEnhance.Contrast(processed_img)
					processed_img = enhancer.enhance(1.5)

				if apply_channel_enhancement:
					# Channel-specific enhancement based on light source
					r, g, b = processed_img.split()

					if light_source == "ALS 415nm (Violet)":
						# Enhance blue and green channels
						g_enhancer = ImageEnhance.Contrast(g)
						g = g_enhancer.enhance(1.2)

						b_enhancer = ImageEnhance.Contrast(b)
						b = b_enhancer.enhance(1.5)

					elif light_source == "ALS 450nm (Blue)":
						# Enhance green channel
						g_enhancer = ImageEnhance.Contrast(g)
						g = g_enhancer.enhance(1.5)

					processed_img = Image.merge("RGB", (r, g, b))

				# Convert back to numpy for segmentation
				processed_array = np.array(processed_img)

				# Create a simulated segmentation mask
				segmentation_mask = None
				if apply_segmentation:
					# Simulate AI-generated segmentation mask
					segmentation_mask = np.zeros((img_size, img_size, 4), dtype=np.uint8)

					# Base mask on the bruise but add some "AI imperfections"
					for i in range(img_size):
						for j in range(img_size):
							if bruise_mask[i, j] > 0.3:
								# Add some noise to make it look like AI prediction
								if np.random.random() > 0.1:  # 90% accuracy
									alpha = int(min(255, bruise_mask[i, j] * 180))
									segmentation_mask[i, j] = [255, 0, 0, alpha]

				# Display original image
				st.markdown("#### Original Image")
				st.image(pil_img, use_column_width=True)

				# Display processed image
				st.markdown("#### Processed Image")
				st.image(processed_img, use_column_width=True)

				# Display segmentation if applied
				if apply_segmentation and segmentation_mask is not None:
					st.markdown("#### AI Segmentation")

					# Overlay segmentation on original image
					seg_img = processed_array.copy()
					for i in range(img_size):
						for j in range(img_size):
							if segmentation_mask[i, j, 3] > 0:
								alpha = segmentation_mask[i, j, 3] / 255.0
								seg_img[i, j, 0] = int(seg_img[i, j, 0] * (1 - alpha) + segmentation_mask[i, j, 0] * alpha)
								seg_img[i, j, 1] = int(seg_img[i, j, 1] * (1 - alpha) + segmentation_mask[i, j, 1] * alpha)
								seg_img[i, j, 2] = int(seg_img[i, j, 2] * (1 - alpha) + segmentation_mask[i, j, 2] * alpha)

					st.image(seg_img, use_column_width=True)

			# Explanation of techniques
			st.markdown('<div class="section-header">Technical Explanation</div>', unsafe_allow_html=True)

			st.markdown("""
			The demo above illustrates the key challenges in bruise detection across skin tones and how different imaging and processing techniques can help:

			1. **Alternate Light Source (ALS)**: Bruises absorb and reflect light differently than surrounding tissue, especially at specific wavelengths:
			   - 415nm (violet) causes bruise hemoglobin to appear with enhanced contrast
			   - 450nm (blue) is effective for highlighting older bruises

			2. **Dark Skin Considerations**:
			   - Lower color contrast between bruise and skin in darker skin tones
			   - ALS significantly improves detection capability (5× improvement shown in Dr. Scafide's research)
			   - Channel-specific enhancement targets the fluorescence patterns

			3. **Segmentation Challenges**:
			   - Class imbalance (small bruise area vs. large background)
			   - Low signal-to-noise ratio
			   - Boundary ambiguity
			   - Domain shift between different skin tones

			This demonstration shows why a multi-spectral approach combined with specialized preprocessing and deep learning is essential for equitable bruise detection.
			""")

	def render_fairness_page(self):
		st.markdown('<div class="main-header">Fairness in Bruise Detection</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"Describe a metric you would report to show equitable performance."**

		The interviewers are looking for awareness of bias and reporting metrics like Demographic Parity Difference.
		""")

		# Create tabs for different aspects of fairness
		f_tab1, f_tab2, f_tab3 = st.tabs(["Fairness Metrics", "Bias Mitigation", "Fairness Dashboard"])

		with f_tab1:
			st.markdown('<div class="sub-header">Fairness Metrics for Skin Tone Equity</div>', unsafe_allow_html=True)

			st.markdown("""
			For the bruise detection system, I would implement a comprehensive fairness evaluation framework that specifically addresses the challenges of detecting bruises across different skin tones (Fitzpatrick scale I-VI).

			The fundamental metric categories are:
			""")

			# Create columns for different metric categories
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Group Fairness Metrics

				1. **Demographic Parity Difference**
				   - Measures whether positive prediction rates are similar across skin tones
				   - Formula: |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
				   - Target: < 0.05 difference between any skin tone groups

				2. **Equalized Odds**
				   - Ensures similar true positive and false positive rates across groups
				   - Formula: |TPR_a - TPR_b| and |FPR_a - FPR_b|
				   - Critical for forensic applications

				3. **Predictive Parity**
				   - Measures if positive predictive value is similar across skin tones
				   - Formula: |PPV_a - PPV_b|
				   - Important for medical decision confidence
				""")

			with col2:
				st.markdown("""
				### Detection-Specific Metrics

				1. **Bruise Detection Rate (BDR)**
				   - The ratio of correctly detected bruises to total bruises
				   - Stratified by skin tone and bruise age
				   - Formula: TP / (TP + FN) for each skin tone group

				2. **Minimum Detectable Contrast (MDC)**
				   - Smallest bruise-to-skin contrast ratio detectable
				   - Lower MDC indicates better detection in challenging cases
				   - Should be reported separately for each skin tone

				3. **Localization Equity**
				   - Measures consistency of bruise boundary accuracy across skin tones
				   - Based on Dice coefficient or IoU ratios between groups
				   - Formula: |IoU_a - IoU_b| < threshold
				""")

			st.markdown('<div class="section-header">Sample Fairness Report</div>', unsafe_allow_html=True)

			# Generate sample fairness metrics data
			skin_tones = ["Type I-II", "Type III-IV", "Type V-VI"]

			# Sample metrics
			metrics_data = {
				"Skin Tone": skin_tones,
				"Bruise Detection Rate": [0.91, 0.89, 0.85],
				"False Positive Rate": [0.03, 0.04, 0.06],
				"IoU Score": [0.85, 0.82, 0.79]
			}

			metrics_df = pd.DataFrame(metrics_data)

			# Calculate fairness gap
			max_bdr = max(metrics_data["Bruise Detection Rate"])
			min_bdr = min(metrics_data["Bruise Detection Rate"])
			fairness_gap = max_bdr - min_bdr

			# Plot metrics
			fig = px.bar(
				metrics_df,
				x="Skin Tone",
				y=["Bruise Detection Rate", "False Positive Rate", "IoU Score"],
				barmode="group",
				title="Detection Performance by Skin Tone Group",
				color_discrete_sequence=["#4CAF50", "#F44336", "#2196F3"]
			)

			fig.update_layout(
				height=400,
				margin=dict(l=20, r=20, t=40, b=20),
				legend_title="Metric"
			)

			st.plotly_chart(fig, use_container_width=True)

			# Show fairness gap callout
			st.markdown(f"""
			<div class="highlight-text">
			<b>Fairness Gap Analysis:</b><br>
			The current model shows a Bruise Detection Rate (BDR) fairness gap of <b>{fairness_gap:.2f}</b> between the highest and lowest performing skin tone groups.

			Our target threshold is <0.05 to ensure equitable performance.

			This gap indicates that further work is needed on:
			1. More training data for Type V-VI skin tones
			2. Enhanced preprocessing for darker skin
			3. Potential model architecture modifications
			</div>
			""", unsafe_allow_html=True)

			# Equity threshold explanation
			st.markdown("""
			### Recommended Equity Thresholds

			Based on literature review and clinical significance, I propose these fairness thresholds:

			1. **Demographic Parity Gap**: < 0.05 difference between any skin tone groups
			2. **Bruise Detection Rate Gap**: < 0.05 difference between skin tone groups
			3. **False Positive Rate Gap**: < 0.03 difference between skin tone groups
			4. **Precision Gap**: < 0.05 difference between skin tone groups
			5. **IoU/Dice Score Gap**: < 0.07 difference between skin tone groups

			These thresholds are based on NIH AIM-AHEAD guidelines and clinical significance for forensic evidence standards.
			""")

		with f_tab2:
			st.markdown('<div class="sub-header">Bias Mitigation Strategies</div>', unsafe_allow_html=True)

			st.markdown("""
			To achieve equitable performance across skin tones, I would implement a multi-faceted bias mitigation strategy:
			""")

			# Create columns for pre-processing, in-processing, and post-processing
			col1, col2, col3 = st.columns(3)

			with col1:
				st.markdown("""
				### Pre-processing Techniques

				1. **Balanced Dataset Creation**
				   - Equal representation across skin tones
				   - Stratified sampling by Fitzpatrick scale
				   - Synthetic data generation for underrepresented groups

				2. **Data Augmentation**
				   - Skin tone transformations
				   - Contrast-preserving augmentations
				   - Bruise appearance variations

				3. **Fitzpatrick-Aware Preprocessing**
				   - Adaptive contrast enhancement
				   - Skin tone specific channel manipulation
				   - Multi-spectral normalization
				""")

			with col2:
				st.markdown("""
				### In-processing Techniques

				1. **Fairness-aware Loss Functions**
				   - Group-DRO (Distributionally Robust Optimization)
				   - Adversarial debiasing
				   - Fairness regularization terms

				2. **Model Architecture**
				   - Skin tone detection branch
				   - Adaptive feature normalization
				   - Attention mechanisms for different skin tones

				3. **Training Strategy**
				   - Curriculum learning by difficulty
				   - Gradient accumulation with fairness constraints
				   - Multi-task learning with auxiliary fairness objectives
				""")

			with col3:
				st.markdown("""
				### Post-processing Techniques

				1. **Threshold Optimization**
				   - Skin tone specific decision thresholds
				   - ROC analysis for optimal operating points
				   - Confidence calibration by group

				2. **Ensemble Methods**
				   - Specialized models for different skin tones
				   - Weighted ensemble based on skin tone
				   - Stacking with fairness-aware meta-learner

				3. **Human-in-the-Loop**
				   - Active learning for edge cases
				   - Feedback incorporation process
				   - Continuous fairness monitoring
				""")

			# Implementation roadmap
			st.markdown('<div class="section-header">Fairness Implementation Roadmap</div>', unsafe_allow_html=True)

			# Create a timeline visualization
			timeline_data = {
				"Stage": [
					"Baseline Assessment",
					"Dataset Enhancement",
					"Initial Debiasing",
					"Model Architecture",
					"Threshold Optimization",
					"Clinical Validation",
					"Continuous Monitoring"
				],
				"Start": [0, 1, 2, 3, 4, 5, 6],
				"Duration": [1, 2, 2, 3, 1, 2, 1],
				"Description": [
					"Evaluate baseline model performance across skin tones",
					"Balanced dataset creation with synthetic data generation",
					"Implement pre-processing debiasing techniques",
					"Develop and test fairness-aware model architectures",
					"Optimize detection thresholds for each skin tone group",
					"Clinical validation with diverse patient populations",
					"Deploy continuous fairness monitoring system"
				]
			}

			# Create Gantt chart
			fig = px.timeline(
				timeline_data,
				x_start = "Start",
				x_end   = lambda x: x["Start"] + x["Duration"],
				y       = "Stage",
				text    = "Description",
				title   = "Fairness Implementation Timeline (Months)",
				color_discrete_sequence = ["#4CAF50"]
			)

			fig.update_layout(
				height=400,
				margin=dict(l=20, r=20, t=40, b=20)
			)

			# Hide axis labels
			fig.update_yaxes(title="")
			fig.update_xaxes(title="Months")

			st.plotly_chart(fig, use_container_width=True)

			# Case study
			st.markdown('<div class="section-header">Case Study: Successful Bias Mitigation</div>', unsafe_allow_html=True)

			st.markdown("""
			<div class="highlight-text">
			<b>Successful Approach in Similar Domain: Skin Lesion Classification</b>

			The ISIC 2019 challenge demonstrated effective bias mitigation in skin lesion detection across skin tones:

			1. <b>Dataset Rebalancing</b>: Created synthetic dark skin examples using CycleGAN to balance training data

			2. <b>Multi-Task Learning</b>: Added skin tone classification as auxiliary task, sharing early features

			3. <b>Adaptive Preprocessing</b>: Developed skin-tone specific preprocessing pipeline

			4. <b>Results</b>: Reduced performance gap from 0.15 to 0.03 in detection rate across skin tone groups

			This approach can be adapted to our bruise detection task with ALS imaging as an additional input channel.
			</div>
			""", unsafe_allow_html=True)

		with f_tab3:
			st.markdown('<div class="sub-header">Fairness Monitoring Dashboard</div>', unsafe_allow_html=True)

			st.markdown("""
			For the EAS-ID platform, I would develop a comprehensive fairness monitoring dashboard that tracks performance across demographic groups over time. This would enable:

			1. Real-time monitoring of fairness metrics
			2. Early detection of performance drift
			3. Transparent reporting for stakeholders
			4. Identification of areas for model improvement
			""")

			# Create a mock dashboard
			st.markdown('<div class="section-header">Fairness Dashboard Preview</div>', unsafe_allow_html=True)

			# Create tabs for different dashboard sections
			dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Performance Overview", "Detailed Metrics", "Failure Analysis"])

			with dash_tab1:
				# Generate some sample data
				np.random.seed(42)

				# Dates for time series
				dates = pd.date_range(start='2025-01-01', periods=12, freq='W')

				# Performance data
				performance_data = {
					"Date": list(dates) * 3,
					"Skin Tone": ["Type I-II"] * 12 + ["Type III-IV"] * 12 + ["Type V-VI"] * 12,
					"Detection Rate": np.clip(
						np.concatenate([
							0.92 + np.random.normal(0, 0.01, 12),  # Type I-II
							0.90 + np.random.normal(0, 0.01, 12),  # Type III-IV
							0.88 + np.random.normal(0, 0.015, 12)  # Type V-VI
						]),
						0, 1
					),
					"False Positive Rate": np.clip(
						np.concatenate([
							0.03 + np.random.normal(0, 0.005, 12),  # Type I-II
							0.04 + np.random.normal(0, 0.005, 12),  # Type III-IV
							0.05 + np.random.normal(0, 0.01, 12)    # Type V-VI
						]),
						0, 1
					),
					"Precision": np.clip(
						np.concatenate([
							0.94 + np.random.normal(0, 0.01, 12),  # Type I-II
							0.92 + np.random.normal(0, 0.01, 12),  # Type III-IV
							0.89 + np.random.normal(0, 0.015, 12)  # Type V-VI
						]),
						0, 1
					)
				}

				performance_df = pd.DataFrame(performance_data)

				# Detection rate over time
				fig1 = px.line(
					performance_df,
					x="Date",
					y="Detection Rate",
					color="Skin Tone",
					title="Bruise Detection Rate by Skin Tone (Over Time)",
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig1.update_layout(
					height=300,
					margin=dict(l=20, r=20, t=40, b=20),
					yaxis=dict(range=[0.85, 0.95])
				)

				st.plotly_chart(fig1, use_container_width=True)

				# Fairness gap over time
				fairness_gap_data = []

				for date in dates:
					date_subset = performance_df[performance_df["Date"] == date]
					max_rate = date_subset["Detection Rate"].max()
					min_rate = date_subset["Detection Rate"].min()
					gap = max_rate - min_rate
					fairness_gap_data.append({
						"Date": date,
						"Fairness Gap": gap
					})

				fairness_gap_df = pd.DataFrame(fairness_gap_data)

				fig2 = px.line(
					fairness_gap_df,
					x="Date",
					y="Fairness Gap",
					title="Fairness Gap Over Time (Detection Rate)",
					color_discrete_sequence=["#FF9800"]
				)

				# Add threshold line
				fig2.add_hline(
					y=0.05,
					line_dash="dash",
					line_color="red",
					annotation_text="Threshold (0.05)",
					annotation_position="bottom right"
				)

				fig2.update_layout(
					height=250,
					margin=dict(l=20, r=20, t=40, b=20),
					yaxis=dict(range=[0, 0.1])
				)

				st.plotly_chart(fig2, use_container_width=True)

				# Summary metrics
				st.markdown("""
				<div class="highlight-text">
				<b>Fairness Summary Metrics:</b>

				- Current Detection Rate Gap: 0.04 (Below threshold of 0.05) ✓
				- Current False Positive Rate Gap: 0.02 (Below threshold of 0.03) ✓
				- Current Precision Gap: 0.05 (At threshold of 0.05) ⚠️

				<b>Trend Analysis:</b> Fairness gap has decreased by 38% since initial deployment, showing continued improvement in equitable performance.
				</div>
				""", unsafe_allow_html=True)

			with dash_tab2:
				# ROC curves for different skin tones
				st.markdown("### ROC Curves by Skin Tone")

				# Generate ROC curve data
				def generate_roc_data(base_tpr, noise):
					fpr = np.linspace(0, 1, 100)
					tpr = np.clip(base_tpr * fpr + (1 - base_tpr) * fpr**2 + np.random.normal(0, noise, 100), 0, 1)
					return fpr, tpr

				fpr1, tpr1 = generate_roc_data(0.95, 0.01)  # Type I-II
				fpr2, tpr2 = generate_roc_data(0.92, 0.015)  # Type III-IV
				fpr3, tpr3 = generate_roc_data(0.88, 0.02)  # Type V-VI

				fig = go.Figure()

				fig.add_trace(go.Scatter(x=fpr1, y=tpr1, name="Type I-II", line=dict(color="#4CAF50", width=2)))
				fig.add_trace(go.Scatter(x=fpr2, y=tpr2, name="Type III-IV", line=dict(color="#2196F3", width=2)))
				fig.add_trace(go.Scatter(x=fpr3, y=tpr3, name="Type V-VI", line=dict(color="#F44336", width=2)))
				fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="grey", width=2, dash="dash")))

				fig.update_layout(
					title="ROC Curves by Skin Tone",
					xaxis_title="False Positive Rate",
					yaxis_title="True Positive Rate",
					height=400,
					margin=dict(l=20, r=20, t=40, b=20),
					legend=dict(x=0.7, y=0.2)
				)

				st.plotly_chart(fig, use_container_width=True)

				# Confusion matrices for different skin tones
				st.markdown("### Confusion Matrices by Skin Tone")

				# Create columns for confusion matrices
				cm_col1, cm_col2, cm_col3 = st.columns(3)

				# Sample confusion matrix data
				def generate_cm(tp, fp, fn, tn):
					return np.array([
						[tp, fp],
						[fn, tn]
					])

				cm1 = generate_cm(92, 8, 6, 94)  # Type I-II
				cm2 = generate_cm(90, 10, 9, 91)  # Type III-IV
				cm3 = generate_cm(87, 13, 11, 89)  # Type V-VI

				# Plot confusion matrices
				with cm_col1:
					fig = px.imshow(
						cm1,
						text_auto=True,
						title="Type I-II",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Greens"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				with cm_col2:
					fig = px.imshow(
						cm2,
						text_auto=True,
						title="Type III-IV",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Blues"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				with cm_col3:
					fig = px.imshow(
						cm3,
						text_auto=True,
						title="Type V-VI",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Reds"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				# Detection performance by bruise age
				st.markdown("### Detection Performance by Bruise Age")

				# Sample data for bruise age vs detection rate
				bruise_age_data = {
					"Bruise Age (days)": [1, 3, 5, 7, 10, 14],
					"Type I-II": [0.95, 0.93, 0.91, 0.88, 0.84, 0.80],
					"Type III-IV": [0.92, 0.90, 0.87, 0.84, 0.80, 0.76],
					"Type V-VI": [0.88, 0.85, 0.82, 0.78, 0.74, 0.70]
				}

				df = pd.DataFrame(bruise_age_data)
				df_melted = pd.melt(
					df,
					id_vars=["Bruise Age (days)"],
					value_vars=["Type I-II", "Type III-IV", "Type V-VI"],
					var_name="Skin Tone",
					value_name="Detection Rate"
				)

				fig = px.line(
					df_melted,
					x="Bruise Age (days)",
					y="Detection Rate",
					color="Skin Tone",
					title="Detection Rate by Bruise Age and Skin Tone",
					markers=True,
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig.update_layout(
					height=350,
					margin=dict(l=20, r=20, t=40, b=20)
				)

				st.plotly_chart(fig, use_container_width=True)

			with dash_tab3:
				st.markdown("### Failure Analysis")

				# Sample failure mode data
				failure_modes = {
					"Failure Mode": [
						"Low Contrast",
						"Bruise Near Hair",
						"Dark Skin + Old Bruise",
						"Multiple Overlapping Bruises",
						"Tattoo Interference",
						"Skin Discoloration",
						"Shadow Effects"
					],
					"Type I-II": [15, 10, 5, 22, 18, 12, 18],
					"Type III-IV": [18, 12, 10, 20, 15, 15, 10],
					"Type V-VI": [35, 8, 22, 15, 12, 5, 3]
				}

				failure_df = pd.DataFrame(failure_modes)

				# Melt the dataframe for plotting
				failure_melted = pd.melt(
					failure_df,
					id_vars=["Failure Mode"],
					value_vars=["Type I-II", "Type III-IV", "Type V-VI"],
					var_name="Skin Tone",
					value_name="Percentage"
				)

				fig = px.bar(
					failure_melted,
					x="Failure Mode",
					y="Percentage",
					color="Skin Tone",
					title="Failure Mode Analysis by Skin Tone",
					barmode="group",
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig.update_layout(
					height=400,
					margin=dict(l=20, r=20, t=40, b=20)
				)

				st.plotly_chart(fig, use_container_width=True)

				# Recommendations based on failure analysis
				st.markdown("""
				<div class="highlight-text">
				<b>Failure Analysis Insights:</b>

				1. <b>Key Issue:</b> Low contrast bruises on Type V-VI skin is the dominant failure mode (35%)

				2. <b>Recommendations:</b>
				   - Enhance ALS preprocessing specifically for dark skin
				   - Collect additional data focusing on low-contrast bruises
				   - Implement specialized detection model for this specific case
				   - Consider dual-wavelength ALS imaging (415nm + 450nm)

				3. <b>Secondary Priority:</b> Improve detection of old bruises on dark skin

				4. <b>Note:</b> Tattoo interference affects all skin tones similarly, suggesting this is not a fairness issue but a general detection challenge
				</div>
				""", unsafe_allow_html=True)

				# Sample misclassification examples
				st.markdown("### Misclassification Examples")

				# Create two columns for examples
				miss_col1, miss_col2 = st.columns(2)

				with miss_col1:
					st.markdown("#### Example 1: Low Contrast Bruise (Type V)")

					# Generate a synthetic "failure case"
					img_size = 300
					dark_skin = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([90, 60, 50], dtype=np.uint8)

					# Add some natural skin texture
					texture = np.random.normal(0, 5, (img_size, img_size, 3))
					dark_skin = np.clip(dark_skin + texture, 0, 255).astype(np.uint8)

					# Add a very faint bruise
					center_x, center_y = img_size // 2, img_size // 2
					radius = 60

					bruise_mask = np.zeros((img_size, img_size))
					for i in range(img_size):
						for j in range(img_size):
							dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
							if dist < radius:
								# Gradient falloff for natural appearance
								bruise_mask[i, j] = max(0, 1 - (dist / radius) ** 2)

					# Apply very subtle bruise coloration
					bruised_img = dark_skin.copy()
					visibility_factor = 0.2

					bruised_img[:, :, 0] = np.clip(dark_skin[:, :, 0] - 10 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)
					bruised_img[:, :, 1] = np.clip(dark_skin[:, :, 1] - 5 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)
					bruised_img[:, :, 2] = np.clip(dark_skin[:, :, 2] - 5 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)

					st.image(bruised_img, use_column_width=True)

					st.markdown("""
					**Issue:** Minimal contrast between bruise and surrounding tissue.

					**Solution:** Dual-wavelength ALS imaging with specialized channel enhancement.
					""")

				with miss_col2:
					st.markdown("#### Example 2: Tattoo Misclassification")

					# Generate a synthetic "tattoo" case
					img_size = 300
					medium_skin = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([180, 140, 120], dtype=np.uint8)

					# Add some natural skin texture
					texture = np.random.normal(0, 8, (img_size, img_size, 3))
					medium_skin = np.clip(medium_skin + texture, 0, 255).astype(np.uint8)

					# Add a tattoo-like pattern that could be confused with a bruise
					tattoo_img = medium_skin.copy()

					# Draw a simple tribal-like tattoo pattern
					for i in range(100, 200):
						for j in range(100, 200):
							# Create a pattern that might be confused with a bruise
							if ((i - 150)**2 + (j - 150)**2 < 40**2) and not ((i - 150)**2 + (j - 150)**2 < 25**2):
								tattoo_img[i, j] = [40, 40, 40]  # Dark tattoo ink

					st.image(tattoo_img, use_column_width=True)

					st.markdown("""
					**Issue:** Tattoo pattern misclassified as bruise.

					**Solution:** Enhanced model training with tattoo examples and spectral analysis (tattoos and bruises have different spectral signatures under ALS).
					""")

			# Dashboard guidance
			st.markdown("""
			### Integrating the Fairness Dashboard

			This fairness monitoring dashboard would be integrated into the EAS-ID platform's development and deployment workflow:

			1. **During Development**: Track fairness metrics across iterative model improvements
			2. **In Clinical Testing**: Monitor performance across different clinical sites and patient demographics
			3. **Post-Deployment**: Continuous monitoring for performance drift or emergent bias
			4. **For Stakeholders**: Transparent reporting of system performance to clinical and community partners

			The dashboard would support the NIH AIM-AHEAD initiative's goals for AI health equity by providing transparent monitoring of algorithm performance across diverse populations.
			""")

	def render_data_engineering_page(self):
		st.markdown('<div class="main-header">Data Engineering for Bruise Detection</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"We capture HL7-FHIR bundles — outline your database schema."**

		The interviewers are looking for the ability to design scalable, secure clinical data stores.
		""")

		# Create tabs for different aspects of data engineering
		d_tab1, d_tab2, d_tab3 = st.tabs(["Database Schema", "Data Security", "FHIR Integration"])

		with d_tab1:
			st.markdown('<div class="sub-header">Proposed Database Schema</div>', unsafe_allow_html=True)

			st.markdown("""
			For the EAS-ID platform, I propose a comprehensive database schema that integrates HL7-FHIR standards with the specialized requirements of forensic bruise imaging:
			""")

			# FHIR-based schema diagram
			st.markdown('<div class="section-header">FHIR-Based Schema Design</div>', unsafe_allow_html=True)

			# Create a visual schema diagram
			fig = go.Figure()

			# Define entities
			entities = {
				"Patient": {"x": 1, "y": 3, "type": "FHIR"},
				"Practitioner": {"x": 3, "y": 1, "type": "FHIR"},
				"DiagnosticReport": {"x": 3, "y": 3, "type": "FHIR"},
				"ImagingStudy": {"x": 5, "y": 3, "type": "FHIR"},
				"Observation": {"x": 3, "y": 5, "type": "FHIR"},
				"Media": {"x": 5, "y": 5, "type": "FHIR"},
				"BruiseImage": {"x": 7, "y": 5, "type": "Custom"},
				"DetectionResult": {"x": 7, "y": 3, "type": "Custom"},
				"AuditEvent": {"x": 5, "y": 1, "type": "FHIR"},
			}

			# Define relationships
			relationships = [
				("Patient", "DiagnosticReport"),
				("Practitioner", "DiagnosticReport"),
				("DiagnosticReport", "ImagingStudy"),
				("DiagnosticReport", "Observation"),
				("ImagingStudy", "Media"),
				("Media", "BruiseImage"),
				("BruiseImage", "DetectionResult"),
				("ImagingStudy", "DetectionResult"),
				("Practitioner", "AuditEvent"),
				("DiagnosticReport", "AuditEvent")
			]

			# Colors for different entity types
			colors = {
				"FHIR": "#2196F3",  # Blue for standard FHIR resources
				"Custom": "#4CAF50"  # Green for custom extensions
			}

			# Plot entities
			for entity, attrs in entities.items():
				fig.add_trace(go.Scatter(
					x=[attrs["x"]],
					y=[attrs["y"]],
					mode="markers+text",
					marker=dict(size=50, color=colors[attrs["type"]]),
					text=[entity],
					textposition="middle center",
					textfont=dict(color="white", size=10),
					name=entity
				))

			# Plot relationships
			for rel in relationships:
				fig.add_trace(go.Scatter(
					x=[entities[rel[0]]["x"], entities[rel[1]]["x"]],
					y=[entities[rel[0]]["y"], entities[rel[1]]["y"]],
					mode="lines",
					line=dict(width=1, color="gray"),
					showlegend=False
				))

			# Update layout
			fig.update_layout(
				title="FHIR-Based Database Schema for Bruise Detection",
				showlegend=False,
				height=500,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
			)

			# Legend
			fig.add_trace(go.Scatter(
				x=[1], y=[1],
				mode="markers+text",
				marker=dict(size=15, color=colors["FHIR"]),
				text=["Standard FHIR Resource"],
				textposition="middle right",
				showlegend=False
			))

			fig.add_trace(go.Scatter(
				x=[1], y=[0.7],
				mode="markers+text",
				marker=dict(size=15, color=colors["Custom"]),
				text=["Custom Extension"],
				textposition="middle right",
				showlegend=False
			))

			st.plotly_chart(fig, use_container_width=True)

			# Schema details
			st.markdown("""
			### Core FHIR Resources

			1. **Patient**
			   - Standard FHIR Patient resource
			   - Extension: FitzpatrickSkinType (integer 1-6)
			   - Extension: ForensicCase (boolean)

			2. **Practitioner**
			   - Standard FHIR Practitioner resource
			   - Extension: ForensicCertification (CodeableConcept)

			3. **DiagnosticReport**
			   - Links findings and images
			   - References ImagingStudy and Observation resources
			   - Extension: ForensicDocumentation (boolean)

			4. **ImagingStudy**
			   - Study metadata (time, modality, etc.)
			   - References to Media resources
			   - Extension: ALSParameter (wavelength, filter settings)

			5. **Observation**
			   - Clinical findings (bruise measurements, age estimation)
			   - References to Media resources
			   - Extension: BruiseCharacteristics (size, color, pattern)

			6. **Media**
			   - Standard FHIR Media resource
			   - Links to actual image content
			   - Extension: ImagingParameters (device settings)

			7. **AuditEvent**
			   - Comprehensive logging for all access events
			   - Critical for chain-of-custody in forensic cases
			""")

			# Custom extensions
			st.markdown("""
			### Custom Extensions

			1. **BruiseImage**
			   - Extends Media resource
			   - Stores image binary data or secure reference
			   - Multiple capture types (white light, ALS variants)
			   - Technical metadata (resolution, format, calibration data)
			   - Image enhancement history

			2. **DetectionResult**
			   - AI detection outputs
			   - Segmentation maps (referenced as Media)
			   - Confidence scores
			   - Classification outputs (bruise age, type, pattern)
			   - Model version reference
			   - Extension: SkinTonePerformance (fairness metrics)
			""")

			# Database technology choices
			st.markdown('<div class="section-header">Database Technology Choices</div>', unsafe_allow_html=True)

			# Create columns for comparison
			col1, col2, col3 = st.columns(3)

			with col1:
				st.markdown("""
				### Primary Data Store

				**Document Database (MongoDB)**

				**Advantages:**
				- Native JSON structure aligns with FHIR
				- Schema flexibility for extensions
				- Horizontal scaling for large datasets
				- Geo-distributed replication

				**Implementation:**
				- FHIR-validated JSON documents
				- Sharded by patient/case
				- Time-series optimized collections for sensor data
				""")

			with col2:
				st.markdown("""
				### Image Storage

				**Hybrid Solution**

				**Components:**
				- Secure object store (S3-compatible)
				- Metadata in primary database
				- On-premise cache for active cases
				- Cold storage for archived cases

				**Features:**
				- Versioning and immutability
				- Geographic redundancy
				- Lifecycle policies (hot→cold transition)
				- Compliant encryption
				""")

			with col3:
				st.markdown("""
				### Analytics Backend

				**Column Store (PostgreSQL + TimescaleDB)**

				**Purpose:**
				- Performance metrics analysis
				- Fairness monitoring
				- Audit trail and compliance
				- ML model performance tracking

				**Implementation:**
				- Denormalized for query performance
				- Retention policies aligned with regulations
				- Real-time dashboards
				- HIPAA-compliant logging
				""")

			# Data lifecycle and partitioning
			st.markdown('<div class="section-header">Data Lifecycle Management</div>', unsafe_allow_html=True)

			st.markdown("""
			For optimal performance and security, the database implements a tiered data lifecycle strategy:

			1. **Hot Storage (0-30 days)**
			   - Active cases under investigation
			   - Full data access for authorized users
			   - Multi-region redundancy
			   - Real-time AI processing and analytics

			2. **Warm Storage (30-90 days)**
			   - Recently closed cases
			   - Slightly reduced performance
			   - Encrypted at rest and in transit
			   - Available for immediate recall if case reopened

			3. **Cold Storage (90+ days)**
			   - Archived cases
			   - Metadata maintained in primary system
			   - Images moved to cold storage with full chain-of-custody
			   - Compliant with evidence retention requirements

			4. **Research Tier (De-identified)**
			   - De-identified data for research
			   - Separate permission structure
			   - Synthetic data generation for AI training
			   - Documented consent and IRB approval tracking
			""")

		with d_tab2:
			st.markdown('<div class="sub-header">Security and Compliance Architecture</div>', unsafe_allow_html=True)

			st.markdown("""
			The EAS-ID platform requires a comprehensive security architecture to protect sensitive patient data, maintain forensic integrity, and ensure regulatory compliance:
			""")

			# Create columns for security aspects
			sec_col1, sec_col2 = st.columns(2)

			with sec_col1:
				st.markdown("""
				### Data Protection Requirements

				1. **Regulatory Compliance**
				   - HIPAA/HITECH for PHI
				   - 21 CFR Part 11 for forensic evidence
				   - GDPR principles for international alignment
				   - State-specific breach notifications

				2. **Clinical Security Standards**
				   - NIST Cybersecurity Framework
				   - HITRUST CSF certification targets
				   - HL7 FHIR security implementation guide
				   - Zero Trust Architecture principles

				3. **Forensic Evidence Requirements**
				   - Chain of custody documentation
				   - Digital signature verification
				   - Tamper-evident storage
				   - Legal admissibility considerations
				""")

			with sec_col2:
				st.markdown("""
				### Key Security Controls

				1. **Data Protection**
				   - Encryption at rest (AES-256)
				   - Encryption in transit (TLS 1.3)
				   - Field-level encryption for PHI
				   - Secure key management (HSM-backed)

				2. **Access Control**
				   - Role-based access control (RBAC)
				   - Attribute-based access control (ABAC)
				   - Multi-factor authentication
				   - Just-in-time access provisioning

				3. **Auditing and Monitoring**
				   - Comprehensive access logging
				   - Immutable audit trails
				   - Real-time threat monitoring
				   - Automated compliance reporting
				""")

			# Security architecture diagram
			st.markdown('<div class="section-header">Multi-Layer Security Architecture</div>', unsafe_allow_html=True)

			# Create a visual security diagram
			fig = go.Figure()

			# Define the security layers
			layers = [
				{"name": "Application Layer", "y": 6, "components": ["API Gateway", "Authentication", "Authorization"]},
				{"name": "Data Security Layer", "y": 5, "components": ["Encryption", "Tokenization", "Data Loss Prevention"]},
				{"name": "Database Layer", "y": 4, "components": ["Query Filtering", "Row-Level Security", "Audit Logging"]},
				{"name": "Storage Layer", "y": 3, "components": ["Encrypted Storage", "Secure Backup", "WORM Storage"]},
				{"name": "Network Layer", "y": 2, "components": ["Firewall", "Micro-segmentation", "VPN"]},
				{"name": "Monitoring Layer", "y": 1, "components": ["SIEM", "Threat Detection", "Compliance Alerts"]}
			]

			# Plot layers
			max_x = 6
			for layer in layers:
				# Layer box
				fig.add_shape(
					type="rect",
					x0=0, y0=layer["y"] - 0.4, x1=max_x, y1=layer["y"] + 0.4,
					line=dict(color="rgba(30, 136, 229, 0.5)"),
					fillcolor="rgba(30, 136, 229, 0.2)"
				)

				# Layer name
				fig.add_trace(go.Scatter(
					x=[0.2],
					y=[layer["y"]],
					mode="text",
					text=[layer["name"]],
					textposition="middle left",
					textfont=dict(size=12, color="#333"),
					showlegend=False
				))

				# Layer components
				component_positions = np.linspace(1.5, max_x - 0.5, len(layer["components"]))
				for i, comp in enumerate(layer["components"]):
					fig.add_shape(
						type="rect",
						x0=component_positions[i] - 0.4, y0=layer["y"] - 0.3,
						x1=component_positions[i] + 0.4, y1=layer["y"] + 0.3,
						line=dict(color="rgba(0, 102, 51, 0.8)"),
						fillcolor="rgba(0, 102, 51, 0.6)"
					)

					fig.add_trace(go.Scatter(
						x=[component_positions[i]],
						y=[layer["y"]],
						mode="text",
						text=[comp],
						textposition="middle center",
						textfont=dict(size=10, color="white"),
						showlegend=False
					))

			# Add connections between layers
			for i in range(len(layers) - 1):
				fig.add_shape(
					type="line",
					x0=3, y0=layers[i]["y"] - 0.4,
					x1=3, y1=layers[i+1]["y"] + 0.4,
					line=dict(color="rgba(100, 100, 100, 0.5)", width=1, dash="dot")
				)

			# Update layout
			fig.update_layout(
				title="Multi-Layer Security Architecture",
				showlegend=False,
				height=500,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, max_x + 0.5]),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 7])
			)

			st.plotly_chart(fig, use_container_width=True)

			# De-identification approach
			st.markdown('<div class="section-header">De-identification Strategy</div>', unsafe_allow_html=True)

			st.markdown("""
			For research datasets and model training, the EAS-ID platform implements a robust de-identification strategy:

			1. **Direct Identifiers**
			   - Removal of all 18 HIPAA identifiers
			   - Cryptographic hashing of patient IDs
			   - Replacement of location data with statistical regions

			2. **Clinical Images**
			   - Removal of DICOM metadata identifiers
			   - Cropping to region of interest only
			   - Pixel-level modifications to remove identifiable features
			   - Synthetic data generation for underrepresented cases

			3. **Research Safeguards**
			   - Statistical disclosure control techniques
			   - K-anonymity preservation (k ≥ 5)
			   - Differential privacy implementation for aggregated reports
			   - Re-identification risk assessment before data release

			4. **Implementation Technologies**
			   - FHIR-specific anonymization profiles
			   - Automated PHI detection and redaction
			   - Blockchain-based consent tracking
			   - Secure multi-party computation for collaborative research
			""")

			# Chain of custody
			st.markdown('<div class="section-header">Forensic Chain of Custody</div>', unsafe_allow_html=True)

			st.markdown("""
			<div class="highlight-text">
			<b>Chain of Custody Implementation</b>

			For forensic cases, the system implements a cryptographically secure chain of custody:

			1. <b>Digital Signatures</b>: All images and data are cryptographically signed when entered into the system

			2. <b>Immutable Audit Trail</b>: Every access, modification, and transmission is logged in an append-only, tamper-evident ledger

			3. <b>Cryptographic Integrity</b>: Content-addressable storage with hash verification ensures no tampering

			4. <b>Access Timestamping</b>: Trusted timestamping service provides legal verification of when records were accessed

			5. <b>Export Verification</b>: Exported evidence includes cryptographic proof of authenticity and completeness

			This approach ensures the EAS-ID platform can produce legally admissible evidence that meets Daubert standard requirements.
			</div>
			""", unsafe_allow_html=True)

		with d_tab3:
			st.markdown('<div class="sub-header">FHIR Integration Architecture</div>', unsafe_allow_html=True)

			st.markdown("""
			The EAS-ID platform uses FHIR (Fast Healthcare Interoperability Resources) as its core data model to ensure seamless integration with existing clinical systems:
			""")

			# Create columns for FHIR examples
			fhir_col1, fhir_col2 = st.columns([3, 2])

			with fhir_col1:
				st.markdown("""
				### FHIR Integration Approach

				1. **Standards Compliance**
				   - FHIR R4 implementation (latest stable version)
				   - US Core Implementation Guide conformance
				   - Structured data exchange using JSON format
				   - OpenID Connect + OAuth 2.0 security framework

				2. **Key FHIR Resources Utilized**
				   - Patient: Patient demographics and identifiers
				   - Practitioner: Healthcare provider information
				   - ImagingStudy: Imaging session metadata
				   - Media: Image and multimedia content
				   - DiagnosticReport: Findings and interpretation
				   - Observation: Structured measurements and findings
				   - Bundle: Packaged resource collections

				3. **Custom Extensions**
				   - FitzpatrickSkinType: For skin tone classification
				   - ALSParameters: For alternate light source settings
				   - AIModelVersion: For tracking algorithm versions
				   - ForensicMetadata: For chain-of-custody information

				4. **Integration Patterns**
				   - RESTful API following FHIR specification
				   - SMART on FHIR for app integration
				   - Bulk Data Access API for research datasets
				   - HL7 v2 to FHIR conversion for legacy systems
				""")

			with fhir_col2:
				# Display a sample FHIR resource
				st.markdown("### Sample FHIR Resource")

				# Create a sample FHIR DiagnosticReport JSON
				fhir_json = {
					"resourceType": "DiagnosticReport",
					"id": "bruise-detection-report-1",
					"status": "final",
					"category": [
						{
							"coding": [
								{
									"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
									"code": "IMG",
									"display": "Imaging"
								}
							]
						}
					],
					"code": {
						"coding": [
							{
								"system": "http://loinc.org",
								"code": "86184-5",
								"display": "Forensic injury evaluation"
							}
						],
						"text": "Bruise Detection Analysis"
					},
					"subject": {
						"reference": "Patient/example-patient-id",
						"display": "Anonymous Patient"
					},
					"effectiveDateTime": "2025-05-10T14:30:00-05:00",
					"issued": "2025-05-10T16:45:00-05:00",
					"performer": [
						{
							"reference": "Practitioner/example-practitioner-id",
							"display": "Dr. Katherine Scafide"
						}
					],
					"result": [
						{
							"reference": "Observation/bruise-observation-1"
						}
					],
					"imagingStudy": [
						{
							"reference": "ImagingStudy/bruise-imaging-study-1"
						}
					],
					"media": [
						{
							"comment": "White light image of bruise",
							"link": {
								"reference": "Media/bruise-image-whitelight-1"
							}
						},
						{
							"comment": "ALS 415nm image of bruise",
							"link": {
								"reference": "Media/bruise-image-als415-1"
							}
						}
					],
					"conclusion": "Multiple bruises detected on left forearm, consistent with defensive injuries. Enhanced visibility under ALS illumination.",
					"extension": [
						{
							"url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensic-documentation",
							"valueBoolean": True
						},
						{
							"url": "https://bruise.gmu.edu/fhir/StructureDefinition/skin-tone",
							"valueInteger": 5
						},
						{
							"url": "https://bruise.gmu.edu/fhir/StructureDefinition/ai-model-version",
							"valueString": "EAS-ID-Model-v2.1.3"
						}
					]
				}

				# Display the JSON with formatting
				st.json(fhir_json)

			# FHIR integration architecture
			st.markdown('<div class="section-header">FHIR Integration Architecture</div>', unsafe_allow_html=True)

			# Create a visual architecture diagram
			fig = go.Figure()

			# Define architecture components
			components = {
				"EHR Systems": {"x": 1, "y": 1, "type": "External"},
				"FHIR Server": {"x": 3, "y": 2, "type": "Core"},
				"EAS-ID API": {"x": 5, "y": 2, "type": "Core"},
				"Mobile App": {"x": 7, "y": 1, "type": "Client"},
				"AI Pipeline": {"x": 5, "y": 4, "type": "Core"},
				"Image Repository": {"x": 3, "y": 4, "type": "Core"},
				"Analytics": {"x": 7, "y": 3, "type": "Support"},
				"Research Portal": {"x": 7, "y": 5, "type": "Support"}
			}

			# Define connections
			connections = [
				("EHR Systems", "FHIR Server", "HL7 FHIR"),
				("FHIR Server", "EAS-ID API", "RESTful API"),
				("EAS-ID API", "Mobile App", "SMART on FHIR"),
				("EAS-ID API", "AI Pipeline", "Internal API"),
				("AI Pipeline", "Image Repository", "Secure Access"),
				("FHIR Server", "Image Repository", "DICOMweb"),
				("EAS-ID API", "Analytics", "Event Stream"),
				("Image Repository", "Research Portal", "De-identified Access")
			]

			# Colors for different component types
			colors = {
				"External": "#FF9800",  # Orange
				"Core": "#2196F3",      # Blue
				"Client": "#4CAF50",    # Green
				"Support": "#9C27B0"    # Purple
			}

			# Plot components
			for comp, attrs in components.items():
				fig.add_trace(go.Scatter(
					x=[attrs["x"]],
					y=[attrs["y"]],
					mode="markers+text",
					marker=dict(size=40, color=colors[attrs["type"]]),
					text=[comp],
					textposition="middle center",
					textfont=dict(color="white", size=10),
					name=comp
				))

			# Plot connections
			for conn in connections:
				fig.add_trace(go.Scatter(
					x=[components[conn[0]]["x"], components[conn[1]]["x"]],
					y=[components[conn[0]]["y"], components[conn[1]]["y"]],
					mode="lines+text",
					line=dict(width=1, color="gray"),
					text=[conn[2]],
					textposition="middle center",
					textfont=dict(size=8, color="#555"),
					showlegend=False
				))

			# Update layout
			fig.update_layout(
				title="FHIR Integration Architecture",
				showlegend=False,
				height=500,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
			)

			# Legend
			for i, (type_name, color) in enumerate(colors.items()):
				fig.add_trace(go.Scatter(
					x=[1], y=[6 - i * 0.3],
					mode="markers+text",
					marker=dict(size=10, color=color),
					text=[type_name],
					textposition="middle right",
					showlegend=False
				))

			st.plotly_chart(fig, use_container_width=True)

			# FHIR implementation best practices
			st.markdown('<div class="section-header">FHIR Implementation Best Practices</div>', unsafe_allow_html=True)

			st.markdown("""
			To ensure robust and compliant FHIR implementation, the EAS-ID platform follows these best practices:

			1. **API Design**
			   - Full implementation of CRUD operations for all resources
			   - Support for _include and _revinclude parameters
			   - Consistent error handling with OperationOutcome
			   - Versioning support with ETag and If-Match headers

			2. **Terminology Binding**
			   - Use of standard LOINC codes for observations
			   - SNOMED CT for clinical findings
			   - DICOM for imaging procedures
			   - Custom code systems for project-specific concepts

			3. **Data Validation**
			   - Schema validation against FHIR structure definitions
			   - Business rule validation for clinical coherence
			   - Referential integrity checks
			   - Terminology validation against value sets

			4. **Performance Optimization**
			   - Efficient search parameter implementation
			   - Bundle transactions for atomic operations
			   - Bulk data operations for research use
			   - Caching and compression strategies
			""")

			# FHIR adoption challenges
			st.markdown("""
			<div class="highlight-text">
			<b>Addressing FHIR Adoption Challenges</b>

			The EAS-ID platform addresses common FHIR integration challenges:

			1. <b>Legacy System Integration</b>
			   - HL7 v2 to FHIR conversion layer
			   - CDA to FHIR transformation
			   - PDF report extraction to structured data

			2. <b>Data Mapping Complexity</b>
			   - Automated mapping tools with validation
			   - Terminology services for code translation
			   - Content negotiation for flexible formats

			3. <b>Performance at Scale</b>
			   - Distributed FHIR server architecture
			   - Specialized indexing for image metadata
			   - Optimization for high-volume write operations

			4. <b>Security Considerations</b>
			   - Granular consent management
			   - Resource-level access control
			   - Secure context propagation
			</div>
			""", unsafe_allow_html=True)

	def render_deployment_page(self):
		st.markdown('<div class="main-header">Mobile Deployment Strategy</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"Inference on-device or in the cloud—convince us."**

		The interviewers are looking for systems-level thinking and understanding of privacy-latency trade-offs.
		""")

		# Create tabs for different aspects of deployment
		m_tab1, m_tab2, m_tab3 = st.tabs(["Deployment Options", "Hybrid Approach", "Implementation"])

		with m_tab1:
			st.markdown('<div class="sub-header">Deployment Model Comparison</div>', unsafe_allow_html=True)

			st.markdown("""
			For the EAS-ID bruise detection platform, choosing between on-device and cloud inference requires careful consideration of multiple factors:
			""")

			# Create comparison table
			comparison_data = {
				"Factor": [
					"Latency",
					"Privacy",
					"Bandwidth Requirements",
					"Compute Power",
					"Battery Impact",
					"Model Size",
					"Update Frequency",
					"Offline Capability",
					"Regulatory Compliance",
					"Implementation Complexity"
				],
				"On-Device": [
					"Low (real-time)",
					"High (data stays local)",
					"Low (no image upload)",
					"Limited (mobile constraints)",
					"High (local processing)",
					"Constrained (<100MB)",
					"Requires app updates",
					"Full capability",
					"Simpler (no PHI transmission)",
					"Higher (device fragmentation)"
				],
				"Cloud-Based": [
					"Higher (network dependent)",
					"Lower (data transmission)",
					"High (full images)",
					"Unlimited (scalable)",
					"Lower (offloaded compute)",
					"Unlimited",
					"Continuous deployment",
					"Requires connectivity",
					"More complex (data transfer)",
					"Lower (centralized)"
				]
			}

			# Convert to DataFrame
			comparison_df = pd.DataFrame(comparison_data)

			# Style the table
			styled_df = comparison_df.style.set_properties(**{
				'background-color': '#f0f7f0',
				'border-color': '#dddddd',
				'border-style': 'solid',
				'border-width': '1px',
				'text-align': 'left',
				'padding': '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color', 'white'),
					('font-weight', 'bold'),
					('border', '1px solid #dddddd'),
					('text-align', 'left'),
					('padding', '8px')
				]}
			])

			# Display the table
			st.table(styled_df)

			# Decision factors visualization
			st.markdown('<div class="section-header">Decision Factors Analysis</div>', unsafe_allow_html=True)

			# Create radar chart for comparing approaches
			categories = [
				'Privacy', 'Latency', 'Compute Power',
				'Battery Life', 'Bandwidth Efficiency',
				'Offline Capability', 'Updateability',
				'Regulatory Compliance', 'Cost Efficiency'
			]

			# Values for each approach (0-5 scale)
			on_device_values = [5, 5, 2, 2, 5, 5, 2, 4, 3]
			cloud_values = [2, 3, 5, 4, 2, 1, 5, 3, 4]
			hybrid_values = [4, 4, 4, 3, 4, 4, 4, 4, 3]

			# Add the first value at the end to close the loop
			categories = categories + [categories[0]]
			on_device_values = on_device_values + [on_device_values[0]]
			cloud_values = cloud_values + [cloud_values[0]]
			hybrid_values = hybrid_values + [hybrid_values[0]]

			# Convert to radians
			angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
			angles += angles[:1]  # Close the loop

			# Create the radar chart
			fig = go.Figure()

			fig.add_trace(go.Scatterpolar(
				r=on_device_values,
				theta=categories,
				fill='toself',
				name='On-Device',
				line_color='#4CAF50'
			))

			fig.add_trace(go.Scatterpolar(
				r=cloud_values,
				theta=categories,
				fill='toself',
				name='Cloud-Based',
				line_color='#2196F3'
			))

			fig.add_trace(go.Scatterpolar(
				r=hybrid_values,
				theta=categories,
				fill='toself',
				name='Hybrid Approach',
				line_color='#FF9800'
			))

			fig.update_layout(
				polar=dict(
					radialaxis=dict(
						visible=True,
						range=[0, 5]
					)
				),
				showlegend=True,
				height=500,
				margin=dict(l=80, r=80, t=20, b=20)
			)

			st.plotly_chart(fig, use_container_width=True)

			# Clinical workflow considerations
			st.markdown('<div class="section-header">Clinical Workflow Considerations</div>', unsafe_allow_html=True)

			st.markdown("""
			Beyond technical considerations, the deployment model must address the specific clinical context of forensic bruise documentation:

			1. **Examination Setting Variability**
			   - Emergency departments (stable connectivity)
			   - Field examinations (unreliable connectivity)
			   - Rural healthcare settings (limited bandwidth)
			   - International deployments (varied infrastructure)

			2. **Workload Characteristics**
			   - Batch processing (multiple images in sequence)
			   - Time-sensitive documentation
			   - Follow-up examinations comparing to baseline
			   - Multi-examiner collaboration

			3. **User Requirements**
			   - Nurse-friendly UI with immediate feedback
			   - Minimal training requirements
			   - Resilience to user error
			   - Continuity between sessions

			4. **Legal and Ethical Considerations**
			   - Court admissibility requirements
			   - Chain of custody preservation
			   - Cross-jurisdictional compliance
			   - Ethical data usage for model improvement
			""")

		with m_tab2:
			st.markdown('<div class="sub-header">Proposed Hybrid Approach</div>', unsafe_allow_html=True)

			st.markdown("""
			After analyzing the requirements and constraints, I recommend a **tiered hybrid approach** that leverages the strengths of both on-device and cloud processing:
			""")

			# Hybrid architecture diagram
			st.markdown('<div class="section-header">Tiered Hybrid Architecture</div>', unsafe_allow_html=True)

			# Create a visual architecture diagram
			fig = go.Figure()

			# Define architecture components
			arch_components = [
				{"name": "Mobile Device", "layer": "Client", "x": 1, "components": [
					"Image Capture", "ALS Control", "Preprocessing", "Triage Model"
				]},
				{"name": "Edge Compute", "layer": "Edge", "x": 3, "components": [
					"Initial Segmentation", "Confidence Scoring", "Temp Storage", "Compression"
				]},
				{"name": "Cloud", "layer": "Cloud", "x": 5, "components": [
					"Advanced Models", "Multi-image Analysis", "Clinical Integration", "Research Data"
				]}
			]

			# Layer heights
			layer_heights = {
				"Client": 1,
				"Edge": 3,
				"Cloud": 5
			}

			# Component colors
			layer_colors = {
				"Client": "#4CAF50",  # Green
				"Edge": "#FF9800",    # Orange
				"Cloud": "#2196F3"    # Blue
			}

			# Plot main components
			for comp in arch_components:
				# Component box
				fig.add_shape(
					type="rect",
					x0=comp["x"] - 0.9, y0=layer_heights[comp["layer"]] - 0.9,
					x1=comp["x"] + 0.9, y1=layer_heights[comp["layer"]] + 0.9,
					line=dict(color=layer_colors[comp["layer"]]),
					fillcolor=f"rgba({int(layer_colors[comp['layer']][1:3], 16)}, {int(layer_colors[comp['layer']][3:5], 16)}, {int(layer_colors[comp['layer']][5:7], 16)}, 0.2)"
				)

				# Component name
				fig.add_trace(go.Scatter(
					x=[comp["x"]],
					y=[layer_heights[comp["layer"]] + 0.7],
					mode="text",
					text=[comp["name"]],
					textposition="middle center",
					textfont=dict(size=14, color="#333"),
					showlegend=False
				))

				# Sub-components
				for i, subcomp in enumerate(comp["components"]):
					y_pos = layer_heights[comp["layer"]] - 0.5 + i * 0.3

					fig.add_trace(go.Scatter(
						x=[comp["x"]],
						y=[y_pos],
						mode="text",
						text=[subcomp],
						textposition="middle center",
						textfont=dict(size=11, color="#333"),
						showlegend=False
					))

			# Add connections between layers
			# Mobile to Edge
			fig.add_shape(
				type="line",
				x0=1.9, y0=layer_heights["Client"],
				x1=2.1, y1=layer_heights["Edge"],
				line=dict(color="gray", width=2, dash="dot")
			)

			fig.add_trace(go.Scatter(
				x=[(1.9 + 2.1) / 2],
				y=[(layer_heights["Client"] + layer_heights["Edge"]) / 2],
				mode="text",
				text=["Wi-Fi/5G"],
				textposition="middle right",
				textfont=dict(size=10, color="#333"),
				showlegend=False
			))

			# Edge to Cloud
			fig.add_shape(
				type="line",
				x0=3.9, y0=layer_heights["Edge"],
				x1=4.1, y1=layer_heights["Cloud"],
				line=dict(color="gray", width=2, dash="dot")
			)

			fig.add_trace(go.Scatter(
				x=[(3.9 + 4.1) / 2],
				y=[(layer_heights["Edge"] + layer_heights["Cloud"]) / 2],
				mode="text",
				text=["Internet"],
				textposition="middle right",
				textfont=dict(size=10, color="#333"),
				showlegend=False
			))

			# Add data flow arrows
			# Image Capture Flow
			fig.add_annotation(
				x=1.5, y=layer_heights["Client"] + 0.3,
				ax=2.5, ay=layer_heights["Edge"] + 0.3,
				text="Images",
				showarrow=True,
				arrowhead=2,
				arrowsize=1,
				arrowwidth=1,
				arrowcolor="#333"
			)

			# Results Flow
			fig.add_annotation(
				x=2.5, y=layer_heights["Edge"] - 0.3,
				ax=1.5, ay=layer_heights["Client"] - 0.3,
				text="Results",
				showarrow=True,
				arrowhead=2,
				arrowsize=1,
				arrowwidth=1,
				arrowcolor="#333"
			)

			# Advanced Processing Flow
			fig.add_annotation(
				x=3.5, y=layer_heights["Edge"] + 0.3,
				ax=4.5, ay=layer_heights["Cloud"] + 0.3,
				text="Complex Cases",
				showarrow=True,
				arrowhead=2,
				arrowsize=1,
				arrowwidth=1,
				arrowcolor="#333"
			)

			# Model Updates Flow
			fig.add_annotation(
				x=4.5, y=layer_heights["Cloud"] - 0.3,
				ax=3.5, ay=layer_heights["Edge"] - 0.3,
				text="Model Updates",
				showarrow=True,
				arrowhead=2,
				arrowsize=1,
				arrowwidth=1,
				arrowcolor="#333"
			)

			# Update layout
			fig.update_layout(
				title="Tiered Hybrid Architecture for Bruise Detection",
				showlegend=False,
				height=400,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6]),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6])
			)

			st.plotly_chart(fig, use_container_width=True)

			# Explain the tiered approach
			st.markdown("""
			### Tier 1: On-Device Processing

			The mobile app performs these functions locally:

			1. **Image Acquisition**
			   - Camera control and ALS hardware integration
			   - Exposure and focus optimization for bruise visibility
			   - Image quality assessment

			2. **Preprocessing**
			   - Calibration against color reference markers
			   - Noise reduction and enhancement
			   - Metadata tagging (geolocation, timestamps)

			3. **Lightweight Detection**
			   - TensorFlow Lite model for initial bruise detection
			   - Optimized MobileNet-based architecture (<20MB)
			   - Preliminary segmentation with confidence scoring

			4. **Triage Logic**
			   - High-confidence cases processed entirely on-device
			   - Low-confidence or complex cases flagged for cloud processing
			   - Offline capability for field examinations
			""")

			st.markdown("""
			### Tier 2: Edge Computing (Optional)

			For clinical settings with reliable local network but limited internet connectivity:

			1. **Edge Server**
			   - On-premise or local network deployment
			   - Higher-capacity models than mobile (500MB-1GB)
			   - Capable of handling batch processing

			2. **Temporary Storage**
			   - HIPAA-compliant local cache
			   - Encrypted at rest
			   - Configurable retention policy

			3. **Preprocessing Offload**
			   - CPU/GPU intensive preprocessing tasks
			   - Multi-spectral image alignment
			   - Advanced noise reduction
			""")

			st.markdown("""
			### Tier 3: Cloud Processing

			Reserved for specific use cases:

			1. **Advanced Analysis**
			   - Full-scale models for complex cases
			   - Multi-image temporal analysis
			   - Integration with patient history

			2. **Model Training & Improvement**
			   - Federated learning from edge devices
			   - Performance monitoring across deployment sites
			   - New model version testing

			3. **Research Data Repository**
			   - De-identified data collection (with consent)
			   - Multi-institutional collaboration
			   - Model fairness analysis

			4. **Integration Services**
			   - EHR connectivity via FHIR
			   - Secure reporting to law enforcement systems
			   - Telemedicine consultation support
			""")

			# Decision algorithm
			st.markdown('<div class="section-header">Deployment Decision Algorithm</div>', unsafe_allow_html=True)

			st.markdown("""
			The system uses a decision algorithm to determine the optimal processing location for each image:

			1. **Initial Assessment**
			   - Image quality check
			   - Available device resources (memory, battery)
			   - Current network conditions

			2. **Confidence Thresholding**
			   - On-device model confidence score
			   - Complexity assessment (multiple bruises, unusual patterns)
			   - Skin tone consideration (darker skin tones may need enhanced processing)

			3. **Context Evaluation**
			   - Clinical urgency
			   - Available connectivity
			   - Battery status
			   - User preference setting

			4. **Adaptive Routing**
			   - Route to appropriate tier based on above factors
			   - Transparent to user with status indicators
			   - Option to override automatic decision
			""")

			# Benefits of hybrid approach
			st.markdown("""
			<div class="highlight-text">
			<b>Benefits of Hybrid Approach</b>

			The proposed hybrid approach offers significant advantages:

			1. <b>Resilience</b>: Functions across all connectivity scenarios

			2. <b>Optimized Performance</b>: Uses appropriate compute for each task

			3. <b>Privacy-Preserving</b>: Minimizes data transmission for routine cases

			4. <b>User-Centric</b>: Fast feedback for clinical workflow

			5. <b>Future-Proof</b>: Scales with improved mobile hardware and connectivity

			6. <b>Deployment Flexibility</b>: Configurable based on institutional requirements

			7. <b>Regulatory Compliance</b>: Adaptable to different regional requirements
			</div>
			""", unsafe_allow_html=True)

		with m_tab3:
			st.markdown('<div class="sub-header">Technical Implementation Plan</div>', unsafe_allow_html=True)

			st.markdown("""
			To implement the hybrid approach effectively, I propose the following technical implementation strategy:
			""")

			# Implementation plan visualization
			st.markdown('<div class="section-header">Implementation Roadmap</div>', unsafe_allow_html=True)

			# Create a Gantt chart for implementation
			tasks = [
				{"Task": "On-Device Model Development", "Start": 0, "Duration": 2, "Layer": "Mobile"},
				{"Task": "Mobile App UI Development", "Start": 1, "Duration": 3, "Layer": "Mobile"},
				{"Task": "Edge Computing Setup", "Start": 2, "Duration": 2, "Layer": "Edge"},
				{"Task": "Cloud Infrastructure Deployment", "Start": 1, "Duration": 2, "Layer": "Cloud"},
				{"Task": "API Development", "Start": 3, "Duration": 2, "Layer": "Integration"},
				{"Task": "Decision Algorithm Implementation", "Start": 4, "Duration": 1, "Layer": "Integration"},
				{"Task": "Security Implementation", "Start": 2, "Duration": 3, "Layer": "Security"},
				{"Task": "Offline Mode Testing", "Start": 5, "Duration": 1, "Layer": "Testing"},
				{"Task": "Clinical Validation", "Start": 6, "Duration": 3, "Layer": "Testing"},
				{"Task": "Deployment & Training", "Start": 8, "Duration": 2, "Layer": "Deployment"}
			]

			# Convert to DataFrame
			tasks_df = pd.DataFrame(tasks)

			# Create Gantt chart
			fig = px.timeline(
				tasks_df,
				x_start="Start",
				x_end=lambda x: x["Start"] + x["Duration"],
				y="Task",
				color="Layer",
				title="Hybrid Deployment Implementation Timeline (Months)",
				color_discrete_map={
					"Mobile": "#4CAF50",
					"Edge": "#FF9800",
					"Cloud": "#2196F3",
					"Integration": "#9C27B0",
					"Security": "#F44336",
					"Testing": "#795548",
					"Deployment": "#607D8B"
				}
			)

			fig.update_layout(
				height=400,
				margin=dict(l=20, r=20, t=40, b=20),
				xaxis_title="Months"
			)

			st.plotly_chart(fig, use_container_width=True)

			# Mobile implementation
			st.markdown('<div class="section-header">Mobile Implementation</div>', unsafe_allow_html=True)

			# Create columns for technical details
			mob_col1, mob_col2 = st.columns(2)

			with mob_col1:
				st.markdown("""
				### Model Optimization

				1. **Model Architecture**
				   - MobileNetV3 + UNet-style decoder
				   - Designed for resource efficiency
				   - INT8 quantization for 4x size reduction

				2. **Optimization Techniques**
				   - Weight pruning (30% parameter reduction)
				   - Knowledge distillation from larger models
				   - Layer fusion for inference speed

				3. **Runtime Optimization**
				   - TensorFlow Lite acceleration
				   - GPU delegate for compatible devices
				   - Neural Engine usage on iOS devices
				   - Thread pool optimization
				""")

			with mob_col2:
				st.markdown("""
				### Mobile App Design

				1. **Cross-Platform Framework**
				   - Flutter for consistent UI across platforms
				   - Native camera integration for ALS control
				   - Hardware acceleration for image processing

				2. **Offline Capability**
				   - SQLite database for local storage
				   - Sync mechanism with resume capability
				   - Forensic-grade data integrity

				3. **User Experience**
				   - Wizard-style guided image capture
				   - Real-time feedback on image quality
				   - Transparent processing status
				   - Nurse-oriented UI design
				""")

			# Edge and Cloud implementation
			st.markdown('<div class="section-header">Edge & Cloud Implementation</div>', unsafe_allow_html=True)

			# Create columns for technical details
			ec_col1, ec_col2 = st.columns(2)

			with ec_col1:
				st.markdown("""
				### Edge Computing Implementation

				1. **Hardware Requirements**
				   - Small form-factor server (NUC or similar)
				   - 32GB RAM, 8-core CPU
				   - Optional GPU acceleration
				   - 1TB encrypted storage

				2. **Software Stack**
				   - Docker containers for easy deployment
				   - TensorFlow Serving for model hosting
				   - NGINX for API gateway and load balancing
				   - Health monitoring and auto-restart

				3. **Network Configuration**
				   - Internal subnet isolation
				   - Secure Wi-Fi connectivity for mobile clients
				   - VPN tunnel to cloud services
				   - Bandwidth throttling for predictable performance
				""")

			with ec_col2:
				st.markdown("""
				### Cloud Architecture

				1. **Infrastructure**
				   - Kubernetes cluster for scalability
				   - Multi-region deployment for redundancy
				   - HIPAA-compliant cloud provider
				   - CI/CD pipeline for continuous updates

				2. **Advanced Model Deployment**
				   - Model versioning system
				   - A/B testing framework
				   - Monitoring for drift detection
				   - Auto-scaling based on demand

				3. **Integration Services**
				   - FHIR server for healthcare interoperability
				   - API gateway with rate limiting
				   - WebSocket for real-time status updates
				   - Bulk analytics processing
				""")

			# Bandwidth optimization
			st.markdown('<div class="section-header">Bandwidth Optimization</div>', unsafe_allow_html=True)

			st.markdown("""
			To minimize bandwidth requirements while preserving image quality:

			1. **Smart Compression**
			   - Lossless compression for region of interest
			   - Progressive JPEG for preview during upload
			   - WebP format for 30% smaller size than JPEG

			2. **Selective Transmission**
			   - Region of interest cropping before transmission
			   - On-device downsampling of non-critical areas
			   - Multi-resolution transmission pipeline

			3. **Intelligent Syncing**
			   - Priority-based upload queue
			   - Background synchronization when on Wi-Fi
			   - Resumable uploads for reliability
			   - Differential updates for follow-up images
			""")

			# Performance benchmarks
			st.markdown('<div class="section-header">Performance Benchmarks</div>', unsafe_allow_html=True)

			# Create sample benchmark data
			benchmark_data = {
				"Metric": [
					"Inference Time",
					"Battery Impact",
					"Bandwidth Usage",
					"Storage Requirements",
					"Offline Functionality"
				],
				"On-Device Only": [
					"2-5 seconds",
					"3.5% per image",
					"0 MB",
					"250 MB app size",
					"100%"
				],
				"Hybrid Approach": [
					"1-3 seconds (simple cases)",
					"2% per image",
					"0.5-5 MB per complex case",
					"150 MB app size",
					"80%"
				],
				"Cloud-Only": [
					"3-10 seconds (network dependent)",
					"1% per image",
					"5-10 MB per image",
					"50 MB app size",
					"0%"
				]
			}

			# Convert to DataFrame
			benchmark_df = pd.DataFrame(benchmark_data)

			# Style the table
			styled_df = benchmark_df.style.set_properties(**{
				'background-color': '#f0f7f0',
				'border-color': '#dddddd',
				'border-style': 'solid',
				'border-width': '1px',
				'text-align': 'left',
				'padding': '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color', 'white'),
					('font-weight', 'bold'),
					('border', '1px solid #dddddd'),
					('text-align', 'left'),
					('padding', '8px')
				]}
			])

			# Display the table
			st.table(styled_df)

			# Security considerations
			st.markdown("""
			<div class="highlight-text">
			<b>Security Implementation</b>

			The hybrid deployment includes comprehensive security measures:

			1. <b>Device Security</b>
			   - App-level encryption for local storage
			   - Biometric authentication for app access
			   - Device attestation before cloud access
			   - Secure camera access policies

			2. <b>Transmission Security</b>
			   - TLS 1.3 with certificate pinning
			   - JWT-based authentication
			   - Image watermarking for authenticity
			   - Encrypted metadata

			3. <b>Cloud Security</b>
			   - HIPAA-compliant infrastructure
			   - Regular penetration testing
			   - Comprehensive audit logging
			   - Automatic intrusion detection

			4. <b>Compliance Features</b>
			   - Configurable data retention policies
			   - Role-based access control
			   - Full chain of custody tracking
			   - Consent management system
			</div>
			""", unsafe_allow_html=True)

	def render_leadership_page(self):
		st.markdown('<div class="main-header">Leadership and Team Coordination</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"Tell us about a time you coordinated a heterogeneous team."**

		The interviewers are looking for readiness to supervise students and clinicians.
		""")

		# Create tabs for different aspects of leadership
		l_tab1, l_tab2, l_tab3 = st.tabs(["Team Management Framework", "Interdisciplinary Communication", "Mentorship Approach"])

		with l_tab1:
			st.markdown('<div class="sub-header">Team Management Framework</div>', unsafe_allow_html=True)

			st.markdown("""
			Based on my experience coordinating heterogeneous teams in healthcare AI development, I've developed a framework that would be applicable to the EAS-ID project:
			""")

			# Team structure visualization
			st.markdown('<div class="section-header">Proposed Team Structure</div>', unsafe_allow_html=True)

			# Create a visual team structure diagram
			fig = go.Figure()

			# Define team structure
			team_roles = [
				{"name": "Postdoc Lead", "x": 4, "y": 6, "group": "Core", "description": "Project coordination & technical leadership"},
				{"name": "PI/Faculty", "x": 4, "y": 8, "group": "Leadership", "description": "Strategic guidance & stakeholder management"},
				{"name": "Computer Vision Engineer", "x": 2, "y": 4, "group": "Technical", "description": "Deep learning model development"},
				{"name": "Mobile Developer", "x": 6, "y": 4, "group": "Technical", "description": "Mobile app & hardware integration"},
				{"name": "Database Specialist", "x": 4, "y": 4, "group": "Technical", "description": "FHIR integration & data management"},
				{"name": "Graduate Student (CS)", "x": 1, "y": 2, "group": "Students", "description": "Algorithm implementation & testing"},
				{"name": "Graduate Student (Nursing)", "x": 4, "y": 2, "group": "Students", "description": "Clinical validation & workflow design"},
				{"name": "Graduate Student (Engineering)", "x": 7, "y": 2, "group": "Students", "description": "Hardware integration & prototyping"},
				{"name": "Forensic Nurse Consultant", "x": 2, "y": 0, "group": "Clinical", "description": "Domain expertise & user testing"},
				{"name": "UX Researcher", "x": 4, "y": 0, "group": "Clinical", "description": "User-centered design & usability testing"},
				{"name": "Clinical Partner", "x": 6, "y": 0, "group": "Clinical", "description": "Field testing & feedback"}
			]

			# Define relationships (reporting lines)
			relationships = [
				("Postdoc Lead", "PI/Faculty"),
				("Computer Vision Engineer", "Postdoc Lead"),
				("Mobile Developer", "Postdoc Lead"),
				("Database Specialist", "Postdoc Lead"),
				("Graduate Student (CS)", "Computer Vision Engineer"),
				("Graduate Student (Nursing)", "Database Specialist"),
				("Graduate Student (Engineering)", "Mobile Developer"),
				("Forensic Nurse Consultant", "Graduate Student (Nursing)"),
				("UX Researcher", "Graduate Student (Nursing)"),
				("Clinical Partner", "Graduate Student (Nursing)")
			]

			# Add team members to the figure
			for member in team_roles:
				fig.add_trace(go.Scatter(x=[member['x']], y=[member['y']], mode='markers', name=member['name'],
					marker=dict(size=20, color='blue', opacity=0.7),
					text=member['description'],
					textposition='bottom center'))

			# Add relationships to the figure
			for relationship in relationships:
				source = next((member for member in team_roles if member['name'] == relationship[0]), None)
				target = next((member for member in team_roles if member['name'] == relationship[1]), None)
				if source and target:
					fig.add_trace(go.Scatter(x=[source['x'], target['x']], y=[source['y'], target['y']], mode='lines',
						line=dict(color='gray', width=1),
						showlegend=False))

			# Update layout
			fig.update_layout(
				showlegend=False,
				plot_bgcolor='white',
				margin=dict(l=50, r=50, t=50, b=50),
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
			)

			# Display the figure
			st.plotly_chart(fig, use_container_width=True)

			# Create columns for management principles
			col1, col2, col3 = st.columns(3)

			with col1:
				st.markdown("""
				### Team Organization

				1. **Agile Methodology**
				   - 2-week sprint cycles
				   - Daily 15-minute stand-ups
				   - End-of-sprint demos
				   - Retrospectives for continuous improvement

				2. **Team Pairing**
				   - Technical-clinical buddy system
				   - Cross-functional teams for feature development
				   - Rotation of graduate students across areas
				   - Mentorship relationships

				3. **Documentation Practice**
				   - Collaborative workspace (Confluence)
				   - Code documentation standards
				   - Decision logs for key technical choices
				   - Knowledge transfer sessions
				""")

			with col2:
				st.markdown("""
				### Project Management

				1. **Milestone Planning**
				   - Quarterly OKRs (Objectives & Key Results)
				   - Monthly review sessions
				   - Critical path management
				   - Risk assessment and mitigation

				2. **Task Management**
				   - Jira for tracking and assignments
				   - Clear definition of done
				   - Task dependencies visualization
				   - Capacity planning for team members

				3. **Progress Tracking**
				   - Burndown charts for sprint progress
				   - Technical debt monitoring
				   - Quality metrics dashboard
				   - Stakeholder update rhythm
				""")

			with col3:
				st.markdown("""
				### Team Development

				1. **Skill Building**
				   - Individual development plans
				   - Technical workshops and tutorials
				   - Conference/publication opportunities
				   - Cross-training sessions

				2. **Feedback Culture**
				   - Regular 1:1 meetings
				   - 360-degree feedback
				   - Constructive code reviews
				   - Recognition of contributions

				3. **Knowledge Sharing**
				   - Weekly tech talks
				   - Paper reading group
				   - Shared learning repository
				   - External speaker series
				""")

			# Team challenges and solutions
			st.markdown('<div class="section-header">Anticipated Team Challenges</div>', unsafe_allow_html=True)

			st.markdown("""
			Based on experience coordinating healthcare AI teams, I anticipate and would proactively address these challenges:
			""")

			# Create challenges and solutions table
			challenges_data = {
				"Challenge": [
					"Technical-clinical communication barriers",
					"Different work rhythms across disciplines",
					"Varying technical skill levels among students",
					"Balancing innovation with clinical requirements",
					"Publication priorities vs. development timeline"
				],
				"Solution Approach": [
					"Establish shared vocabulary glossary; regular knowledge translation sessions; visual communication tools",
					"Flexible core hours; asynchronous updates; clear deadlines with buffer time; respect for clinical schedules",
					"Tiered task assignment; peer mentoring system; skills assessment and targeted training",
					"Clinical validation criteria in technical specs; rapid prototype testing with clinicians; user story mapping",
					"Publication planning aligned with project milestones; targeted conference submissions; balanced authorship plan"
				]
			}

			# Convert to DataFrame
			challenges_df = pd.DataFrame(challenges_data)

			# Style the table
			styled_df = challenges_df.style.set_properties(**{
				'background-color': '#f0f7f0',
				'border-color': '#dddddd',
				'border-style': 'solid',
				'border-width': '1px',
				'text-align': 'left',
				'padding': '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color', 'white'),
					('font-weight', 'bold'),
					('border', '1px solid #dddddd'),
					('text-align', 'left'),
					('padding', '8px')
				]}
			])

			# Display the table
			st.table(styled_df)

		with l_tab2:
			st.markdown('<div class="sub-header">Interdisciplinary Communication</div>', unsafe_allow_html=True)

			st.markdown("""
			Effective communication across disciplines is critical for the success of the EAS-ID project, which bridges nursing, computer science, and engineering domains.
			""")

			# Communication strategies
			st.markdown('<div class="section-header">Building Shared Understanding</div>', unsafe_allow_html=True)

			st.markdown("""
			To facilitate effective cross-disciplinary communication, I would implement these strategies:

			1. **Domain-Specific Onboarding**
			   - Cross-disciplinary training sessions
			   - Field observation opportunities (shadowing)
			   - Shared reading lists (technical and clinical)
			   - Recorded knowledge-transfer sessions

			2. **Communication Tools**
			   - Visual artifacts (diagrams, flowcharts)
			   - Shared glossary of terms
			   - Regular cross-functional meetings
			   - Decision-making framework documentation

			3. **Feedback Mechanisms**
			   - Regular retrospectives
			   - Anonymous suggestion system
			   - Cross-disciplinary peer reviews
			   - User-centered design workshops
			""")

			# Translation examples
			st.markdown('<div class="section-header">Technical-Clinical Translation Examples</div>', unsafe_allow_html=True)

			# Create columns for translation examples
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Technical to Clinical Translation

				**Technical Concept**: "The model struggles with low contrast in Type V-VI skin."

				**Translated Communication**: "Our software currently has difficulty detecting lighter bruises on darker skin tones. This means examiners may need to use the alternate light source more consistently with these patients."

				---

				**Technical Concept**: "We need to implement a sliding window detection algorithm."

				**Translated Communication**: "We're updating the app to scan the image in sections, which will help detect very small bruises that might otherwise be missed."

				---

				**Technical Concept**: "The mobile quantization reduced model accuracy by 3%."

				**Translated Communication**: "To make the app work without internet, we had to simplify the AI system, which means it might miss approximately 3 more bruises out of every 100 compared to the full system."
				""")

			with col2:
				st.markdown("""
				### Clinical to Technical Translation

				**Clinical Concept**: "The bruise presents with green-yellow margins indicating 3-5 days old."

				**Translated Communication**: "We need to adjust our color classification thresholds to recognize this specific green-yellow gradient pattern (RGB approximate range) as a key feature for the 72-120 hour age category."

				---

				**Clinical Concept**: "Documentation needs to follow IAFN guidelines for court admissibility."

				**Translated Communication**: "The export function must include these specific metadata fields, maintain original images alongside enhanced versions, and implement a cryptographic signature system for chain-of-custody verification."

				---

				**Clinical Concept**: "Patients may be traumatized and examination must be trauma-informed."

				**Translated Communication**: "The UI needs a 'pause' function, clear consent checkpoints, progress indicators, and minimal startling elements (no sudden sounds/flashes)."
				""")

			# Visual communication tools
			st.markdown('<div class="section-header">Visual Communication Tools</div>', unsafe_allow_html=True)

			st.markdown("""
			To bridge communication gaps, I would implement these visual tools:
			""")

			# Create tabs for different visual tools
			v_tab1, v_tab2, v_tab3 = st.tabs(["Clinical Journey Map", "Decision Trees", "Technical-Clinical Matrix"])

			with v_tab1:
				st.markdown("### Clinical Journey Mapping")

				st.markdown("""
				This visualization tool maps the clinical process alongside technical implementation, ensuring all team members understand the complete workflow:
				""")

				# Sample journey map visualization
				stages = ["Patient Arrival", "Consent Process", "Image Capture", "Analysis", "Documentation", "Follow-up"]

				# Create multi-layer journey map
				fig = go.Figure()

				# Define the y-positions for different layers
				y_positions = {
					"Clinical Actions": 3,
					"User Interface": 2,
					"Technical Process": 1
				}

				# Clinical actions
				clinical_actions = [
					"Initial patient assessment",
					"Informed consent discussion",
					"ALS examination procedure",
					"Clinical assessment of images",
					"Forensic documentation",
					"Safety planning & referrals"
				]

				# UI elements
				ui_elements = [
					"Case creation screen",
					"Consent capture form",
					"Guided imaging wizard",
					"Results dashboard",
					"Report generation",
					"Follow-up scheduling"
				]

				# Technical processes
				tech_processes = [
					"Patient record creation",
					"Digital consent storage",
					"Image preprocessing & quality check",
					"ML inference pipeline",
					"FHIR bundle generation",
					"Data synchronization"
				]

				# Calculate x positions for stages
				x_positions = np.linspace(0, 10, len(stages))

				# Plot stage headers
				for i, stage in enumerate(stages):
					fig.add_shape(
						type="rect",
						x0=x_positions[i] - 0.4, y0=3.5, x1=x_positions[i] + 0.4, y1=4,
						line=dict(color="#006633"),
						fillcolor="rgba(0, 102, 51, 0.2)"
					)

					fig.add_trace(go.Scatter(
						x=[x_positions[i]],
						y=[3.75],
						mode="text",
						text=[stage],
						textposition="middle center",
						textfont=dict(size=12, color="#006633"),
						showlegend=False
					))

				# Plot clinical actions
				for i, action in enumerate(clinical_actions):
					fig.add_shape(
						type="rect",
						x0=x_positions[i] - 0.45, y0=y_positions["Clinical Actions"] - 0.3,
						x1=x_positions[i] + 0.45, y1=y_positions["Clinical Actions"] + 0.3,
						line=dict(color="#F44336"),
						fillcolor="rgba(244, 67, 54, 0.2)"
					)

					fig.add_trace(go.Scatter(
						x=[x_positions[i]],
						y=[y_positions["Clinical Actions"]],
						mode="text",
						text=[action],
						textposition="middle center",
						textfont=dict(size=10, color="#333"),
						showlegend=False
					))

				# Plot UI elements
				for i, element in enumerate(ui_elements):
					fig.add_shape(
						type="rect",
						x0=x_positions[i] - 0.45, y0=y_positions["User Interface"] - 0.3,
						x1=x_positions[i] + 0.45, y1=y_positions["User Interface"] + 0.3,
						line=dict(color="#2196F3"),
						fillcolor="rgba(33, 150, 243, 0.2)"
					)

					fig.add_trace(go.Scatter(
						x=[x_positions[i]],
						y=[y_positions["User Interface"]],
						mode="text",
						text=[element],
						textposition="middle center",
						textfont=dict(size=10, color="#333"),
						showlegend=False
					))

				# Plot technical processes
				for i, process in enumerate(tech_processes):
					fig.add_shape(
						type="rect",
						x0=x_positions[i] - 0.45, y0=y_positions["Technical Process"] - 0.3,
						x1=x_positions[i] + 0.45, y1=y_positions["Technical Process"] + 0.3,
						line=dict(color="#4CAF50"),
						fillcolor="rgba(76, 175, 80, 0.2)"
					)

					fig.add_trace(go.Scatter(
						x=[x_positions[i]],
						y=[y_positions["Technical Process"]],
						mode="text",
						text=[process],
						textposition="middle center",
						textfont=dict(size=10, color="#333"),
						showlegend=False
					))

				# Draw connections between stages
				for i in range(len(stages) - 1):
					# Clinical to clinical
					fig.add_shape(
						type="line",
						x0=x_positions[i] + 0.45, y0=y_positions["Clinical Actions"],
						x1=x_positions[i+1] - 0.45, y1=y_positions["Clinical Actions"],
						line=dict(color="#F44336", width=1)
					)

					# UI to UI
					fig.add_shape(
						type="line",
						x0=x_positions[i] + 0.45, y0=y_positions["User Interface"],
						x1=x_positions[i+1] - 0.45, y1=y_positions["User Interface"],
						line=dict(color="#2196F3", width=1)
					)

					# Technical to technical
					fig.add_shape(
						type="line",
						x0=x_positions[i] + 0.45, y0=y_positions["Technical Process"],
						x1=x_positions[i+1] - 0.45, y1=y_positions["Technical Process"],
						line=dict(color="#4CAF50", width=1)
					)

				# Add vertical connections between layers
				for i in range(len(stages)):
					# Clinical to UI
					fig.add_shape(
						type="line",
						x0=x_positions[i], y0=y_positions["Clinical Actions"] - 0.3,
						x1=x_positions[i], y1=y_positions["User Interface"] + 0.3,
						line=dict(color="gray", width=1, dash="dot")
					)

					# UI to Technical
					fig.add_shape(
						type="line",
						x0=x_positions[i], y0=y_positions["User Interface"] - 0.3,
						x1=x_positions[i], y1=y_positions["Technical Process"] + 0.3,
						line=dict(color="gray", width=1, dash="dot")
					)

				# Add layer labels
				layers = list(y_positions.keys())
				colors = ["#F44336", "#2196F3", "#4CAF50"]

				for i, (layer, y) in enumerate(y_positions.items()):
					fig.add_trace(go.Scatter(
						x=[-0.5],
						y=[y],
						mode="text",
						text=[layer],
						textposition="middle right",
						textfont=dict(size=12, color=colors[i]),
						showlegend=False
					))

				# Update layout
				fig.update_layout(
					title="Clinical-Technical Journey Map",
					showlegend=False,
					height=400,
					margin=dict(l=100, r=20, t=40, b=20),
					plot_bgcolor="white",
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 4.5])
				)

				st.plotly_chart(fig, use_container_width=True)

				st.markdown("""
				This journey map helps:

				- Clinical team members understand the technical processes supporting their workflow
				- Developers visualize how their code impacts the patient experience
				- UI designers align interface elements with both clinical and technical requirements
				- All team members identify integration points and dependencies
				""")

			with v_tab2:
				st.markdown("### Decision Trees for Cross-Disciplinary Decisions")

				st.markdown("""
				Decision trees help structure complex cross-disciplinary decisions with clear criteria:
				""")

				# Sample decision tree visualization
				fig = go.Figure()

				# Node positions
				nodes = {
					"root": {"text": "Is bruise visible\nunder white light?", "x": 5, "y": 5},
					"als": {"text": "Capture using\nALS imaging", "x": 7, "y": 4},
					"skin_tone": {"text": "Is skin tone\nType V-VI?", "x": 3, "y": 4},
					"enhance": {"text": "Apply enhancement\nalgorithm", "x": 1, "y": 3},
					"standard": {"text": "Standard\ndetection", "x": 5, "y": 3},
					"als_process": {"text": "Apply ALS-specific\nprocessing", "x": 9, "y": 3},
					"als_visible": {"text": "Is bruise visible\nunder ALS?", "x": 9, "y": 2},
					"document": {"text": "Document with\nhigh confidence", "x": 7, "y": 1},
					"inconclusive": {"text": "Mark as\ninconclusive", "x": 11, "y": 1},
					"confident": {"text": "Continue with\nstandard workflow", "x": 3, "y": 2},
					"review": {"text": "Flag for\nexpert review", "x": 1, "y": 2}
				}

				# Connections
				connections = [
					{"from": "root", "to": "als", "label": "No", "x": 6, "y": 4.5},
					{"from": "root", "to": "skin_tone", "label": "Yes", "x": 4, "y": 4.5},
					{"from": "skin_tone", "to": "enhance", "label": "Yes", "x": 2, "y": 3.5},
					{"from": "skin_tone", "to": "standard", "label": "No", "x": 4, "y": 3.5},
					{"from": "als", "to": "als_process", "label": "", "x": 8, "y": 3.5},
					{"from": "als_process", "to": "als_visible", "label": "", "x": 9, "y": 2.5},
					{"from": "als_visible", "to": "document", "label": "Yes", "x": 8, "y": 1.5},
					{"from": "als_visible", "to": "inconclusive", "label": "No", "x": 10, "y": 1.5},
					{"from": "enhance", "to": "confident", "label": "High conf.", "x": 3, "y": 2.5},
					{"from": "enhance", "to": "review", "label": "Low conf.", "x": 1, "y": 2.5}
				]

				# Colors for different node types
				node_colors = {
					"root": "#FF9800",       # Orange
					"als": "#2196F3",        # Blue
					"skin_tone": "#2196F3",  # Blue
					"enhance": "#4CAF50",    # Green
					"standard": "#4CAF50",   # Green
					"als_process": "#4CAF50",# Green
					"als_visible": "#2196F3",# Blue
					"document": "#9C27B0",   # Purple
					"inconclusive": "#9C27B0",# Purple
					"confident": "#9C27B0",  # Purple
					"review": "#9C27B0"      # Purple
				}

				# Plot nodes
				for node_id, attrs in nodes.items():
					fig.add_shape(
						type="rect",
						x0=attrs["x"] - 0.9, y0=attrs["y"] - 0.4,
						x1=attrs["x"] + 0.9, y1=attrs["y"] + 0.4,
						line=dict(color=node_colors[node_id]),
						fillcolor=f"rgba({int(node_colors[node_id][1:3], 16)}, {int(node_colors[node_id][3:5], 16)}, {int(node_colors[node_id][5:7], 16)}, 0.2)"
					)

					fig.add_trace(go.Scatter(
						x=[attrs["x"]],
						y=[attrs["y"]],
						mode="text",
						text=[attrs["text"]],
						textposition="middle center",
						textfont=dict(size=10, color="#333"),
						showlegend=False
					))

				# Plot connections
				for conn in connections:
					from_node = nodes[conn["from"]]
					to_node = nodes[conn["to"]]

					fig.add_shape(
						type="line",
						x0=from_node["x"], y0=from_node["y"] - 0.4,
						x1=to_node["x"], y1=to_node["y"] + 0.4,
						line=dict(color="gray", width=1)
					)

					if conn["label"]:
						fig.add_trace(go.Scatter(
							x=[conn["x"]],
							y=[conn["y"]],
							mode="text",
							text=[conn["label"]],
							textposition="middle center",
							textfont=dict(size=9, color="#555"),
							showlegend=False
						))

				# Legend for node colors
				categories = [
					{"name": "Decision Point", "color": "#FF9800"},
					{"name": "Observation", "color": "#2196F3"},
					{"name": "Process", "color": "#4CAF50"},
					{"name": "Outcome", "color": "#9C27B0"}
				]

				for i, cat in enumerate(categories):
					fig.add_trace(go.Scatter(
						x=[0.5],
						y=[5 - i * 0.5],
						mode="markers+text",
						marker=dict(size=10, color=cat["color"]),
						text=[cat["name"]],
						textposition="middle right",
						showlegend=False
					))

				# Update layout
				fig.update_layout(
					title="Bruise Examination Decision Tree",
					showlegend=False,
					height=500,
					margin=dict(l=20, r=20, t=40, b=20),
					plot_bgcolor="white",
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 12]),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6])
				)

				st.plotly_chart(fig, use_container_width=True)

				st.markdown("""
				These decision trees:

				- Create a shared decision framework for all team members
				- Combine clinical and technical decision points
				- Provide clear guidance for edge cases
				- Document the reasoning behind workflow choices
				- Serve as training material for new team members
				""")

			with v_tab3:
				st.markdown("### Technical-Clinical Translation Matrix")

				st.markdown("""
				This matrix maps technical parameters to clinical implications, ensuring all team members understand the connections:
				""")

				# Sample matrix data
				matrix_data = {
					"Technical Parameter": [
						"Segmentation IoU Threshold",
						"Model Confidence Score",
						"Image Resolution",
						"ALS Wavelength",
						"False Positive Rate"
					],
					"Technical Meaning": [
						"Overlap required between predicted and actual bruise boundaries",
						"Statistical certainty of bruise detection",
						"Pixel density of captured images",
						"Light frequency used for illumination",
						"Proportion of non-bruises misclassified as bruises"
					],
					"Clinical Implication": [
						"Affects measured bruise size and shape documentation",
						"Determines when manual review is required",
						"Impacts visibility of subtle bruising patterns",
						"Affects visibility on different skin tones and bruise ages",
						"Influences potential for overdiagnosis"
					],
					"Optimization Balance": [
						"Higher threshold → more precise but may miss diffuse bruises",
						"Higher threshold → fewer false positives but more missed bruises",
						"Higher resolution → better detection but larger file size and slower processing",
						"Multiple wavelengths → better detection but more complex protocol",
						"Stringent threshold → fewer false positives but may miss subtle bruises"
					]
				}

				# Convert to DataFrame
				matrix_df = pd.DataFrame(matrix_data)

				# Style the table
				styled_df = matrix_df.style.set_properties(**{
					'background-color': '#f0f7f0',
					'border-color': '#dddddd',
					'border-style': 'solid',
					'border-width': '1px',
					'text-align': 'left',
					'padding': '8px'
				})

				# Apply specific styling to the header
				styled_df = styled_df.set_table_styles([
					{'selector': 'th', 'props': [
						('background-color', '#006633'),
						('color', 'white'),
						('font-weight', 'bold'),
						('border', '1px solid #dddddd'),
						('text-align', 'left'),
						('padding', '8px')
					]}
				])

				# Display the table
				st.table(styled_df)

				st.markdown("""
				This translation matrix helps:

				- Technical team members understand the clinical impact of their parameter choices
				- Clinical team members provide informed input on technical trade-offs
				- Create a shared language for discussing optimization
				- Document the reasoning behind parameter selection
				- Guide future adjustments as the system evolves
				""")

		with l_tab3:
			st.markdown('<div class="sub-header">Graduate Student Mentorship</div>', unsafe_allow_html=True)

			st.markdown("""
			A key responsibility of the postdoc position is supervising and mentoring graduate students. I would implement a structured mentorship approach that balances guidance with independence.
			""")

			# Mentorship framework
			st.markdown('<div class="section-header">Mentorship Philosophy</div>', unsafe_allow_html=True)

			st.markdown("""
			My mentorship philosophy is based on these core principles:

			1. **Growth-Oriented Leadership**
			   - Focus on developing students' skills rather than just project outcomes
			   - Challenge students with increasing responsibility
			   - Provide constructive feedback focused on improvement
			   - Celebrate progress and milestone achievements

			2. **Balanced Autonomy**
			   - Start with structured guidance and clear expectations
			   - Gradually increase independence as skills develop
			   - Maintain regular check-ins to prevent roadblocks
			   - Balance support with space for creativity and learning

			3. **Holistic Development**
			   - Technical skills development
			   - Research methodology training
			   - Academic writing and presentation coaching
			   - Career development and networking opportunities

			4. **Inclusive Mentoring**
			   - Recognize and adapt to different learning styles
			   - Create equitable opportunities for all students
			   - Address implicit biases in feedback and project assignments
			   - Foster a supportive environment for diverse perspectives
			""")

			# Mentorship plan visualization
			st.markdown('<div class="section-header">Structured Mentorship Plan</div>', unsafe_allow_html=True)

			# Create columns for mentorship plan elements
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Regular Touchpoints

				**Weekly Individual Meetings**
				- Progress review
				- Technical guidance
				- Roadblock resolution
				- Next steps planning

				**Bi-weekly Team Meetings**
				- Cross-project updates
				- Knowledge sharing
				- Collaborative problem-solving
				- Peer feedback

				**Monthly Career Development**
				- Publication planning
				- Conference preparation
				- Professional networking
				- Skill development check-in

				**Quarterly Review Sessions**
				- Formal progress assessment
				- Goals adjustment
				- Long-term planning
				- Feedback in both directions
				""")

			with col2:
				st.markdown("""
				### Development Activities

				**Technical Skills**
				- Guided coding sessions
				- Paper implementation exercises
				- Code review practice
				- Architecture design workshops

				**Research Methods**
				- Experiment design tutorials
				- Statistical analysis workshops
				- Literature review techniques
				- Research ethics discussions

				**Communication Skills**
				- Technical writing workshops
				- Presentation practice sessions
				- Visualization techniques
				- Interdisciplinary communication

				**Professional Development**
				- Conference abstract preparation
				- CV/resume building
				- Interview preparation
				- Network building opportunities
				""")

			# Student project progression
			st.markdown('<div class="section-header">Student Project Progression Model</div>', unsafe_allow_html=True)

			# Create a visual progression model
			fig = go.Figure()

			# Stages of progression
			stages = [
				{"stage": "Onboarding", "week": 1, "autonomy": 1, "activities": [
					"Project introduction",
					"Background reading",
					"Environment setup",
					"Basic tutorials"
				]},
				{"stage": "Foundational", "week": 4, "autonomy": 2, "activities": [
					"Structured tasks",
					"Guided implementation",
					"Weekly checkpoints",
					"Defined milestones"
				]},
				{"stage": "Developing", "week": 8, "autonomy": 3, "activities": [
					"Independent modules",
					"Self-directed research",
					"Solution exploration",
					"Regular updates"
				]},
				{"stage": "Advanced", "week": 12, "autonomy": 4, "activities": [
					"Project leadership",
					"Novel contributions",
					"Mentoring junior members",
					"Publication drafting"
				]},
				{"stage": "Expert", "week": 16, "autonomy": 5, "activities": [
					"Full project ownership",
					"Strategic direction",
					"External presentations",
					"Publication completion"
				]}
			]

			# Plot progression stages
			for i, stage in enumerate(stages):
				# Plot stage box
				fig.add_shape(
					type="rect",
					x0=stage["week"] - 1.5, y0=0.5, x1=stage["week"] + 1.5, y1=5.5,
					line=dict(color="#006633"),
					fillcolor=f"rgba(0, 102, 51, {0.1 + 0.1 * i})"
				)

				# Stage title
				fig.add_trace(go.Scatter(
					x=[stage["week"]],
					y=[6],
					mode="text",
					text=[stage["stage"]],
					textposition="middle center",
					textfont=dict(size=14, color="#006633"),
					showlegend=False
				))

				# Week label
				fig.add_trace(go.Scatter(
					x=[stage["week"]],
					y=[0],
					mode="text",
					text=[f"Week {stage['week']}"],
					textposition="middle center",
					textfont=dict(size=10, color="#555"),
					showlegend=False
				))

				# Activities
				for j, activity in enumerate(stage["activities"]):
					fig.add_trace(go.Scatter(
						x=[stage["week"]],
						y=[4.5 - j],
						mode="text",
						text=[activity],
						textposition="middle center",
						textfont=dict(size=10, color="#333"),
						showlegend=False
					))

			# Add autonomy line
			autonomy_x = [stage["week"] for stage in stages]
			autonomy_y = [stage["autonomy"] for stage in stages]

			fig.add_trace(go.Scatter(
				x=autonomy_x,
				y=autonomy_y,
				mode="lines+markers",
				line=dict(color="#F44336", width=3),
				marker=dict(size=10, color="#F44336"),
				name="Student Autonomy Level"
			))

			# Add autonomy axis label
			fig.add_trace(go.Scatter(
				x=[1],
				y=[5],
				mode="text",
				text=["Student Autonomy"],
				textposition="middle center",
				textfont=dict(size=12, color="#F44336"),
				showlegend=False
			))

			# Update layout
			fig.update_layout(
				title="Student Project Progression Model",
				showlegend=False,
				height=400,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 18]),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5])
			)

			st.plotly_chart(fig, use_container_width=True)

			# Tailored mentorship strategies
			st.markdown('<div class="section-header">Tailored Mentorship Strategies</div>', unsafe_allow_html=True)

			st.markdown("""
			Different students require different mentorship approaches based on their background, skills, and learning styles:
			""")

			# Create tabs for different student types
			s_tab1, s_tab2, s_tab3 = st.tabs(["Technical Students", "Clinical Students", "Mixed Background"])

			with s_tab1:
				st.markdown("""
				### CS/Engineering Graduate Students

				**Strengths to Leverage:**
				- Strong technical foundation
				- Programming experience
				- Algorithmic thinking
				- Mathematical background

				**Areas to Develop:**
				- Understanding of clinical context
				- Appreciation for user needs
				- Documentation for non-technical audiences
				- Translation of clinical requirements to specifications

				**Mentorship Strategies:**
				- Pair with clinical team members for shadowing
				- Assign combined technical-clinical deliverables
				- Practice explaining technical concepts without jargon
				- Focus on real-world impact of technical decisions

				**Project Assignment Approach:**
				- Start with well-defined technical components
				- Gradually introduce clinical constraints
				- Encourage direct interaction with clinical partners
				- Guide toward interdisciplinary publication opportunities
				""")

			with s_tab2:
				st.markdown("""
				### Nursing/Clinical Graduate Students

				**Strengths to Leverage:**
				- Clinical domain expertise
				- Patient care perspective
				- Practical workflow knowledge
				- Evidence-based practice background

				**Areas to Develop:**
				- Technical vocabulary
				- Basic programming concepts
				- Understanding of AI limitations
				- Data analysis skills

				**Mentorship Strategies:**
				- Provide accessible technical tutorials and resources
				- Use visual explanations of technical concepts
				- Start with high-level systems understanding before details
				- Focus on translating clinical requirements to technical specifications

				**Project Assignment Approach:**
				- Begin with user interface and workflow components
				- Involve in training data annotation and quality review
				- Engage in evaluation design and metrics selection
				- Guide toward publications bridging clinical and technical domains
				""")

			with s_tab3:
				st.markdown("""
				### Mixed Background / Health Informatics Students

				**Strengths to Leverage:**
				- Interdisciplinary perspective
				- Bridge-building capabilities
				- Balanced understanding of constraints
				- Systems thinking

				**Areas to Develop:**
				- Depth in specific technical areas
				- Advanced clinical knowledge in forensic nursing
				- Research methodology refinement
				- Specialized tool proficiency

				**Mentorship Strategies:**
				- Identify specific technical and clinical areas for focused development
				- Leverage interdisciplinary background for team coordination roles
				- Provide opportunities to translate between domains
				- Focus on end-to-end system integration

				**Project Assignment Approach:**
				- Integration components between clinical and technical systems
				- Evaluation and validation studies
				- Documentation and knowledge management
				- User testing and feedback incorporation
				""")

			# Publication and professional development
			st.markdown('<div class="section-header">Publication and Professional Development</div>', unsafe_allow_html=True)

			st.markdown("""
			<div class="highlight-text">
			<b>Supporting Student Publication and Career Development</b>

			I would create structured opportunities for students to develop professionally through:

			1. <b>Publication Planning</b>
			   - Develop publication roadmap with each student
			   - Target appropriate venues (journals and conferences)
			   - Schedule regular writing sessions
			   - Provide iterative feedback on drafts

			2. <b>Presentation Opportunities</b>
			   - Internal team presentations for practice
			   - Departmental seminars
			   - Conference abstracts and posters
			   - Community and stakeholder presentations

			3. <b>Networking Support</b>
			   - Introduction to relevant researchers
			   - Guidance on professional social media presence
			   - Support for conference attendance
			   - Involvement in collaborative projects

			4. <b>Career Guidance</b>
			   - Regular discussions about career goals
			   - CV/resume development
			   - Interview preparation
			   - Exposure to diverse career paths in academia, industry, and healthcare
			</div>
			""", unsafe_allow_html=True)

	def render_funding_page(self):
		st.markdown('<div class="main-header">Funding and Strategic Impact</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"How would you position a competing-renewal NIH R01 from this work?"**

		The interviewers are looking for strategic vision and grant-writing awareness.
		""")

		# Create tabs for different aspects of funding
		f_tab1, f_tab2, f_tab3 = st.tabs(["Grant Strategy", "Impact Narrative", "Research Trajectory"])

		with f_tab1:
			st.markdown('<div class="sub-header">NIH Renewal Strategy</div>', unsafe_allow_html=True)

			st.markdown("""
			Based on the EAS-ID project's current funding and outcomes, I would develop a comprehensive strategy for competitive renewal focused on extending impact while building on established foundations.
			""")

			# Funding context
			st.markdown('<div class="section-header">Current Funding Context</div>', unsafe_allow_html=True)

			# Create columns for current funding
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Current Funding Sources

				1. **Philanthropic Gift**
				   - $4.85 million
				   - Focus on technology development
				   - Emphasis on equity across skin tones
				   - Platform development and validation

				2. **NIH AIM-AHEAD Supplement**
				   - Focus on algorithmic fairness
				   - Equity metrics development
				   - Community engagement

				3. **Department of Justice Grant**
				   - Nearly $1 million
				   - Focus on forensic applications
				   - Legal admissibility research
				   - Criminal justice system integration
				""")

			with col2:
				st.markdown("""
				### Relevant NIH Institutes/Centers

				1. **National Institute of Nursing Research (NINR)**
				   - Primary institute for nursing science
				   - Focus on innovative technologies for healthcare
				   - Emphasis on health equity

				2. **National Library of Medicine (NLM)**
				   - Biomedical informatics and data science
				   - AI/ML in healthcare applications
				   - Biomedical image analysis

				3. **National Institute of Biomedical Imaging and Bioengineering (NIBIB)**
				   - Technological innovations in healthcare
				   - Point-of-care technologies
				   - Image processing and analysis

				4. **National Institute on Minority Health and Health Disparities (NIMHD)**
				   - Health equity focus
				   - Technology to address healthcare disparities
				   - Community engagement requirements
				""")

			# NIH R01 renewal strategy
			st.markdown('<div class="section-header">R01 Renewal Strategy</div>', unsafe_allow_html=True)

			st.markdown("""
			For a successful competitive renewal (Type 2) application, I would position the project along these strategic dimensions:
			""")

			# Create columns for renewal strategy
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Scientific Progress Narrative

				1. **Accomplishments Summary**
				   - Quantifiable results from initial funding period
				   - Technology development milestones achieved
				   - Preliminary clinical validation results
				   - Publications and presentations

				2. **Knowledge Gaps Identified**
				   - Areas requiring further investigation
				   - Limitations of current approach
				   - Emergent research questions
				   - Technological challenges

				3. **Innovation Narrative**
				   - New approaches developed since original grant
				   - Novel methodologies emerging from current work
				   - Technological advances enabling new directions
				   - Cross-disciplinary insights
				""")

			with col2:
				st.markdown("""
				### Strategic Expansion Areas

				1. **Clinical Translation Focus**
				   - Moving from prototype to clinical implementation
				   - Multi-site validation studies
				   - Clinical workflow integration
				   - Real-world evidence generation

				2. **Population Expansion**
				   - Pediatric applications
				   - Geriatric considerations
				   - Diverse clinical settings
				   - International validation

				3. **Technical Enhancement**
				   - Multimodal AI integration
				   - Longitudinal monitoring capabilities
				   - Edge computing optimization
				   - Interoperability with health systems
				""")

			# NIH priorities alignment
			st.markdown('<div class="section-header">Alignment with NIH Priorities</div>', unsafe_allow_html=True)

			# Sample alignment data
			alignment_data = {
				"NIH Priority": [
					"Health Equity",
					"AI/ML in Healthcare",
					"Rural Health Access",
					"Violence Prevention",
					"Digital Health Technologies"
				],
				"Current Project Alignment": [
					"Skin tone equity in bruise detection",
					"Deep learning for image analysis",
					"Limited mobile deployment capability",
					"Intimate partner violence documentation",
					"Mobile platform prototype"
				],
				"Renewal Enhancement": [
					"Expanded demographic validation across populations",
					"Explainable AI features for clinical trust",
					"Telemedicine integration for remote assessment",
					"Predictive risk assessment capabilities",
					"Full digital clinical workflow integration"
				]
			}

			# Convert to DataFrame
			alignment_df = pd.DataFrame(alignment_data)

			# Style the table
			styled_df = alignment_df.style.set_properties(**{
				'background-color': '#f0f7f0',
				'border-color': '#dddddd',
				'border-style': 'solid',
				'border-width': '1px',
				'text-align': 'left',
				'padding': '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color', 'white'),
					('font-weight', 'bold'),
					('border', '1px solid #dddddd'),
					('text-align', 'left'),
					('padding', '8px')
				]}
			])

			# Display the table
			st.table(styled_df)

			# Grant writing strategy
			st.markdown('<div class="section-header">Grant Writing Strategy</div>', unsafe_allow_html=True)

			st.markdown("""
			To maximize competitiveness, I would use these proven strategies for NIH renewal applications:

			1. **Strong Continuity Narrative**
			   - Clear connections to original aims
			   - Logical progression based on findings
			   - Explanation of any pivots or new directions
			   - Consistent intellectual framework

			2. **Productivity Demonstration**
			   - Publications in high-impact journals
			   - Citations and research impact
			   - Technology development milestones
			   - Team stability and growth

			3. **Collaborative Expansion**
			   - New strategic partners
			   - Multi-institutional approach
			   - Industry collaborations for implementation
			   - Patient advocacy involvement

			4. **Budget Justification Strategy**
			   - Detailed expenditure history
			   - Resource leverage from other funding
			   - Cost-effective technology deployment
			   - Clear value proposition for renewal
			""")

			# Suggested review panel
			st.markdown("""
			<div class="highlight-text">
			<b>Suggested NIH Study Section Strategy</b>

			For a competitive renewal, I would target the following review groups:

			1. <b>Healthcare Information Technology Research (HITR)</b>
			   - Focus on innovative health IT solutions
			   - Expertise in clinical decision support
			   - Mobile health applications expertise
			   - AI/ML in clinical applications

			2. <b>Biomedical Imaging Technology A (BMIT-A)</b>
			   - Medical image analysis expertise
			   - Computer vision applications
			   - Novel imaging modalities
			   - Image processing methods

			3. <b>Nursing and Related Clinical Sciences (NRCS)</b>
			   - Nursing science and clinical applications
			   - Practice-based research expertise
			   - Patient-centered outcomes focus
			   - Implementation science perspective

			The cover letter would request assignment to NINR as primary institute with NLM as secondary, and suggest these study sections based on the interdisciplinary nature of the work.
			</div>
			""", unsafe_allow_html=True)

		with f_tab2:
			st.markdown('<div class="sub-header">Impact Narrative Development</div>', unsafe_allow_html=True)

			st.markdown("""
			A compelling impact narrative is essential for competitive renewal funding. I would develop a multi-dimensional impact framework:
			""")

			# Impact dimensions
			st.markdown('<div class="section-header">Impact Framework</div>', unsafe_allow_html=True)

			# Create impact framework visualization
			fig = go.Figure()

			# Define impact dimensions
			dimensions = [
				{"name": "Clinical Impact", "x": 3, "y": 5, "radius": 1.8, "color": "#4CAF50", "items": [
					"Enhanced bruise detection across skin tones",
					"Improved documentation quality",
					"Reduced examination time",
					"Standardized forensic evidence"
				]},
				{"name": "Technological Impact", "x": 7, "y": 5, "radius": 1.8, "color": "#2196F3", "items": [
					"Novel AI architecture for medical imaging",
					"Mobile deployment optimization",
					"Multi-spectral imaging techniques",
					"Secure clinical data architecture"
				]},
				{"name": "Scientific Impact", "x": 1, "y": 2, "radius": 1.8, "color": "#9C27B0", "items": [
					"New knowledge on bruise detection",
					"Validation of ALS techniques",
					"Skin tone algorithm fairness metrics",
					"Bruise aging methodology"
				]},
				{"name": "Social Impact", "x": 5, "y": 2, "radius": 1.8, "color": "#F44336", "items": [
					"Improved IPV documentation",
					"Healthcare equity advancement",
					"Legal justice support",
					"Patient empowerment"
				]},
				{"name": "Economic Impact", "x": 9, "y": 2, "radius": 1.8, "color": "#FF9800", "items": [
					"Reduced healthcare costs",
					"Commercial technology potential",
					"Workforce training efficiency",
					"Legal system cost savings"
				]}
			]

			# Plot impact dimensions
			for dim in dimensions:
				# Dimension circle
				fig.add_shape(
					type="circle",
					x0=dim["x"] - dim["radius"], y0=dim["y"] - dim["radius"],
					x1=dim["x"] + dim["radius"], y1=dim["y"] + dim["radius"],
					line=dict(color=dim["color"]),
					fillcolor=f"rgba({int(dim['color'][1:3], 16)}, {int(dim['color'][3:5], 16)}, {int(dim['color'][5:7], 16)}, 0.2)"
				)

				# Dimension name
				fig.add_trace(go.Scatter(
					x=[dim["x"]],
					y=[dim["y"] + 0.5],
					mode="text",
					text=[dim["name"]],
					textposition="middle center",
					textfont=dict(size=14, color=dim["color"]),
					showlegend=False
				))

				# Dimension items
				for i, item in enumerate(dim["items"]):
					angle = 2 * np.pi * i / len(dim["items"])
					r = dim["radius"] * 0.6
					item_x = dim["x"] + r * np.cos(angle)
					item_y = dim["y"] + r * np.sin(angle)

					fig.add_trace(go.Scatter(
						x=[item_x],
						y=[item_y],
						mode="markers+text",
						marker=dict(size=8, color=dim["color"]),
						text=[item],
						textposition="middle center",
						textfont=dict(size=9, color="#333"),
						showlegend=False
					))

			# Add connections between dimensions
			connections = [
				("Clinical Impact", "Technological Impact"),
				("Clinical Impact", "Scientific Impact"),
				("Clinical Impact", "Social Impact"),
				("Technological Impact", "Economic Impact"),
				("Scientific Impact", "Social Impact"),
				("Social Impact", "Economic Impact")
			]

			for conn in connections:
				# Find coordinates for both dimensions
				dim1 = next(dim for dim in dimensions if dim["name"] == conn[0])
				dim2 = next(dim for dim in dimensions if dim["name"] == conn[1])

				fig.add_trace(go.Scatter(
					x=[dim1["x"], dim2["x"]],
					y=[dim1["y"], dim2["y"]],
					mode="lines",
					line=dict(width=1, color="gray", dash="dot"),
					showlegend=False
				))

			# Update layout
			fig.update_layout(
				title="Multidimensional Impact Framework",
				showlegend=False,
				height=500,
				margin=dict(l=20, r=20, t=40, b=20),
				plot_bgcolor="white",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 7])
			)

			st.plotly_chart(fig, use_container_width=True)

			# Quantifying impact
			st.markdown('<div class="section-header">Quantifying Impact for NIH</div>', unsafe_allow_html=True)

			st.markdown("""
			For the renewal application, I would develop specific metrics and evidence to quantify the project's impact:
			""")

			# Create tabs for different impact areas
			imp_tab1, imp_tab2, imp_tab3, imp_tab4 = st.tabs(["Clinical", "Scientific", "Technical", "Societal"])

			with imp_tab1:
				st.markdown("### Clinical Impact Metrics")

				st.markdown("""
				**First-Phase Accomplishments**

				1. **Detection Accuracy**
				   - 87% overall detection rate across skin tones
				   - 92% detection rate for ALS imaging
				   - Validation across 500+ clinical cases
				   - 5× improvement over white light alone

				2. **Clinical Workflow**
				   - 35% reduction in examination time
				   - 78% of clinicians reported improved confidence
				   - 90% documentation completeness rate
				   - Integration with 3 clinical sites

				**Renewal Goals**

				1. **Clinical Validation**
				   - Multi-center validation (12+ sites)
				   - 2,000+ patient dataset
				   - Diverse practice settings (ED, clinics, rural)
				   - Long-term outcome tracking

				2. **Specialty Applications**
				   - Pediatric protocol development
				   - Geriatric-specific validation
				   - Integration with sexual assault nurse examiner workflow
				   - Telehealth consultation capability
				""")

			with imp_tab2:
				st.markdown("### Scientific Impact Metrics")

				st.markdown("""
				**First-Phase Accomplishments**

				1. **Publications**
				   - 7 peer-reviewed journal articles
				   - 3 in nursing journals
				   - 4 in technical/engineering journals
				   - 12 conference presentations

				2. **Knowledge Generation**
				   - First validated dataset of bruises across Fitzpatrick skin types
				   - Novel algorithms for ALS image processing
				   - Quantitative model of bruise aging characteristics
				   - Objective measurement protocol for bruise assessment

				**Renewal Goals**

				1. **Research Expansion**
				   - Longitudinal studies of bruise evolution
				   - Cross-institutional collaborative research
				   - Integration with other biometric measures
				   - Machine learning explainability advancements

				2. **Scientific Dissemination**
				   - Open science framework for data sharing
				   - Algorithm publication with code repository
				   - Clinical practice guideline development
				   - Educational curriculum for nursing programs
				""")

			with imp_tab3:
				st.markdown("### Technical Impact Metrics")

				st.markdown("""
				**First-Phase Accomplishments**

				1. **Technology Development**
				   - Functional mobile prototype
				   - Cloud-based analysis pipeline
				   - Secure FHIR-compliant database
				   - ALS hardware integration

				2. **Algorithm Performance**
				   - 92% accuracy on validation dataset
				   - <5% disparity across skin tones
				   - 80% accuracy in bruise age estimation
				   - Real-time performance on mobile devices

				**Renewal Goals**

				1. **Platform Expansion**
				   - Full production-ready system
				   - EHR integration capabilities
				   - Telehealth platform compatibility
				   - Multi-device support

				2. **Technical Advancements**
				   - Multimodal deep learning architecture
				   - 3D reconstruction capabilities
				   - Privacy-preserving federated learning
				   - Automated reporting generation
				""")

			with imp_tab4:
				st.markdown("### Societal Impact Metrics")

				st.markdown("""
				**First-Phase Accomplishments**

				1. **Health Equity Advancement**
				   - Reduced detection disparity across skin tones
				   - Equal documentation quality regardless of race
				   - Technology accessible to diverse clinical settings
				   - Community engagement in development

				2. **Patient Impact**
				   - Improved evidence quality for IPV cases
				   - Reduced examination discomfort
				   - Increased confidence in forensic findings
				   - Enhanced access to forensic nursing services

				**Renewal Goals**

				1. **Justice System Integration**
				   - Court admissibility studies
				   - Legal proceedings outcome tracking
				   - Expert testimony support package
				   - Law enforcement collaboration

				2. **Policy Influence**
				   - Standard of care recommendations
				   - Insurance reimbursement pathways
				   - Health equity policy advocacy
				   - International protocol adoption
				""")

			# Stakeholder impact
			st.markdown('<div class="section-header">Impact Across Stakeholders</div>', unsafe_allow_html=True)

			# Sample stakeholder impact data
			stakeholder_data = {
				"Stakeholder": [
					"Patients/Survivors",
					"Forensic Nurses",
					"Healthcare Systems",
					"Legal System",
					"Researchers",
					"Technology Industry"
				],
				"Current Impact": [
					"Improved documentation of injuries",
					"Enhanced detection capabilities",
					"Standardized forensic protocols",
					"More reliable evidence collection",
					"Novel datasets and algorithms",
					"New application for computer vision"
				],
				"Renewal-Phase Impact": [
					"Better legal outcomes and protection",
					"Reduced workload with higher quality",
					"Cost reduction and risk mitigation",
					"Stronger evidence for prosecution",
					"Expanded collaborative opportunities",
					"Commercial translation potential"
				]
			}

			# Convert to DataFrame
			stakeholder_df = pd.DataFrame(stakeholder_data)

			# Style the table
			styled_df = stakeholder_df.style.set_properties(**{
				'background-color': '#f0f7f0',
				'border-color': '#dddddd',
				'border-style': 'solid',
				'border-width': '1px',
				'text-align': 'left',
				'padding': '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color', 'white'),
					('font-weight', 'bold'),
					('border', '1px solid #dddddd'),
					('text-align', 'left'),
					('padding', '8px')
				]}
			])

			# Display the table
			st.table(styled_df)

			# Letter of support strategy
			st.markdown("""
			<div class="highlight-text">
			<b>Letter of Support Strategy</b>

			To strengthen the renewal application, I would secure letters of support demonstrating multi-stakeholder impact:

			1. <b>Clinical Partners</b>
			   - Forensic nursing program directors
			   - Hospital system administrators
			   - Emergency department chiefs
			   - Testimony on workflow improvements

			2. <b>Patient Advocacy</b>
			   - Domestic violence organizations
			   - Survivor advocacy groups
			   - Patient representatives
			   - Testimonials on importance to survivors

			3. <b>Legal System</b>
			   - District attorneys' offices
			   - Judges with IPV case experience
			   - Legal aid organizations
			   - Statements on evidence quality

			4. <b>Research Community</b>
			   - Collaborative researchers
			   - Technical experts
			   - Clinical researchers
			   - Statements on scientific significance

			5. <b>Industry Partners</b>
			   - Technology commercialization partners
			   - Healthcare IT companies
			   - Statements on translation potential
			</div>
			""")


if __name__ == "__main__":
	InterviewPrepDashboard().run()
