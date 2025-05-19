# Standard library imports
import os
import tempfile
from datetime import datetime

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components
from pyvis.network import Network
import io
import base64
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# Local application imports
from core.data_module import DatabaseSchema, FHIRDataModel
from core.deployment_module import DeploymentComparison
from core.fairness_module import FairnessMetrics, generate_fairness_report
from core.leadership_module import TeamManagement
from core.vision_module import BruiseDetectionModel, apply_als_filter, preprocess_image


class ComputerVisionPage:
	def __init__(self):
		pass

	def render(self):
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
				st.image(pil_img, use_container_width=True)

				# Display processed image
				st.markdown("#### Processed Image")
				st.image(processed_img, use_container_width=True)

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

					st.image(seg_img, use_container_width=True)

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
