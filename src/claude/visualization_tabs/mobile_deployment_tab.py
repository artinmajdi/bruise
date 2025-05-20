# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Local application imports
from core.deployment_module import DeploymentComparison

class MobileDeploymentPage:
	def __init__(self):
		pass

	def render(self):
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
				'border-color'    : '#dddddd',
				'border-style'    : 'solid',
				'border-width'    : '1px',
				'text-align'      : 'left',
				'padding'         : '8px'
			})

			# Apply specific styling to the header
			styled_df = styled_df.set_table_styles([
				{'selector': 'th', 'props': [
					('background-color', '#006633'),
					('color'           , 'white'       ),
					('font-weight'     , 'bold'        ),
					('border'          , '1px solid #dddddd'),
					('text-align'      , 'left'        ),
					('padding'         , '8px'         )
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
			cloud_values     = [2, 3, 5, 4, 2, 1, 5, 3, 4]
			hybrid_values    = [4, 4, 4, 3, 4, 4, 4, 4, 3]

			# Add the first value at the end to close the loop
			categories       = categories + [categories[0]]
			on_device_values = on_device_values + [on_device_values[0]]
			cloud_values     = cloud_values + [cloud_values[0]]
			hybrid_values    = hybrid_values + [hybrid_values[0]]

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

			# Add End column for the timeline
			tasks_df['End'] = tasks_df['Start'] + tasks_df['Duration']

			# Create Gantt chart
			fig = px.timeline(
				tasks_df,
				x_start="Start",
				x_end="End",
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
