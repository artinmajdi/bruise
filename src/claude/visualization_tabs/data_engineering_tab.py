# Standard library imports
import os
import tempfile

# Third-party imports
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components
from pyvis.network import Network


# Local application imports
from core.data_module import DatabaseSchema, FHIRDataModel
from core.deployment_module import DeploymentComparison


class DataEngineeringPage:
	def __init__(self):
		pass

	def render(self):

		def _fhir_integration_architecture_graph():


			st.markdown('<div class="section-header">FHIR Integration Architecture</div>', unsafe_allow_html=True)

			# Create a network
			net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333")
			net.toggle_hide_edges_on_drag(False)
			net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)

			# Define node groups for styling
			groups = {
				"External": {"color": "#FF9800", "shape": "box"},  # Orange
				"Core": {"color": "#2196F3", "shape": "circle"},   # Blue
				"Client": {"color": "#4CAF50", "shape": "box"},    # Green
				"Support": {"color": "#9C27B0", "shape": "hexagon"}  # Purple
			}

			# Define architecture components
			components = {
				# Layer 1: External Systems
				"EHR Systems": {"type": "External", "level": 1},
				"Mobile App": {"type": "Client", "level": 1},
				# Layer 2: Core Services
				"FHIR Server": {"type": "Core", "level": 2},
				"EAS-ID API": {"type": "Core", "level": 2},
				# Layer 3: Data Layer
				"Image Repository": {"type": "Core", "level": 3},
				"AI Pipeline": {"type": "Core", "level": 3},
				# Layer 4: Analytics & Research
				"Analytics": {"type": "Support", "level": 4},
				"Research Portal": {"type": "Support", "level": 4}
			}

			# Add nodes with proper styling
			for name, attrs in components.items():
				group = attrs["type"]
				net.add_node(
					name,
					label=name,
					title=f"<b>{name}</b><br>Type: {attrs['type']}",
					color=groups[group]["color"],
					shape=groups[group]["shape"],
					size=30,
					level=attrs["level"],
					borderWidth=2,
					borderWidthSelected=4,
					font={"size": 14, "face": "Arial"}
				)

			# Define connections with types
			connections = [
				# External connections
				("EHR Systems", "FHIR Server", "HL7 FHIR", "#FF9800"),
				("EAS-ID API", "Mobile App", "SMART on FHIR", "#4CAF50"),
				# Core connections
				("FHIR Server", "EAS-ID API", "REST API", "#2196F3"),
				("FHIR Server", "Image Repository", "DICOMweb", "#2196F3"),
				# AI and Data connections
				("EAS-ID API", "AI Pipeline", "Internal API", "#9C27B0"),
				("AI Pipeline", "Image Repository", "Secure Access", "#9C27B0"),
				# Analytics connections
				("EAS-ID API", "Analytics", "Event Stream", "#FF5722"),
				("Image Repository", "Research Portal", "De-identified", "#8BC34A")
			]

			# Add edges with proper styling
			for src, tgt, label, color in connections:
				is_api = "API" in label
				net.add_edge(
					src, tgt,
					title=label,
					label=label,
					color=color,
					width=2,
					dashes=not is_api,
					smooth={"type": "curvedCW", "roundness": 0.2},
					font={"size": 10, "color": color, "strokeWidth": 0, "strokeColor": "#ffffff"}
				)

			# Set physics options for better layout
			net.set_options("""
			var options = {
				"physics": {
				"hierarchicalRepulsion": {
					"centralGravity": 0.0,
					"springLength": 150,
					"springConstant": 0.01,
					"nodeDistance": 200,
					"damping": 0.09
				},
				"solver": "hierarchicalRepulsion",
				"stabilization": {
					"enabled": true,
					"iterations": 1000
				}
				},
				"layout": {
				"hierarchical": {
					"enabled": true,
					"levelSeparation": 150,
					"nodeSpacing": 200,
					"treeSpacing": 200,
					"direction": "UD",
					"sortMethod": "directed"
				}
				},
				"interaction": {
				"hover": true,
				"navigationButtons": true,
				"keyboard": true
				}
			}
			""")

			# Generate the HTML file in a temporary directory
			with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
				temp_path = tmp.name
				net.save_graph(temp_path)

			# Read the HTML content
			with open(temp_path, 'r', encoding='utf-8') as f:
				html_content = f.read()

			# Add custom CSS to improve the appearance
			html_content = html_content.replace('</head>', '''
			<style>
				body {
					font-family: Arial, sans-serif;
				}
				.vis-network {
					border: 1px solid #e1e1e1;
					border-radius: 8px;
					box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
				}
				.header {
					text-align: center;
					padding: 10px;
					font-size: 18px;
					font-weight: bold;
					color: #333;
					background-color: #f9f9f9;
					border-bottom: 1px solid #e1e1e1;
					border-radius: 8px 8px 0 0;
				}
				.footer {
					text-align: right;
					padding: 5px 10px;
					font-size: 10px;
					color: #e0e0e0;
					background-color: #f9f9f9;
					border-top: 1px solid #e1e1e1;
					border-radius: 0 0 8px 8px;
				}
				.legend {
					position: absolute;
					top: 10px;
					right: 10px;
					background: white;
					padding: 10px;
					border: 1px solid #e1e1e1;
					border-radius: 5px;
					font-size: 12px;
					z-index: 1000;
				}
				.legend-item {
					display: flex;
					align-items: center;
					margin-bottom: 5px;
				}
				.legend-color {
					width: 15px;
					height: 15px;
					margin-right: 5px;
					border-radius: 3px;
				}
			</style>
			</head>''')

			# Add header, footer, and legend
			html_content = html_content.replace('<body>', '''<body>
			<div class="header">FHIR Integration Architecture</div>
			<div class="legend">
				<div class="legend-item"><div class="legend-color" style="background-color:#FF9800;"></div>External Systems</div>
				<div class="legend-item"><div class="legend-color" style="background-color:#4CAF50;"></div>Client Applications</div>
				<div class="legend-item"><div class="legend-color" style="background-color:#2196F3;"></div>Core Services</div>
				<div class="legend-item"><div class="legend-color" style="background-color:#9C27B0;"></div>Support Systems</div>
			</div>''')

			html_content = html_content.replace('</body>', '''<div class="footer">EAS-ID Platform</div></body>''')

			# Display the network graph in Streamlit
			st_components.html(html_content, height=700)

			# Clean up the temporary file
			os.unlink(temp_path)

		def _sample_fhir_resource():
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
				"Patient"         : {"x": 1, "y": 3, "type": "FHIR"},
				"Practitioner"    : {"x": 3, "y": 1, "type": "FHIR"},
				"DiagnosticReport": {"x": 3, "y": 3, "type": "FHIR"},
				"ImagingStudy"    : {"x": 5, "y": 3, "type": "FHIR"},
				"Observation"     : {"x": 3, "y": 5, "type": "FHIR"},
				"Media"           : {"x": 5, "y": 5, "type": "FHIR"},
				"BruiseImage"     : {"x": 7, "y": 5, "type": "Custom"},
				"DetectionResult" : {"x": 7, "y": 3, "type": "Custom"},
				"AuditEvent"      : {"x": 5, "y": 1, "type": "FHIR"},
			}

			# Define relationships
			relationships = [
				("Patient"          , "DiagnosticReport") ,
				("Practitioner"     , "DiagnosticReport") ,
				("DiagnosticReport" , "ImagingStudy")     ,
				("DiagnosticReport" , "Observation")      ,
				("ImagingStudy"     , "Media")            ,
				("Media"            , "BruiseImage")      ,
				("BruiseImage"      , "DetectionResult")  ,
				("ImagingStudy"     , "DetectionResult")  ,
				("Practitioner"     , "AuditEvent")       ,
				("DiagnosticReport" , "AuditEvent")
			]

			# Colors for different entity types
			colors = {
				"FHIR": "#2196F3",  # Blue for standard FHIR resources
				"Custom": "#4CAF50"  # Green for custom extensions
			}

			# Plot entities
			for entity, attrs in entities.items():
				fig.add_trace(go.Scatter(
					x            = [attrs["x"]],
					y            = [attrs["y"]],
					mode         = "markers+text",
					marker       = dict(size=50, color=colors[attrs["type"]]),
					text         = [entity],
					textposition = "middle center",
					textfont     = dict(color="white", size=10),
					name         = entity
				))

			# Plot relationships
			for rel in relationships:
				fig.add_trace(go.Scatter(
					x          = [entities[rel[0]]["x"], entities[rel[1]]["x"]],
					y          = [entities[rel[0]]["y"], entities[rel[1]]["y"]],
					mode       = "lines",
					line       = dict(width=1, color="gray"),
					showlegend = False
				))

			# Update layout
			fig.update_layout(
				title        = "FHIR-Based Database Schema for Bruise Detection",
				showlegend   = False,
				height       = 500,
				margin       = dict(l=20, r=20, t=40, b=20),
				plot_bgcolor = "white",
				xaxis        = dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis        = dict(showgrid=False, zeroline=False, showticklabels=False)
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
			fhir_col1, fhir_col2 = st.columns([1, 1])

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
				_sample_fhir_resource()

			# FHIR integration architecture
			_fhir_integration_architecture_graph()

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
