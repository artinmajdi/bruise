# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


class FundingPage:
	def __init__(self):
		pass

	def render(self):
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

