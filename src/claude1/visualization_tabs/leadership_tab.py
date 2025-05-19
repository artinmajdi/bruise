# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Local application imports
from core.leadership_module import TeamManagement


class LeadershipPage:
	def __init__(self):
		pass

	def render(self):
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
