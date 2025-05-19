import streamlit as st
import logging

# Import panel functions from other modules
from computer_vision_panel import ComputerVisionPanel
from fairness_panel import FairnessPanel
from data_engineering_panel import DataEngineeringPanel
from mobile_deployment_panel import MobileDeploymentPanel
from leadership_panel import LeadershipPanel
from funding_impact_panel import FundingImpactPanel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Dashboard:
	"""Main dashboard class."""

	def __init__(self):
		logging.info("Initializing dashboard...")


	def run(self):
		"""
		Main function to run the Streamlit dashboard.
		It sets up the sidebar navigation and calls the appropriate panel display function.
		"""
		Dashboard.setup_page()

		st.sidebar.title("GMU Postdoc Dashboard")
		st.sidebar.markdown("Navigate through the key areas relevant to the Postdoctoral position at George Mason University, focusing on the EAS-ID project.")

		panels = {
			"🏠 Home"                 : self.display_home,
			"👁️ Computer Vision"      : ComputerVisionPanel().display,
			"⚖️ Fairness in AI"       : FairnessPanel().display,
			"🛠️ Data Engineering"     : DataEngineeringPanel().display,
			"📱 Mobile Deployment"    : MobileDeploymentPanel().display,
			"🤝 Leadership & Teamwork": LeadershipPanel().display,
			"💰 Funding & Impact"     : FundingImpactPanel().display,
		}

		# Initialize panel classes
		cv_panel            = ComputerVisionPanel()
		fairness_panel      = FairnessPanel()
		data_eng_panel      = DataEngineeringPanel()
		mobile_deploy_panel = MobileDeploymentPanel()
		leadership_panel    = LeadershipPanel()
		funding_panel       = FundingImpactPanel()

		# Navigation options
		selection = st.sidebar.radio("Go to", list(panels.keys()))

		# Display the selected section
		page_function = panels[selection]
		page_function()


	@staticmethod
	def setup_page():

		st.set_page_config(
			page_title="GMU Postdoc Interview Dashboard",
			page_icon="🔬",
			layout="wide",
			initial_sidebar_state="expanded",
		)

		st.markdown("""
		<style>
			/* Main app background */
			.main {
				background-color: #f0f2f6; /* Light grey background */
			}

			/* Sidebar styling */
			[data-testid="stSidebar"] {
				background-color: #2c3e50; /* Dark blue-grey */
				color: white;
			}
			[data-testid="stSidebar"] .st-emotion-cache-16txtl3 { /* Sidebar navigation items */
				color: white;
			}
			[data-testid="stSidebar"] .st-emotion-cache-16txtl3:hover {
				background-color: #34495e; /* Slightly lighter blue-grey on hover */
			}
			[data-testid="stSidebar"] .st-emotion-cache-16txtl3.st-emotion-cache-16txtl3 { /* Active item */
				background-color: #1abc9c; /* Teal for active item */
				color: white;
				font-weight: bold;
			}

			/* Titles and headers */
			h1, h2, h3 {
				color: #2c3e50; /* Dark blue-grey for titles */
			}

			/* Custom button style */
			.stButton>button {
				background-color: #1abc9c; /* Teal */
				color: white;
				border-radius: 5px;
				border: none;
				padding: 10px 20px;
				font-weight: bold;
			}
			.stButton>button:hover {
				background-color: #16a085; /* Darker teal on hover */
				color: white;
			}

			/* Info boxes */
			.info-box {
				background-color: #eafaf1; /* Light teal background */
				border-left: 5px solid #1abc9c; /* Teal left border */
				padding: 15px;
				border-radius: 5px;
				margin-bottom: 15px;
				color: #2c3e50;
			}
			.info-box h3 {
				color: #16a085; /* Darker teal for titles within info box */
			}
		</style>
		""", unsafe_allow_html=True)


	def display_home(self):
		"""Displays the home page of the dashboard."""
		st.title("Postdoctoral Researcher: EAS-ID Project Dashboard")
		st.markdown("Welcome! This dashboard provides an overview of key technical and strategic areas relevant to the Postdoctoral Researcher position for the **Equitable and Accessible Software for Injury Detection (EAS-ID)** project at George Mason University.")
		st.markdown("Use the navigation panel on the left to explore different facets of the role, aligned with the project's goals and potential interview questions.")

		st.markdown("---")
		st.header("Project Overview (EAS-ID)")
		st.markdown("""
		The EAS-ID project aims to develop a mobile AI tool that makes bruises visible across all skin tones, leveraging deep learning and multi-spectral imaging (including Alternate Light Sources - ALS). This is a multidisciplinary effort involving nursing science, computer vision research, and AI engineering.
		""")

		st.markdown("<div class='info-box'><h3>Key Responsibilities for the Postdoc:</h3><ul><li>Developing components of the project platform or deep learning application.</li><li>Supervising a team of graduate students and programmers.</li><li>Writing manuscripts and reports.</li><li>Participating in education and mentoring activities.</li></ul></div>", unsafe_allow_html=True)


		st.subheader("Core Team (Illustrative)")
		st.markdown("""
		- **Dr. Katherine Scafide (Nursing):** PI, forensic nursing, injury equity.
		- **Dr. Janusz Wojtusiak (Health Informatics):** AI/data science lead, model design, bias metrics.
		- **Dr. David Lattanzi (Civil/AI Engineering):** Imaging, computer vision pipelines, mobile deployment.
		""")
		st.markdown("---")
		st.markdown("This dashboard simulates discussions and technical considerations you might encounter. Select a topic from the sidebar to begin.")


if __name__ == "__main__":
	Dashboard().run()
