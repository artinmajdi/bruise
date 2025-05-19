
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


from pages import (
    HomePage,
    ComputerVisionPage,
    FairnessPage,
    DataEngineeringPage,
    MobileDeploymentPage,
    LeadershipPage,
    FundingPage,
)

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
		self.vision_model          = BruiseDetectionModel()
		self.fairness_metrics      = FairnessMetrics()
		self.database_schema       = DatabaseSchema()
		self.fhir_model            = FHIRDataModel()
		self.deployment_comparison = DeploymentComparison()
		self.team_management       = TeamManagement()

	def run(self):
		# Add custom CSS
		self.apply_custom_css()

		# Sidebar for navigation
		self.create_sidebar()

		# Main content
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

if __name__ == "__main__":
	InterviewPrepDashboard().run()
