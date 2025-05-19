
import streamlit as st

class FundingImpactPanel:
	"""
	Handles the display and logic for the Funding & Impact section.
	Focuses on positioning a competing-renewal NIH R01.
	"""

	def display(self):
		"""Displays the funding and impact content."""
		st.header("💰 Funding & Impact: Positioning an NIH R01 Renewal")
		st.markdown("""
		**Challenge:** "How would you position a competing-renewal NIH R01 from this work?" (Or similar questions about future funding, grant writing, and broader impact).

		**What they're looking for:** Strategic vision, grant-writing awareness, ability to see the bigger picture and long-term potential of the research.
		""")
		st.markdown("---")

		st.subheader("Context: Building on the EAS-ID Project")
		st.markdown("""
		The current EAS-ID project (funded by a significant gift and NIH AIM-AHEAD supplement) provides a strong foundation. A competing renewal for an NIH R01 grant would aim to build upon its successes, address new research questions, and expand its impact.

		An R01 is a mature research project grant, implying substantial preliminary data, a well-developed research plan, and significant potential impact.
		""")

		st.subheader("Key Elements for Positioning a Competing Renewal R01")
		st.markdown("""
		To successfully position a competing renewal, the application would need to demonstrate:
		1.  **Progress & Accomplishments from the Current Funding:** Clearly articulate the successes of the EAS-ID project (e.g., developed platform, initial validation results, fairness metrics achieved, publications, dissemination).
		2.  **New, Compelling Research Questions:** The renewal shouldn't just be "more of the same." It needs to propose the next logical steps, addressing significant gaps or novel avenues that arose from the initial work.
		3.  **Innovation:** Highlight what is new and innovative in the proposed research (e.g., novel AI techniques, new applications of the technology, unique study designs).
		4.  **Rigorous Approach:** Detail a robust methodology for the proposed research aims.
		5.  **Significant Impact:** Emphasize the potential public health impact, clinical utility, and contribution to scientific knowledge.
		""")

		st.subheader("Potential Directions for an R01 Renewal Based on EAS-ID")
		st.markdown("Here are some hypothetical directions an R01 renewal could take, leveraging the EAS-ID platform:")

		aims_tabs = st.tabs([
			"Aim 1: Advanced AI & Longitudinal Analysis",
			"Aim 2: Clinical Validation & Workflow Integration",
			"Aim 3: Dissemination & Real-World Impact"
		])

		with aims_tabs[0]:
			st.markdown("#### Potential Aim Area 1: Advancing AI Capabilities and Longitudinal Bruise Analysis")
			st.markdown("""
			- **Focus:** Develop next-generation AI models for more nuanced bruise assessment.
			- **Specific Research Questions:**
				- Can we develop AI models to more accurately **estimate bruise age** across diverse skin tones and under various lighting conditions (including different ALS wavelengths)?
				- How can we incorporate **longitudinal imaging data** (tracking a bruise over time) to improve age estimation and differentiate injury evolution from other skin changes?
				- Can we leverage **explainable AI (XAI)** techniques to provide clinicians with more transparent and trustworthy insights into the model's decision-making process for bruise characterization (e.g., highlighting specific visual features indicative of age or severity)?
				- Explore **federated learning or privacy-preserving distributed learning** approaches to train more robust models across multiple institutions without centralizing sensitive image data.
			- **Innovation:** Novel deep learning architectures for spatio-temporal analysis of injuries, advanced XAI for clinical decision support in forensics.
			""")

		with aims_tabs[1]:
			st.markdown("#### Potential Aim Area 2: Large-Scale Clinical Validation and Workflow Integration")
			st.markdown("""
			- **Focus:** Rigorously validate the EAS-ID tool in diverse clinical settings and integrate it into existing workflows.
			- **Specific Research Questions:**
				- What is the diagnostic accuracy (sensitivity, specificity, NPV, PPV) of the enhanced EAS-ID tool in **real-world clinical populations** (e.g., emergency departments, forensic nursing units, pediatric clinics) across a wider range of demographics and injury types?
				- How does the use of EAS-ID impact **clinical decision-making, documentation quality, and time efficiency** compared to standard care?
				- What are the barriers and facilitators to **integrating EAS-ID into existing Electronic Health Record (EHR) systems** and clinical workflows? (e.g., developing FHIR-based integration).
				- Can we conduct a multi-site trial to assess the tool's reliability and generalizability?
			- **Innovation:** Prospective, multi-site clinical validation studies; development of standardized protocols for AI-assisted injury documentation; novel EHR integration strategies for mobile health tools.
			""")

		with aims_tabs[2]:
			st.markdown("#### Potential Aim Area 3: Dissemination, Implementation Science, and Broader Impact")
			st.markdown("""
			- **Focus:** Ensure the tool reaches end-users effectively and explore its utility in new contexts.
			- **Specific Research Questions:**
				- What are the most effective strategies for **training healthcare providers and forensic specialists** to use the EAS-ID tool accurately and ethically?
				- How can we adapt and validate the EAS-ID platform for use in **underserved or resource-limited settings**, potentially including global health applications?
				- Can the underlying AI technology be extended to detect other types of skin injuries or conditions where equitable assessment is a challenge?
				- What are the long-term impacts on **health equity and justice outcomes** (e.g., in cases of intimate partner violence or child abuse) through improved injury documentation?
			- **Innovation:** Application of implementation science frameworks to guide dissemination; exploring novel applications of the core technology; assessing long-term societal benefits.
			""")

		st.markdown("---")
		st.subheader("Structuring the R01 'Argument'")
		st.markdown("""
		A compelling R01 renewal application would weave these elements into a cohesive narrative:

		1.  **Significance:** Reiterate the problem (health disparities in injury detection, challenges in forensic documentation). Highlight how EAS-ID has begun to address this. Then, identify the *new* significant gaps the renewal will tackle.
		2.  **Investigators:** Emphasize the strength and multidisciplinary nature of the team, highlighting the Postdoc's (your) contributions and growing expertise.
		3.  **Innovation:** Clearly state what's novel about the proposed research aims – new algorithms, new clinical applications, new validation methodologies.
		4.  **Approach:** Detail the methods for each Specific Aim. This section must demonstrate feasibility based on preliminary data from the initial EAS-ID project. For example:
			* *"Building on our successfully developed EAS-ID mobile platform (currently achieving X% accuracy in Y population) and our curated dataset of Z images, Aim 1 will develop novel spatio-temporal deep learning models to improve bruise age estimation..."*
		5.  **Environment:** Highlight the supportive institutional environment at George Mason University.
		6.  **Overall Impact:** Conclude by emphasizing how the proposed research will lead to significant advancements in equitable healthcare, forensic science, and AI technology, ultimately benefiting vulnerable populations.

		**Role of the Postdoctoral Researcher in the Renewal:**
		As a Postdoctoral Researcher who has been integral to EAS-ID, you would be positioned as a key contributor to the renewal, potentially:
		- Leading the development and refinement of specific AI models.
		- Overseeing data collection and analysis for certain aims.
		- Contributing significantly to writing the technical sections of the grant.
		- Mentoring junior students involved in the renewed project.
		This sets the stage for your own transition to an independent research career (e.g., future K99/R00 or R21 grants).
		""")

		st.markdown("<div class='info-box'><h3>Key Takeaway:</h3>Positioning a competing R01 renewal involves demonstrating past success, proposing innovative and impactful new research that logically extends the current work, and presenting a rigorous plan. It's about showing NIH that further investment will yield significant scientific and societal returns. The Postdoc's role is to show how their expertise and contributions will be critical to achieving these future goals.</div>", unsafe_allow_html=True)
