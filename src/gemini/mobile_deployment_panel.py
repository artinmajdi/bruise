import streamlit as st

class MobileDeploymentPanel:
	"""
	Handles the display and logic for the Mobile Deployment section.
	Focuses on on-device vs. cloud inference.
	"""

	def display(self):
		"""Displays the mobile deployment content."""
		st.header("📱 Mobile Deployment: On-Device vs. Cloud Inference")
		st.markdown("""
		**Challenge:** "Inference on-device or in the cloud—convince us."

		**What they're looking for:** Systems-level thinking, privacy-latency trade-offs, understanding of model deployment constraints.
		""")
		st.markdown("---")

		st.subheader("The Core Question: Where Should the AI Model Run?")
		st.markdown("""
		For the EAS-ID mobile app, a key architectural decision is whether the deep learning model for bruise detection runs directly on the user's smartphone (on-device) or if the images are sent to a server for analysis (cloud-based). A hybrid approach is also possible.
		""")

		# Comparison Table
		st.subheader("Comparison: On-Device vs. Cloud-Based Inference")

		col1_title = "**Factor**"
		col2_title = "**On-Device Inference**"
		col3_title = "**Cloud-Based Inference**"

		# Using st.columns for a more structured layout
		header_cols = st.columns([1.5, 2, 2])
		header_cols[0].markdown(f"<h6>{col1_title}</h6>", unsafe_allow_html=True)
		header_cols[1].markdown(f"<h6>{col2_title}</h6>", unsafe_allow_html=True)
		header_cols[2].markdown(f"<h6>{col3_title}</h6>", unsafe_allow_html=True)
		st.markdown("---", unsafe_allow_html=True) # Visual separator

		data = [
			("**Latency**",
			 "Low (milliseconds). Results are near-instant.",
			 "Higher (seconds, depends on network). Can be an issue for real-time feedback."),
			("**Privacy & Security**",
			 "High. Sensitive image data (PHI) does not leave the device. Reduces HIPAA compliance burden for data in transit/at rest on servers for inference.",
			 "Lower by default. Images (PHI) are transmitted to a server. Requires robust encryption, secure infrastructure, and strict HIPAA compliance for the server-side processing and storage."),
			("**Connectivity**",
			 "Works offline. No internet connection required for inference.",
			 "Requires a stable internet connection."),
			("**Model Complexity & Size**",
			 "Constrained. Models must be smaller, optimized (quantization, pruning) for mobile CPUs/GPUs/NPUs. May lead to slight accuracy trade-offs.",
			 "Less constrained. Can deploy larger, more complex, and potentially more accurate models using powerful server hardware (GPUs/TPUs)."),
			("**Computational Cost (User)**",
			 "Higher. Uses phone's battery and processing power.",
			 "Lower. Phone mainly handles image capture and UI."),
			("**Computational Cost (Provider)**",
			 "Low. No server costs for inference computation.",
			 "Higher. Requires server infrastructure, maintenance, and scaling, incurring ongoing operational costs."),
			("**Model Updates & Management**",
			 "More complex. Updates require distributing new app versions or model files to all devices.",
			 "Easier. Models can be updated centrally on the server without requiring users to update their app (unless API changes). Allows for A/B testing and rapid iteration."),
			("**Scalability (User Load)**",
			 "Highly scalable. Each device handles its own load.",
			 "Requires careful server-side scaling to handle concurrent users."),
			("**Data Collection for Retraining (Optional)**",
			 "More complex to get inference data back for model improvement (requires explicit user consent and upload mechanism).",
			 "Easier to collect data (with consent) from images processed on the server for continuous model improvement and monitoring.")
		]

		for row in data:
			cols = st.columns([1.5, 2, 2])
			cols[0].markdown(row[0], unsafe_allow_html=True)
			cols[1].markdown(f"<div style='font-size: 0.95em;'>{row[1]}</div>", unsafe_allow_html=True)
			cols[2].markdown(f"<div style='font-size: 0.95em;'>{row[2]}</div>", unsafe_allow_html=True)
			st.markdown("---", unsafe_allow_html=True)


		st.subheader("Making the Decision for EAS-ID")
		st.markdown("""
		The optimal choice for EAS-ID depends on prioritizing these factors. Given the sensitive nature of bruise images (often related to violence and involving PHI) and the potential need for use in varied environments (which might have poor connectivity):

		**Arguments for On-Device Inference:**
		* **Privacy:** This is paramount. Keeping images on the device significantly reduces risks and simplifies HIPAA compliance for the inference step.
		* **Offline Capability:** Crucial for usability in settings without reliable internet (e.g., remote areas, certain clinical environments).
		* **Low Latency:** Immediate feedback can be important for the user (e.g., a nurse assessing an injury).

		**Arguments for Cloud-Based Inference:**
		* **Model Power:** Ability to use more computationally intensive and potentially more accurate models. This could be critical for detecting very faint bruises or complex patterns.
		* **Centralized Updates:** Easier to improve and deploy model updates.
		* **Controlled Environment:** Consistent processing environment, unlike diverse mobile hardware.

		**A Hybrid Approach could be considered:**
		1.  **Screening/Triage On-Device:** A smaller, faster model runs on-device for initial assessment or to guide image capture.
		2.  **Optional Cloud Refinement:** If the on-device model is uncertain, or for specific cases requiring higher accuracy, the user (with explicit consent) could opt to send images for more powerful cloud-based analysis.
		3.  **Federated Learning (Advanced):** Train models locally on devices and send aggregated, anonymized model updates (not raw data) to a central server for creating an improved global model. This is complex to implement but offers strong privacy.

		**Convincing Argument for EAS-ID (Leaning towards On-Device or Hybrid):**

		*"For the EAS-ID platform, I would advocate for **prioritizing on-device inference** due to the paramount importance of **patient privacy and data security**, especially given the sensitive context of injury documentation, often related to intimate partner violence. Processing images directly on the device minimizes the transmission of PHI, reducing HIPAA compliance complexities and the risk of data breaches. Furthermore, on-device inference ensures **offline functionality**, which is critical for usability in diverse clinical settings or field situations where internet connectivity may be unreliable. The **low latency** also provides immediate feedback to the clinician.*

		*While cloud-based inference allows for more powerful models, advancements in model optimization techniques (like quantization, pruning, and the use of mobile-specific NPUs through frameworks like TensorFlow Lite or Core ML) make it increasingly feasible to deploy accurate and efficient models on modern smartphones. We can develop a highly optimized model that balances accuracy with on-device performance constraints.*

		*A **phased or hybrid approach** could also be viable. An initial version could rely on robust on-device inference for core functionality. For future enhancements, or if a particularly challenging case requires more computational power, we could explore an **opt-in, consent-based mechanism for secure cloud-based analysis** of specific, de-identified or pseudonymized images, ensuring full transparency with the user. This allows us to leverage the strengths of both approaches while maintaining a strong privacy-first posture."*
		""")

		st.markdown("<div class='info-box'><h3>Key Takeaway:</h3>The decision involves a trade-off. For healthcare applications dealing with sensitive data like EAS-ID, a strong case can be made for on-device inference due to privacy, offline capability, and low latency. However, the limitations on model size and complexity must be carefully managed. A hybrid approach might offer the best of both worlds in the long run.</div>", unsafe_allow_html=True)
