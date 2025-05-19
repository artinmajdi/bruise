import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import io

# --- Modular Classes ---

class ComputerVisionSimulator:
    """Simulates bruise detection/segmentation challenges."""
    def __init__(self):
        self.skin_tone_levels = ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark']
        self.als_effects = {
            'Blue Light': "Enhances visibility of superficial bruises.",
            'UV Light': "May reveal older or deeper bruises not visible under white light.",
            'White Light': "Standard examination light, less effective on darker skin tones."
        }

    def simulate_detection_score(self, skin_tone_index, bruise_faintness, als_type):
        """
        Simulates a detection confidence score.
        Score is lower for darker skin and fainter bruises under white light,
        improved by appropriate ALS.
        """
        base_score = 1.0 - (bruise_faintness / 100.0) # Higher faintness -> lower base score
        skin_penalty = (skin_tone_index / (len(self.skin_tone_levels) - 1)) * 0.3 # Penalty increases with darker skin

        score = base_score - skin_penalty

        # Adjust based on ALS
        if als_type == 'White Light':
             # White light penalty is highest on darker skin
             als_adjustment = -(skin_tone_index / (len(self.skin_tone_levels) - 1)) * 0.4
        elif als_type == 'Blue Light':
             # Blue light helps, especially on darker tones, but less for very faint
             als_adjustment = (skin_tone_index / (len(self.skin_tone_levels) - 1)) * 0.3 + (1.0 - bruise_faintness/100.0) * 0.1
        elif als_type == 'UV Light':
             # UV helps with faint/older bruises, less dependent on skin tone directly, but score might still be lower overall
             als_adjustment = (1.0 - bruise_faintness/100.0) * 0.2

        score = np.clip(score + als_adjustment, 0.1, 0.95) # Keep score within a reasonable range

        return score

    def get_als_description(self, als_type):
        return self.als_effects.get(als_type, "Unknown light source.")


class FairnessMetrics:
    """Calculates and displays simulated fairness metrics."""
    def __init__(self, skin_tones):
        self.skin_tones = skin_tones

    def simulate_performance_data(self, overall_detection_rate=0.85, base_unevenness=0.1):
        """Simulates detection rates for different skin tones."""
        n_groups = len(self.skin_tones)
        # Create a base rate that decreases slightly with skin tone index
        base_rates = np.linspace(overall_detection_rate + base_unevenness/2, overall_detection_rate - base_unevenness/2, n_groups)
        # Add some random noise
        simulated_rates = np.clip(base_rates + np.random.randn(n_groups) * 0.03, 0.05, 0.95)
        return dict(zip(self.skin_tones, simulated_rates))

    def calculate_demographic_parity_difference(self, detection_rates):
        """
        Calculates the Demographic Parity Difference.
        Difference between the highest and lowest detection rate across groups.
        Lower is better (closer to 0).
        """
        rates = list(detection_rates.values())
        if not rates:
            return 0
        return max(rates) - min(rates)

    def display_rates(self, detection_rates):
        df = pd.DataFrame(list(detection_rates.items()), columns=['Skin Tone', 'Detection Rate']).set_index('Skin Tone')
        st.dataframe(df.T)

    def display_metrics(self, detection_rates):
        dpd = self.calculate_demographic_parity_difference(detection_rates)
        st.metric(label="Demographic Parity Difference (lower is better)", value=f"{dpd:.3f}")
        st.info("Demographic Parity Difference measures the difference between the highest and lowest positive prediction rates across different groups (skin tones in this case). An ideal model would have a DPD close to 0, meaning the detection rate is similar for all skin tones.")

class DatabaseSchema:
    """Describes a simplified FHIR-based database schema."""
    def __init__(self):
        self.schema_description = """
        **Simplified FHIR-based Schema for Bruise Data**

        This schema outlines key FHIR Resources relevant to storing bruise imaging and associated clinical data.

        1.  **Patient Resource:** Stores demographic and administrative information about the individual.
            *   `id`: Unique patient identifier.
            *   `identifier`: Medical record number, etc.
            *   `name`: Patient name (consider pseudonymization).
            *   `gender`: Patient gender.
            *   `birthDate`: Patient date of birth.
            *   `extension` (for race/ethnicity, skin tone, etc. - handle carefully for privacy/standardization).

        2.  **Encounter Resource:** Represents a clinical encounter where the injury was documented.
            *   `id`: Unique encounter identifier.
            *   `status`: e.g., 'finished'.
            *   `class`: e.g., 'AMB' (ambulatory).
            *   `subject`: Reference to the `Patient` resource.
            *   `period`: Start and end time of the encounter.
            *   `location`: Reference to the location of the encounter.

        3.  **Condition Resource:** Documents the observed injury (bruise).
            *   `id`: Unique condition identifier.
            *   `subject`: Reference to the `Patient` resource.
            *   `encounter`: Reference to the `Encounter` resource.
            *   `code`: SNOMED CT or other code for 'bruise'.
            *   `bodySite`: Location of the bruise on the body.
            *   `onsetDateTime`: Estimated time of injury (crucial for age estimation).
            *   `severity`: e.g., 'Mild', 'Moderate', 'Severe'.
            *   `note`: Free text description by the clinician.

        4.  **ImagingStudy Resource:** Represents the imaging study performed (e.g., bruise photography session). [10, 18, 35]
            *   `id`: Unique imaging study identifier.
            *   `identifier`: Study Instance UID (from DICOM, if applicable).
            *   `status`: e.g., 'available'.
            *   `subject`: Reference to the `Patient` resource.
            *   `started`: Date/time the study started. [10]
            *   `numberOfSeries`: Number of image series in the study.
            *   `numberOfInstances`: Total number of images.
            *   `procedureCode`: Code for the imaging procedure (e.g., 'Bruise Photography').
            *   `endpoint`: Reference to where the images are stored (e.g., secure file server, cloud storage).

        5.  **Observation Resource:** Could store specific measurements or AI inferences (e.g., estimated bruise age, dimensions, colorimetry data, AI detection confidence).
            *   `id`: Unique observation identifier.
            *   `subject`: Reference to the `Patient` resource.
            *   `encounter`: Reference to the `Encounter` resource.
            *   `code`: Code for the type of observation (e.g., 'Bruise Age Estimate', 'Bruise Area').
            *   `valueQuantity` or `valueString` or `valueDateTime`: The actual measurement or inference result.
            *   `method`: Describes how the observation was made (e.g., 'AI Model vX.Y', 'Clinician Assessment').

        6.  **Media Resource:** Represents the image data itself.
            *   `id`: Unique media identifier.
            *   `status`: e.g., 'completed'.
            *   `subject`: Reference to the `Patient` resource.
            *   `type`: 'photo'.
            *   `content`: Binary data of the image (or a link/reference to it).
            *   `issued`: When the image was captured.
            *   `bodySite`: Location of the image.

        **Relationships:**
        *   `Encounter`, `Condition`, `ImagingStudy`, `Observation`, and `Media` all link back to the `Patient`.
        *   `Condition`, `ImagingStudy`, `Observation` may link to the `Encounter`.
        *   `Observation` can link to the `Media` it was derived from.
        *   `ImagingStudy` points to the storage location (`endpoint`) of related `Media`.

        **Security & Privacy (HIPAA, 21 CFR Part 11):**
        *   **Access Control:** Role-based access to different resources.
        *   **De-identification/Pseudonymization:** Remove direct identifiers from data used for research/AI training.
        *   **Audit Trails:** Log all access and modifications to data.
        *   **Encryption:** Encrypt data at rest and in transit.
        *   **Secure Storage:** Store image data in a HIPAA-compliant environment.
        *   **Consent Management:** Track patient consent for data usage.
        *   **Compliance with 21 CFR Part 11:** Ensure electronic records and signatures are trustworthy, reliable, and equivalent to paper records.

        This schema provides a structured way to store and link diverse data types relevant to the bruise detection project, facilitating both clinical use and research analysis.
        """

    def get_schema_description(self):
        return self.schema_description

class DeploymentStrategizer:
    """Discusses mobile vs. cloud inference trade-offs."""
    def __init__(self):
        self.tradeoffs = {
            "On-Device Inference": {
                "Pros": [
                    "Low Latency (near real-time)",
                    "Enhanced Privacy (data stays on device, aids HIPAA compliance)",
                    "Offline Capability (no network needed)",
                    "Lower Bandwidth Requirements (only results transmitted)",
                ],
                "Cons": [
                    "Limited Computational Power (depends on device hardware)",
                    "Limited Memory/Storage (model size is a constraint)",
                    "Model Updates (requires app updates)",
                    "Battery Consumption",
                    "Development Complexity (optimizing for various devices)"
                ]
            },
            "Cloud Inference": {
                "Pros": [
                    "High Computational Power (leverage powerful GPUs)",
                    "Larger Model Support",
                    "Easier Model Updates (server-side)",
                    "Lower Device Battery Consumption",
                ],
                "Cons": [
                    "Higher Latency (network dependency)",
                    "Requires Network Connectivity",
                    "Privacy Concerns (data transmitted to cloud, requires robust security measures)",
                    "Higher Bandwidth Requirements (uploading images)",
                    "Ongoing Cloud Costs"
                ]
            }
        }

    def display_tradeoffs(self, strategy):
        if strategy in self.tradeoffs:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Pros")
                for item in self.tradeoffs[strategy]["Pros"]:
                    st.write(f"- {item}")
            with col2:
                st.subheader("Cons")
                for item in self.tradeoffs[strategy]["Cons"]:
                    st.write(f"- {item}")
        else:
            st.warning("Select a deployment strategy.")

class LeadershipSimulator:
    """Placeholder for leadership discussion."""
    def get_leadership_prompt(self):
        return """
        Leading a multidisciplinary team of researchers, software developers, engineers, and clinicians requires effective communication, goal alignment, and conflict resolution.

        **Consider the following scenario:** You are supervising a graduate student programmer who is struggling to implement a specific deep learning model architecture required by the computer vision experts, and the deadline for integrating it into the mobile prototype is approaching. The nursing team is also requesting features that require this model's output.

        How would you approach this situation to ensure the student is supported, the technical goals are met, and the project timeline stays on track while managing expectations from other team members?
        """

class FundingStrategizer:
    """Placeholder for funding and impact discussion."""
    def get_funding_text(self):
        return """
        **Strategic Vision for Funding and Impact**

        The EAS-ID platform has significant momentum with the recent generous gift and DOJ funding. [2, 4] To ensure long-term sustainability and impact, future funding strategies should focus on demonstrating clinical utility, expanding reach, and exploring new applications.

        *   **Competing Renewal NIH R01:** A future R01 proposal could focus on a large-scale clinical trial validating the platform's effectiveness and fairness in diverse clinical settings. This would require robust data demonstrating improved bruise detection, documentation, and impact on patient care and legal outcomes compared to current methods. Emphasizing the AI's role in reducing disparities would be crucial for NIH priorities (e.g., AIM-AHEAD). [4, 15, 16]
        *   **NIH K99/R00 or NSF SmartHealth:** As a postdoc, transitioning to an independent career could be supported by these grants. A K99/R00 would focus on developing the candidate's research program, potentially extending the AI's capabilities (e.g., automated age estimation refinement, 3D reconstruction) and linking it to broader health outcomes research. An NSF SmartHealth proposal could emphasize the technological innovation, usability, and public health impact.
        *   **Philanthropic Support:** Continued engagement with donors interested in intimate partner violence prevention and health equity could provide flexible funding for platform expansion, training programs for clinicians, or specific feature development.
        *   **Industry Partnerships:** Collaborating with electronic health record (EHR) vendors or mobile health technology companies could facilitate integration and wider adoption.

        Highlighting the project's impact on vulnerable populations, its interdisciplinary nature, and the rigorous scientific approach will be key to securing future funding and ensuring the platform's sustained success in improving injury detection and care.
        """

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="GMU Bruise Detection Postdoc Simulator")

st.title("George Mason University Postdoc Interview Simulator: Bruise Detection Project")
st.subheader("Exploring key technical and research areas")

st.markdown("""
Welcome to this interactive simulator designed to help you prepare for the George Mason University Postdoctoral Fellow interview focused on the bruise detection project. This tool explores several key areas mentioned in the job description and project details, allowing you to think through potential interview questions and demonstrate your expertise.

Navigate through the sections below to engage with different aspects of the project.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Explore Project Areas")
section = st.sidebar.radio(
    "Go to",
    ['Project Overview', 'Computer Vision', 'Fairness & Bias', 'Data Engineering', 'Mobile Deployment', 'Leadership & Funding']
)

# --- Project Overview ---
if section == 'Project Overview':
    st.header("Project Overview: EAS-ID Platform")
    st.markdown("""
    You are applying for a Postdoctoral Fellow position at George Mason University to support the Equitable and Accessible Software for Injury Detection (EAS-ID) platform project. This ambitious, interdisciplinary effort aims to develop a mobile AI tool that utilizes deep learning and alternate light sources (ALS) to detect and document bruises across all skin tones, improving care and legal recourse for victims of violence. [2, 3, 4]

    The core team includes:
    *   **Dr. Katherine Scafide:** Nurse-scientist, PI, expert in forensic nursing and ALS bruise detection. [4, 5, 15]
    *   **Dr. Janusz Wojtusiak:** Informatics scholar, leads AI/data science, expertise in machine learning and health informatics. [6, 9, 19]
    *   **Dr. David Lattanzi:** Civil/AI engineer, focuses on imaging, computer vision, and mobile deployment. [8, 24]

    The project is well-funded and has garnered national attention. [2, 4] Your role will be critical in bridging deep learning engineering with the clinical and forensic aspects of the project.
    """)
    st.image("https://www2.gmu.edu/sites/gmu.edu/files/styles/featured_media_teaser_wide/public/2024-04/bruise-detection-main.jpg", caption="GMU Bruise Detection Team. Photo by Ron Aira/Office of University Branding.", use_container_width=True)
    st.markdown("[Learn more about the EAS-ID project here.](https://publichealth.gmu.edu/research/bruise-detection-system)") # Linking to a relevant GMU page

# --- Computer Vision Section ---
elif section == 'Computer Vision':
    st.header("Computer Vision: Detecting Bruises Across Skin Tones")
    st.markdown("""
    A core technical challenge is reliably detecting and segmenting bruises, especially faint ones or those on darker skin tones, often utilizing alternate light sources (ALS).

    **Interview Question:** "How would you segment a faint bruise on dark skin under ALS illumination?"

    Use the controls below to simulate the challenge:
    """)

    cv_simulator = ComputerVisionSimulator()

    skin_tone_selection = st.select_slider(
        "Select Simulated Skin Tone (Fitzpatrick Type)",
        options=cv_simulator.skin_tone_levels,
        value='Dark'
    )
    skin_tone_index = cv_simulator.skin_tone_levels.index(skin_tone_selection)

    bruise_faintness_percent = st.slider(
        "Simulate Bruise Faintness (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="Higher percentage means a fainter, harder-to-detect bruise."
    )

    als_selection = st.radio(
        "Select Simulated Illumination Source",
        options=list(cv_simulator.als_effects.keys()),
        index=2 # Default to White Light
    )

    st.info(f"**ALS Effect:** {cv_simulator.get_als_description(als_selection)}")

    # Simulate detection score
    detection_score = cv_simulator.simulate_detection_score(
        skin_tone_index,
        bruise_faintness_percent,
        als_selection
    )

    st.metric(label="Simulated AI Detection Confidence Score", value=f"{detection_score:.2f}")

    st.markdown("""
    **Your Approach:**

    To answer the interview question, consider discussing:
    *   **Data Preprocessing:** How to handle multi-spectral images from ALS (e.g., combining channels, normalization).
    *   **Model Architectures:** Which deep learning models are suitable for segmentation (e.g., U-Net, Mask R-CNN) and how to adapt them for low-contrast medical images. Mentioning architectures like EfficientNet or Vision Transformers could show awareness of recent advancements.
    *   **Training Strategies:** Addressing class imbalance (bruised vs. non-bruised pixels), data augmentation (simulating variations in lighting, skin tone, bruise appearance), potentially using synthetic data generated by GANs. [27]
    *   **Handling ALS:** How different wavelengths reveal different bruise characteristics and how to leverage this information in the model. Blue and violet lights have been shown to be effective. [20]
    *   **Evaluation:** Metrics for segmentation (e.g., Dice score, IoU) and how to evaluate performance specifically for faint bruises and across different skin tones.
    *   **Explainability:** How to visualize which parts of the image the model focuses on (e.g., Grad-CAM) to build clinician trust.

    *Example thinking:* Detecting a faint bruise on dark skin under ALS is challenging due to low contrast and potential artifacts from the light source. I would start with advanced image preprocessing tailored to ALS data, perhaps using techniques to enhance bruise visibility before feeding into a segmentation model. A U-Net architecture, commonly used in medical imaging for segmentation, could be a good starting point, potentially modified to handle multi-channel ALS input. Training would require a carefully curated dataset with diverse skin tones and bruise types under various ALS, using data augmentation and potentially synthetic data to address imbalances.
    """)

# --- Fairness & Bias Section ---
elif section == 'Fairness & Bias':
    st.header("Fairness & Bias: Ensuring Equitable Performance")
    st.markdown("""
    A critical goal of the EAS-ID project is to ensure equitable bruise detection across all skin tones. AI models can inherit biases from training data. [27]

    **Interview Question:** "Describe a metric you would report to show equitable performance."

    Use the simulation below to explore detection rates and fairness metrics.
    """)

    fairness_simulator = FairnessMetrics(skin_tones=['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'])

    st.subheader("Simulated Detection Rates by Skin Tone")
    st.markdown("Adjust the 'Unevenness' slider to simulate different levels of bias in the model's performance across skin tones.")

    overall_rate = st.slider("Overall Simulated Detection Rate", min_value=0.5, max_value=0.99, value=0.85, step=0.01)
    unevenness = st.slider("Simulate Performance Unevenness/Bias", min_value=0.0, max_value=0.4, value=0.1, step=0.01)

    simulated_rates = fairness_simulator.simulate_performance_data(overall_detection_rate=overall_rate, base_unevenness=unevenness)

    fairness_simulator.display_rates(simulated_rates)

    st.subheader("Fairness Metrics")
    fairness_simulator.display_metrics(simulated_rates)

    st.markdown("""
    **Your Approach:**

    To answer the interview question, discuss:
    *   **Defining Fairness:** How "fairness" is defined in this context (e.g., ensuring similar detection rates, false positive rates, or false negative rates across different skin tone groups).
    *   **Metrics:** Explain specific fairness metrics beyond overall accuracy.
        *   **Demographic Parity (or Statistical Parity):** Measures if the positive prediction rate is the same across all groups. [16, 17, 28] (Simulated above).
        *   **Equalized Odds:** Compares True Positive Rates (sensitivity) and False Positive Rates across groups. [16, 28]
        *   **Equal Opportunity:** Compares True Positive Rates (sensitivity) across groups (a subset of Equalized Odds). [16, 28]
    *   **Bias Mitigation Strategies:**
        *   **Data-level:** Ensuring representative data collection for all skin tones, data augmentation, or synthetic data generation to balance the dataset. [27]
        *   **Model-level:** Incorporating fairness constraints during training, using re-weighting techniques.
        *   **Post-processing:** Adjusting model outputs to improve fairness after inference.
    *   **Evaluation Protocol:** How to evaluate fairness metrics rigorously using a held-out test set that is representative of the target population's diversity.
    *   **NIH AIM-AHEAD:** Mentioning awareness of initiatives like NIH AIM-AHEAD that focus on health equity in AI/ML is relevant. [15, 16]

    *Example thinking:* For this project, equitable performance means the model should detect bruises with similar reliability regardless of the patient's skin tone. I would report metrics like the Equal Opportunity Difference, which specifically looks at the difference in True Positive Rate (sensitivity) between different skin tone groups. This is crucial because we want to ensure that bruises are detected with the same likelihood on dark skin as on light skin. To achieve this, we would need to implement bias mitigation strategies during training, such as ensuring our training dataset is well-balanced across skin tones or using techniques like re-weighting the loss function during training.
    """)

# --- Data Engineering Section ---
elif section == 'Data Engineering':
    st.header("Data Engineering: Managing Secure Healthcare Data")
    st.markdown("""
    The project involves building a robust data repository that combines bruise images with clinical, demographic, and AI-inferred data. [4] The job description mentions experience with database systems, APIs, server/cloud environments, computer security, and patient privacy. [30]

    **Interview Question:** "We capture HL7-FHIR bundles — outline your database schema."

    Below is a description of a simplified FHIR-based schema relevant to the project and considerations for security and privacy.
    """)

    db_schema = DatabaseSchema()
    st.markdown(db_schema.get_schema_description())

    st.markdown("""
    **Your Approach:**

    To elaborate on the database schema and data engineering aspects, consider discussing:
    *   **FHIR Resources:** Deepen the explanation of key FHIR resources like `Patient`, `Encounter`, `Condition`, `ImagingStudy`, `Observation`, and `Media`, and how they relate to storing bruise data. [10, 18, 25, 35, 36]
    *   **Data Storage:** Where the large volume of image data will be stored (e.g., secure cloud storage like AWS S3, Azure Blob Storage, or an on-premise PACS/VNA).
    *   **API Design:** How APIs will be designed to allow the mobile application to securely upload images and data, and for researchers/clinicians to access de-identified data for analysis. Mentioning RESTful APIs is relevant. [36]
    *   **ETL/ELT Pipelines:** How data from different sources (mobile app, potentially EHRs via FHIR) will be ingested, transformed, and loaded into the repository.
    *   **Data Governance:** Policies and procedures for data access, usage, and retention.
    *   **Security Measures:** Detail implementation of security measures mentioned in the schema (encryption, access control, audit trails, de-identification). Highlight HIPAA and 21 CFR Part 11 compliance.
    *   **Scalability and Performance:** How the schema and infrastructure can handle a growing dataset and concurrent access.

    *Example thinking:* Given that you're working with HL7-FHIR bundles, the database design should ideally align with the FHIR data model to facilitate interoperability. The core resources would be `Patient`, `Encounter`, `Condition` (for the bruise details), `ImagingStudy` (representing the photo session), and `Media` (for the images). `Observation` resources could store AI-inferred data like estimated bruise age or confidence scores. The images themselves, being large binary data, would likely be stored separately in a secure, scalable object storage service (like S3) with references in the `Media` resource. Security is paramount; we would implement strict access controls based on user roles, encrypt data at rest and in transit, and maintain detailed audit logs. Pseudonymization of patient identifiers would be essential for data used in research and AI training.
    """)

# --- Mobile Deployment Section ---
elif section == 'Mobile Deployment':
    st.header("Mobile Deployment: On-Device vs. Cloud Inference")
    st.markdown("""
    The EAS-ID platform includes a mobile component. A key architectural decision is where the deep learning inference happens: directly on the mobile device (on-device/edge) or on a remote server in the cloud. [31, 34]

    **Interview Question:** "Inference on-device or in the cloud—convince us."

    Explore the trade-offs below.
    """)

    deployment_strategizer = DeploymentStrategizer()

    deployment_strategy = st.radio(
        "Select a Deployment Strategy to view Trade-offs:",
        options=["On-Device Inference", "Cloud Inference"]
    )

    deployment_strategizer.display_tradeoffs(deployment_strategy)

    st.markdown("""
    **Your Approach:**

    To answer the interview question, argue for the strategy you believe is most suitable for the EAS-ID project, considering its specific needs:
    *   **Clinical Workflow:** Real-time results in a clinical setting (ED, forensic exam) likely favor lower latency.
    *   **Data Sensitivity:** Bruise images and patient data are highly sensitive; keeping data on the device maximizes privacy. [30]
    *   **Accessibility:** Clinicians may be in areas with unreliable network connectivity. [31, 33]
    *   **Model Complexity:** The required deep learning models might be computationally intensive.
    *   **Development and Maintenance:** How easy is it to update models and the application?

    Consider a hybrid approach:
    *   Perform basic, latency-critical inferences (e.g., initial bruise detection/localization) on-device.
    *   Send images/metadata to the cloud for more complex, computationally intensive tasks (e.g., precise age estimation using multiple images over time, detailed fairness analysis, long-term storage). [34]

    *Example thinking (arguing for On-Device, potentially hybrid):* For the EAS-ID platform, I would strongly advocate for prioritizing on-device inference for several critical reasons. Firstly, patient privacy is paramount. Keeping sensitive bruise images and associated data on the device during the initial analysis significantly reduces HIPAA concerns compared to automatically uploading everything to the cloud. [30] Secondly, forensic exams can happen in various locations where network connectivity might be unreliable, making offline capability a major advantage. [31] Finally, for immediate clinical assessment, low latency is essential – waiting for data to transfer to the cloud and back could disrupt workflow. While initial model training requires cloud resources, inference optimized for mobile hardware allows for rapid feedback. A hybrid approach, where initial detection happens on-device and data is then securely uploaded for more detailed analysis or long-term storage, might offer the best balance of privacy, accessibility, and computational power.
    """)

# --- Leadership & Funding Section ---
elif section == 'Leadership & Funding':
    st.header("Leadership, Mentorship, Funding, and Impact")
    st.markdown("""
    Beyond technical skills, the postdoc role involves supervision, collaboration, and contributing to the project's strategic direction and sustainability.

    *   Supervising a team of graduate students and programmers.
    *   Writing manuscripts and reports.
    *   Participating in education and mentoring activities.
    *   Contributing to securing future funding (e.g., NIH R01 renewals).

    Let's consider aspects of leadership and then strategic funding.
    """)

    st.subheader("Leadership & Mentorship")
    leadership_sim = LeadershipSimulator()
    st.markdown(leadership_sim.get_leadership_prompt())

    st.text_area("Reflect on your leadership approach (or type out a brief response to the scenario):", height=200)

    st.markdown("""
    **Your Approach:**

    When discussing leadership and mentorship, draw on your past experiences. Highlight:
    *   Experience supervising or mentoring junior researchers or students.
    *   How you manage projects and timelines in a team setting.
    *   Your communication style, especially with individuals from diverse disciplinary backgrounds (e.g., explaining technical concepts to clinicians, understanding clinical needs as an engineer).
    *   How you foster a collaborative and supportive team environment.
    *   Your commitment to mentoring and contributing to the educational mission (e.g., in the Nursing PhD program).

    """)

    st.subheader("Funding & Impact")
    funding_strat = FundingStrategizer()
    st.markdown(funding_strat.get_funding_text())

    st.markdown("""
    **Your Approach:**

    Discussing funding and impact shows your strategic thinking. Connect your specific work to the broader goals and future of the project. Highlight:
    *   Your understanding of the project's significant societal impact on victims of violence, particularly addressing health disparities. [2, 3, 5, 20]
    *   How your research contributions lay the groundwork for future grant applications (e.g., generating key data, developing novel methods, building prototypes).
    *   Envisioning how the platform can evolve and where future research is needed (e.g., different injury types, integration with other health systems, long-term monitoring).
    *   Your interest in contributing to writing grant proposals and disseminating findings through publications and presentations.
    """)

# --- End of App ---
st.sidebar.markdown("---")
st.sidebar.info("This simulator is based on the provided job description, project information, and general knowledge of AI/ML in healthcare. It is intended for interview preparation purposes.")
