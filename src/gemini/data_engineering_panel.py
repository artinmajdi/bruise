import streamlit as st
import graphviz # Ensure graphviz is installed: pip install graphviz

class DataEngineeringPanel:
	"""
	Handles the display and logic for the Data Engineering section.
	Focuses on outlining a database schema for HL7-FHIR bundles.
	"""

	def _create_fhir_schema_diagram(self):
		"""
		Creates a conceptual FHIR-like schema diagram using Graphviz.
		This is a simplified representation.
		"""
		dot = graphviz.Digraph('FHIR_Schema', comment='Conceptual FHIR-like Schema for Bruise Data')
		dot.attr(rankdir='LR') # Left to Right layout
		dot.attr(bgcolor='transparent') # For better theme integration

		# Node styling
		node_attrs = {
			'shape': 'record', # Use record shape for table-like nodes
			'style': 'filled',
			'fillcolor': '#e6f7ff', # Light blue
			'fontname': 'Arial',
			'fontsize': '10'
		}
		edge_attrs = {
			'fontname': 'Arial',
			'fontsize': '9'
		}

		# Define resources (nodes)
		dot.node('Patient',
				 '{Patient | + id (string) \l + identifier (Identifier) \l + active (boolean) \l + name (HumanName) \l + birthDate (date) \l + gender (code) \l + race (Extension) \l + ethnicity (Extension) \l + skinType (Extension - Fitzpatrick) \l ...}',
				 **node_attrs)

		dot.node('Encounter',
				 '{Encounter | + id (string) \l + status (code) \l + class (Coding) \l + subject (Reference to Patient) \l + period (Period) \l + location (Reference to Location) \l ...}',
				 **node_attrs)

		dot.node('Observation_Bruise',
				 '{Observation (Bruise) | + id (string) \l + status (code) \l + category (CodeableConcept - Injury) \l + code (CodeableConcept - Bruise) \l + subject (Reference to Patient) \l + encounter (Reference to Encounter) \l + effectiveDateTime (dateTime) \l + bodySite (CodeableConcept) \l + valueCodeableConcept (e.g., color, shape) \l + component: dimensions (length, width) \l + component: age_estimation (string) \l + derivedFrom (Reference to Media/ImagingStudy) \l ...}',
				 **node_attrs)

		dot.node('Media_Image',
				 '{Media (Image) | + id (string) \l + status (code) \l + type (CodeableConcept - Photo) \l + subject (Reference to Patient) \l + encounter (Reference to Encounter) \l + createdDateTime (dateTime) \l + content (Attachment - image data/URL) \l + deviceName (string - e.g., phone model) \l + height (positiveInt) \l + width (positiveInt) \l + frames (positiveInt - for multi-spectral) \l + extension: lighting_source (e.g., White Light, ALS 415nm) \l ...}',
				 **node_attrs)

		dot.node('ImagingStudy',
				 '{ImagingStudy | + id (string) \l + status (code) \l + subject (Reference to Patient) \l + encounter (Reference to Encounter) \l + started (dateTime) \l + numberOfSeries (positiveInt) \l + numberOfInstances (positiveInt) \l + procedureCode (CodeableConcept - Bruise Documentation) \l + series: [ { uid, modality (e.g., PHO), bodySite, instance: [ {uid, sopClass, number, media (Reference to Media)} ] } ] \l ...}',
				 **node_attrs)

		dot.node('Device',
				 '{Device | + id (string) \l +deviceName (DeviceName) \l + modelNumber (string) \l + manufacturer (string) \l + extension: als_wavelengths_supported (string list) \l ...}',
				 **node_attrs)

		# Define relationships (edges)
		dot.edge('Encounter', 'Patient', label='subject (1..1)', **edge_attrs)
		dot.edge('Observation_Bruise', 'Patient', label='subject (1..1)', **edge_attrs)
		dot.edge('Observation_Bruise', 'Encounter', label='encounter (0..1)', **edge_attrs)
		dot.edge('Media_Image', 'Patient', label='subject (1..1)', **edge_attrs)
		dot.edge('Media_Image', 'Encounter', label='encounter (0..1)', **edge_attrs)
		dot.edge('ImagingStudy', 'Patient', label='subject (1..1)', **edge_attrs)
		dot.edge('ImagingStudy', 'Encounter', label='encounter (0..1)', **edge_attrs)
		dot.edge('ImagingStudy', 'Media_Image', label='references series.instance.media (0..*)', arrowhead='normal', style='dashed', **edge_attrs)
		dot.edge('Observation_Bruise', 'Media_Image', label='derivedFrom (0..*)', arrowhead='normal', style='dashed', **edge_attrs)
		dot.edge('Media_Image', 'Device', label='capturedBy (0..1)', arrowhead='normal', style='dashed', **edge_attrs)


		return dot


	def display(self):
		"""Displays the data engineering content."""
		st.header("🛠️ Data Engineering: HL7-FHIR Database Schema")
		st.markdown("""
		**Challenge:** "We capture HL7-FHIR bundles—outline your database schema." (Note: FHIR itself defines resources, not a relational schema. The question likely implies how you'd store/manage FHIR data or design a system that interacts with FHIR.)

		**What they're looking for:** Ability to design scalable, secure clinical data stores, understanding FHIR concepts.
		""")
		st.markdown("---")

		st.subheader("Understanding HL7-FHIR")
		st.markdown("""
		HL7 Fast Healthcare Interoperability Resources (FHIR) is a standard for exchanging healthcare information electronically. It defines:
		- **Resources:** Modular components representing clinical and administrative information (e.g., Patient, Observation, ImagingStudy, Media).
		- **Interactions:** Standard ways to interact with these resources (e.g., RESTful APIs).
		- **Profiles:** Constraints and extensions on resources for specific use cases.

		When "designing a database schema for FHIR bundles," one might be:
		1.  Designing a relational or NoSQL database to **store parsed FHIR resources** persistently.
		2.  Designing a system that **natively uses a FHIR server** (which manages its own persistence).
		3.  Designing a data model for an application that will **transform data into FHIR format** for interoperability.

		For the EAS-ID project, we'd likely need to store image metadata, analysis results, and patient demographics, making them interoperable via FHIR.
		""")

		st.subheader("Conceptual FHIR-based Data Model for Bruise Analysis")
		st.markdown("""
		Below is a conceptual diagram illustrating key FHIR resources and their relationships relevant to storing data for the EAS-ID project. This isn't a strict relational schema but shows how FHIR resources might be linked.
		A FHIR server (like HAPI FHIR, Medplum, or Azure API for FHIR) would manage the underlying persistence. If building a custom store, one might map these to tables/collections.
		""")

		try:
			schema_diagram = self._create_fhir_schema_diagram()
			st.graphviz_chart(schema_diagram)
		except Exception as e:
			st.error(f"Could not render diagram. Ensure Graphviz is installed and in your system PATH. Error: {e}")
			st.markdown("""
			**Fallback Textual Representation of Key Resources & Relationships:**

			* **Patient:** Demographics, skin type (Fitzpatrick via extension).
				* `Links to:` Encounter, Observation, ImagingStudy, Media.
			* **Encounter:** Context of the examination (e.g., clinic visit, forensic examination).
				* `Links to:` Patient.
			* **Observation (Bruise Characteristics):** Detailed findings about a bruise.
				* `code`: Specifies it's a bruise observation.
				* `bodySite`: Location on the body.
				* `value[x]`: Color, shape, size (e.g., `valueCodeableConcept`, `valueQuantity`).
				* `component`: For structured data like dimensions, AI-estimated age.
				* `derivedFrom`: Reference to Media/ImagingStudy containing the visual evidence.
				* `Links to:` Patient, Encounter.
			* **Media (Image):** Stores the actual image data (or link to it) and metadata.
				* `content`: The image itself (e.g., Attachment with URL or base64).
				* `deviceName`: Capturing device (e.g., smartphone model).
				* `extension: lighting_source`: Crucial for EAS-ID (e.g., "White Light", "ALS 415nm").
				* `Links to:` Patient, Encounter, Device.
			* **ImagingStudy:** Groups related images (series) from a study.
				* `series.instance.media`: Links to individual Media resources (images).
				* `procedureCode`: Type of imaging procedure (e.g., "Bruise Documentation Protocol").
				* `Links to:` Patient, Encounter.
			* **Device:** Information about the capture device (e.g., mobile phone, ALS system).
				* `extension: als_wavelengths_supported`: If the device has specific ALS capabilities.
			""")

		st.subheader("Key Design Considerations for Storing/Managing FHIR Data")
		st.markdown("""
		1.  **Scalability:**
			* **Database Choice:**
				* **FHIR Servers:** Many are built on robust databases (e.g., PostgreSQL, Elasticsearch) and handle scaling.
				* **Custom Store:** If storing parsed FHIR JSON, NoSQL databases (e.g., MongoDB, Couchbase) can offer schema flexibility and horizontal scaling. Relational databases can also work with JSONB types or well-defined schemas.
			* **Image Storage:** Images (Media resources) are often large. Store metadata in the primary database and image binaries in a scalable object store (e.g., AWS S3, Azure Blob Storage, MinIO), linking via URLs in the FHIR Media resource.
			* **Indexing:** Proper indexing on frequently queried fields (Patient ID, dates, codes, extensions like skin type) is crucial for performance.

		2.  **Security & Privacy (HIPAA Compliance):**
			* **Encryption:** Data at rest and in transit must be encrypted.
			* **Access Controls:** Role-based access control (RBAC) to ensure only authorized users can access/modify data. FHIR defines security tags and consent resources.
			* **Audit Trails:** Log all access and modifications to PHI. FHIR's `Provenance` resource can track data history.
			* **De-identification/Pseudonymization:** For research purposes, robust de-identification of data is necessary, while retaining links for longitudinal study if permitted. FHIR provides guidance on this.
			* **Secure APIs:** Use HTTPS, OAuth 2.0/OpenID Connect for API authentication and authorization.

		3.  **Interoperability:**
			* **FHIR Conformance:** Adhere to FHIR standards and relevant Implementation Guides (IGs) (e.g., US Core IG).
			* **Profiles & Extensions:** Define custom profiles for EAS-ID specific data (e.g., detailed bruise characteristics, ALS parameters) as extensions on standard FHIR resources. This ensures the data remains FHIR-compliant while capturing necessary detail.
			* **Terminology Services:** Use standard terminologies (SNOMED CT, LOINC, ICD-10) for coding data elements (e.g., bruise type, body site) to ensure semantic interoperability. FHIR `CodeableConcept` supports this.

		4.  **Data Integrity & Versioning:**
			* FHIR resources are versioned. Each update creates a new version, allowing for history tracking.
			* Validate incoming FHIR resources against profiles to ensure data quality.

		5.  **Data Lifecycle Management:**
			* Policies for data retention, archiving, and disposal.
		""")

		st.markdown("<div class='info-box'><h3>Key Takeaway:</h3>Leveraging FHIR provides a standardized way to represent and exchange clinical data. For EAS-ID, this means using resources like `Patient`, `Observation`, `Media`, and `ImagingStudy`, potentially with custom extensions for bruise-specific and ALS-specific details. The 'database schema' involves how these resources are stored and managed, whether in a native FHIR server or a custom backend, always prioritizing scalability, security, and interoperability.</div>", unsafe_allow_html=True)
