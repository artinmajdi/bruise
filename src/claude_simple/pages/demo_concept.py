import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from components.bruise_detector import BruiseDetector

def show():
    """
    Display the demo concept page with interactive bruise detection visualization.
    """
    st.title("Demo Concept: Bruise Detection Prototype")

    st.markdown("""
    ## Prototype Concept

    This interactive demo simulates how the bruise detection system might work across different skin tones.
    In a real interview, you could prepare a similar but more sophisticated demo using:

    1. A pre-trained model fine-tuned on public bruise datasets
    2. Multi-spectral input simulation
    3. Fairness metrics across skin tones

    The demo below illustrates the basic concept of how ALS illumination improves bruise visibility.
    """)

    st.markdown("---")

    # Initialize bruise detector
    detector = BruiseDetector()

    # Demo interface
    st.subheader("Bruise Detection Simulation")

    col1, col2 = st.columns(2)

    with col1:
        # Create a synthetic image with different skin tones
        skin_tone = st.select_slider(
            "Select skin tone (Fitzpatrick scale)",
            options=["Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"]
        )

        # Map skin tone to RGB color
        skin_tone_map = {
            "Type I": [255, 236, 210],    # Very light
            "Type II": [241, 194, 125],   # Light
            "Type III": [224, 172, 105],  # Medium
            "Type IV": [198, 134, 66],    # Medium-dark
            "Type V": [141, 85, 36],      # Dark
            "Type VI": [67, 38, 24]       # Very dark
        }

        # Generate skin image with selected tone
        skin_image = detector.generate_synthetic_skin(
            np.array(skin_tone_map[skin_tone], dtype=np.uint8)
        )

        st.write("### White Light Image")
        st.image(skin_image, use_container_width=True)

        st.write("Bruise visibility decreases with darker skin tones under white light.")

    with col2:
        st.write("### Alternate Light Source (ALS) Simulation")

        als_intensity = st.slider("ALS intensity", 0.0, 1.0, 0.7)

        # Apply ALS simulation
        als_image = detector.apply_als_simulation(skin_image, als_intensity)

        st.image(als_image, use_container_width=True)

        st.write("ALS (400-520nm) with orange filter enhances bruise visibility regardless of skin tone.")

    st.markdown("---")

    st.subheader("AI Model Integration Concept")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Model Architecture

        **Proposed Approach:**
        - U-Net architecture with EfficientNet-V2 backbone
        - Multi-channel input (white light + ALS)
        - Transfer learning from dermatology datasets
        - Attention mechanisms for better feature localization
        - Model explainability via Grad-CAM for clinical interpretation
        """)

        st.markdown("""
        ### Training Strategy

        - Pre-train on large dermatology datasets
        - Fine-tune on bruise datasets
        - Generate synthetic bruises using GANs across skin tones
        - Balanced sampling to ensure fairness
        - Augmentation for lighting conditions and camera variations
        - Cross-validation stratified by skin tone
        """)

    with col2:
        st.markdown("""
        ### Fairness Metrics

        **Key Metrics:**
        - True Positive Rate Parity across skin tones
        - False Positive Rate Parity across skin tones
        - AUC per Fitzpatrick skin type
        - F1-score per Fitzpatrick skin type
        - Equalized odds difference
        """)

        # Create a simple fairness visualization
        fig, ax = plt.subplots(figsize=(8, 5))

        skin_types = ["Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"]

        # Hypothetical model performance metrics
        baseline_tpr = [0.95, 0.93, 0.88, 0.82, 0.75, 0.70]
        improved_tpr = [0.94, 0.92, 0.91, 0.90, 0.89, 0.88]

        x = np.arange(len(skin_types))
        width = 0.35

        ax.bar(x - width/2, baseline_tpr, width, label='Standard Model')
        ax.bar(x + width/2, improved_tpr, width, label='Equitable Model')

        ax.set_title('True Positive Rate Across Skin Tones')
        ax.set_xlabel('Fitzpatrick Skin Type')
        ax.set_ylabel('True Positive Rate')
        ax.set_ylim([0.5, 1.0])
        ax.set_xticks(x)
        ax.set_xticklabels(skin_types)
        ax.legend()

        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Mobile Application UI Concept")

    st.markdown("""
    For an actual interview, consider creating a one-slide "nurse-facing UI mock-up"
    illustrating intuitive icons, one-tap export to charting, and auto-generated legal report.

    **Key UI Features:**
    - Simple camera controls for white light and ALS capture
    - Side-by-side image comparison
    - Automated bruise detection overlay
    - Easy documentation tools
    - Integration with electronic health records
    - Secure image storage and sharing
    """)

    # Simple mockup of a mobile UI
    ui_cols = st.columns(3)

    with ui_cols[0]:
        st.markdown("### Camera View")
        st.markdown("""
        ```
        +------------------+
        |   ALS Mode       |
        |                  |
        |  [Camera View]   |
        |                  |
        |                  |
        |                  |
        |                  |
        +------------------+
        | White | ALS | AI |
        +------------------+
        ```
        """)

    with ui_cols[1]:
        st.markdown("### Detection View")
        st.markdown("""
        ```
        +------------------+
        | Detection Results|
        |                  |
        | [Image with      |
        |  highlighted     |
        |  bruise regions] |
        |                  |
        |                  |
        +------------------+
        | Save | Share | ⚙️ |
        +------------------+
        ```
        """)

    with ui_cols[2]:
        st.markdown("### Documentation")
        st.markdown("""
        ```
        +------------------+
        | Documentation    |
        |                  |
        | Location: [____] |
        | Size: __ x __ cm |
        | Notes: [________]|
        |        [________]|
        |                  |
        +------------------+
        | Export | Report  |
        +------------------+
        ```
        """)
