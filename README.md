# Bruise Detection Dashboard

This repository contains a Streamlit dashboard for the GMU Bruise Detection Postdoc interview preparation. The dashboard showcases technical approaches for the Equitable and Accessible Software for Injury Detection (EAS-ID) platform.

## Project Structure

```
src/claude1/
├── app.py                   # Main Streamlit application
├── core/                    # Core implementation modules
│   ├── data_module.py       # Database schema and FHIR data models
│   ├── deployment_module.py # Mobile deployment comparison
│   ├── fairness_module.py   # Fairness metrics and evaluation
│   ├── leadership_module.py # Team management and coordination
│   └── vision_module.py     # Computer vision and image processing
├── visualization_tabs/      # UI components for each dashboard section
│   ├── computer_vision_tab.py
│   ├── data_engineering_tab.py
│   ├── fairness_page.py
│   ├── funding_tab.py
│   ├── home_page_tab.py
│   ├── leadership_tab.py
│   └── mobile_deployment_tab.py
```

## Core Modules

The core modules contain the actual implementation code for different aspects of the bruise detection system:

- **vision_module.py**: Implements computer vision methods for bruise detection, including image preprocessing, ALS filtering, and deep learning-based segmentation.
- **data_module.py**: Defines database schema for FHIR-compliant data storage and handling.
- **deployment_module.py**: Provides analysis of different deployment strategies (on-device, edge, cloud, hybrid).
- **fairness_module.py**: Implements fairness metrics for evaluating model performance across skin tones.
- **leadership_module.py**: Contains team management structures and methodologies.

## Visualization Tabs

The visualization tabs create the UI components for each dashboard section:

- **home_page_tab.py**: Overview of the project
- **computer_vision_tab.py**: Bruise detection approaches
- **fairness_page.py**: Equity metrics and evaluation
- **data_engineering_tab.py**: Database schema and FHIR integration
- **mobile_deployment_tab.py**: Deployment strategies
- **leadership_tab.py**: Team management framework
- **funding_tab.py**: Impact and sustainability

## Running the Dashboard

To run the dashboard:

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install streamlit numpy pandas matplotlib plotly pillow pyvis
   ```
3. Navigate to the project directory
4. Run the Streamlit app:
   ```bash
   streamlit run src/claude1/app.py
   ```

## Key Features

- **Interactive Demos**: Visualize bruise detection with different image processing techniques
- **Architectural Diagrams**: Explore system architecture and database design
- **Decision Frameworks**: Evaluate deployment options and fairness strategies
- **Team Management**: Understand interdisciplinary team coordination approaches

## Project Context

The EAS-ID platform aims to develop a mobile AI tool that makes bruises visible across all skin tones using:

- Deep neural networks and computer vision
- Alternate Light Source (ALS) imaging technology
- Multi-spectral analysis techniques
- Fairness-aware model development
- Secure, HIPAA-compliant cloud architecture
