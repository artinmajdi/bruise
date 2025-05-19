# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local application imports
from core.fairness_module import FairnessMetrics, generate_fairness_report


class FairnessPage:
	def __init__(self):
		pass

	def render(self):
		st.markdown('<div class="main-header">Fairness in Bruise Detection</div>', unsafe_allow_html=True)

		st.markdown("""
		This section addresses the interview question:

		> **"Describe a metric you would report to show equitable performance."**

		The interviewers are looking for awareness of bias and reporting metrics like Demographic Parity Difference.
		""")

		# Create tabs for different aspects of fairness
		f_tab1, f_tab2, f_tab3 = st.tabs(["Fairness Metrics", "Bias Mitigation", "Fairness Dashboard"])

		with f_tab1:
			st.markdown('<div class="sub-header">Fairness Metrics for Skin Tone Equity</div>', unsafe_allow_html=True)

			st.markdown("""
			For the bruise detection system, I would implement a comprehensive fairness evaluation framework that specifically addresses the challenges of detecting bruises across different skin tones (Fitzpatrick scale I-VI).

			The fundamental metric categories are:
			""")

			# Create columns for different metric categories
			col1, col2 = st.columns(2)

			with col1:
				st.markdown("""
				### Group Fairness Metrics

				1. **Demographic Parity Difference**
				   - Measures whether positive prediction rates are similar across skin tones
				   - Formula: |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
				   - Target: < 0.05 difference between any skin tone groups

				2. **Equalized Odds**
				   - Ensures similar true positive and false positive rates across groups
				   - Formula: |TPR_a - TPR_b| and |FPR_a - FPR_b|
				   - Critical for forensic applications

				3. **Predictive Parity**
				   - Measures if positive predictive value is similar across skin tones
				   - Formula: |PPV_a - PPV_b|
				   - Important for medical decision confidence
				""")

			with col2:
				st.markdown("""
				### Detection-Specific Metrics

				1. **Bruise Detection Rate (BDR)**
				   - The ratio of correctly detected bruises to total bruises
				   - Stratified by skin tone and bruise age
				   - Formula: TP / (TP + FN) for each skin tone group

				2. **Minimum Detectable Contrast (MDC)**
				   - Smallest bruise-to-skin contrast ratio detectable
				   - Lower MDC indicates better detection in challenging cases
				   - Should be reported separately for each skin tone

				3. **Localization Equity**
				   - Measures consistency of bruise boundary accuracy across skin tones
				   - Based on Dice coefficient or IoU ratios between groups
				   - Formula: |IoU_a - IoU_b| < threshold
				""")

			st.markdown('<div class="section-header">Sample Fairness Report</div>', unsafe_allow_html=True)

			# Generate sample fairness metrics data
			skin_tones = ["Type I-II", "Type III-IV", "Type V-VI"]

			# Sample metrics
			metrics_data = {
				"Skin Tone": skin_tones,
				"Bruise Detection Rate": [0.91, 0.89, 0.85],
				"False Positive Rate": [0.03, 0.04, 0.06],
				"IoU Score": [0.85, 0.82, 0.79]
			}

			metrics_df = pd.DataFrame(metrics_data)

			# Calculate fairness gap
			max_bdr = max(metrics_data["Bruise Detection Rate"])
			min_bdr = min(metrics_data["Bruise Detection Rate"])
			fairness_gap = max_bdr - min_bdr

			# Plot metrics
			fig = px.bar(
				metrics_df,
				x="Skin Tone",
				y=["Bruise Detection Rate", "False Positive Rate", "IoU Score"],
				barmode="group",
				title="Detection Performance by Skin Tone Group",
				color_discrete_sequence=["#4CAF50", "#F44336", "#2196F3"]
			)

			fig.update_layout(
				height=400,
				margin=dict(l=20, r=20, t=40, b=20),
				legend_title="Metric"
			)

			st.plotly_chart(fig, use_container_width=True)

			# Show fairness gap callout
			st.markdown(f"""
			<div class="highlight-text">
			<b>Fairness Gap Analysis:</b><br>
			The current model shows a Bruise Detection Rate (BDR) fairness gap of <b>{fairness_gap:.2f}</b> between the highest and lowest performing skin tone groups.

			Our target threshold is <0.05 to ensure equitable performance.

			This gap indicates that further work is needed on:
			1. More training data for Type V-VI skin tones
			2. Enhanced preprocessing for darker skin
			3. Potential model architecture modifications
			</div>
			""", unsafe_allow_html=True)

			# Equity threshold explanation
			st.markdown("""
			### Recommended Equity Thresholds

			Based on literature review and clinical significance, I propose these fairness thresholds:

			1. **Demographic Parity Gap**: < 0.05 difference between any skin tone groups
			2. **Bruise Detection Rate Gap**: < 0.05 difference between skin tone groups
			3. **False Positive Rate Gap**: < 0.03 difference between skin tone groups
			4. **Precision Gap**: < 0.05 difference between skin tone groups
			5. **IoU/Dice Score Gap**: < 0.07 difference between skin tone groups

			These thresholds are based on NIH AIM-AHEAD guidelines and clinical significance for forensic evidence standards.
			""")

		with f_tab2:
			st.markdown('<div class="sub-header">Bias Mitigation Strategies</div>', unsafe_allow_html=True)

			st.markdown("""
			To achieve equitable performance across skin tones, I would implement a multi-faceted bias mitigation strategy:
			""")

			# Create columns for pre-processing, in-processing, and post-processing
			col1, col2, col3 = st.columns(3)

			with col1:
				st.markdown("""
				### Pre-processing Techniques

				1. **Balanced Dataset Creation**
				   - Equal representation across skin tones
				   - Stratified sampling by Fitzpatrick scale
				   - Synthetic data generation for underrepresented groups

				2. **Data Augmentation**
				   - Skin tone transformations
				   - Contrast-preserving augmentations
				   - Bruise appearance variations

				3. **Fitzpatrick-Aware Preprocessing**
				   - Adaptive contrast enhancement
				   - Skin tone specific channel manipulation
				   - Multi-spectral normalization
				""")

			with col2:
				st.markdown("""
				### In-processing Techniques

				1. **Fairness-aware Loss Functions**
				   - Group-DRO (Distributionally Robust Optimization)
				   - Adversarial debiasing
				   - Fairness regularization terms

				2. **Model Architecture**
				   - Skin tone detection branch
				   - Adaptive feature normalization
				   - Attention mechanisms for different skin tones

				3. **Training Strategy**
				   - Curriculum learning by difficulty
				   - Gradient accumulation with fairness constraints
				   - Multi-task learning with auxiliary fairness objectives
				""")

			with col3:
				st.markdown("""
				### Post-processing Techniques

				1. **Threshold Optimization**
				   - Skin tone specific decision thresholds
				   - ROC analysis for optimal operating points
				   - Confidence calibration by group

				2. **Ensemble Methods**
				   - Specialized models for different skin tones
				   - Weighted ensemble based on skin tone
				   - Stacking with fairness-aware meta-learner

				3. **Human-in-the-Loop**
				   - Active learning for edge cases
				   - Feedback incorporation process
				   - Continuous fairness monitoring
				""")

			# Implementation roadmap
			st.markdown('<div class="section-header">Fairness Implementation Roadmap</div>', unsafe_allow_html=True)

			# Create a timeline visualization
			timeline_data = {
				"Stage": [
					"Baseline Assessment",
					"Dataset Enhancement",
					"Initial Debiasing",
					"Model Architecture",
					"Threshold Optimization",
					"Clinical Validation",
					"Continuous Monitoring"
				],
				"Start": [0, 1, 2, 3, 4, 5, 6],
				"Duration": [1, 2, 2, 3, 1, 2, 1],
				"Description": [
					"Evaluate baseline model performance across skin tones",
					"Balanced dataset creation with synthetic data generation",
					"Implement pre-processing debiasing techniques",
					"Develop and test fairness-aware model architectures",
					"Optimize detection thresholds for each skin tone group",
					"Clinical validation with diverse patient populations",
					"Deploy continuous fairness monitoring system"
				]
			}

			# Add end time calculation to the timeline data
			timeline_df = pd.DataFrame(timeline_data)
			timeline_df['End'] = timeline_df['Start'] + timeline_df['Duration']

			# Create Gantt chart
			fig = px.timeline(
				timeline_df,
				x_start="Start",
				x_end="End",
				y="Stage",
				text="Description",
				title="Fairness Implementation Timeline (Months)",
				color_discrete_sequence=["#4CAF50"]
			)

			fig.update_layout(
				height=400,
				margin=dict(l=20, r=20, t=40, b=20)
			)

			# Hide axis labels
			fig.update_yaxes(title="")
			fig.update_xaxes(title="Months")

			st.plotly_chart(fig, use_container_width=True)

			# Case study
			st.markdown('<div class="section-header">Case Study: Successful Bias Mitigation</div>', unsafe_allow_html=True)

			st.markdown("""
			<div class="highlight-text">
			<b>Successful Approach in Similar Domain: Skin Lesion Classification</b>

			The ISIC 2019 challenge demonstrated effective bias mitigation in skin lesion detection across skin tones:

			1. <b>Dataset Rebalancing</b>: Created synthetic dark skin examples using CycleGAN to balance training data

			2. <b>Multi-Task Learning</b>: Added skin tone classification as auxiliary task, sharing early features

			3. <b>Adaptive Preprocessing</b>: Developed skin-tone specific preprocessing pipeline

			4. <b>Results</b>: Reduced performance gap from 0.15 to 0.03 in detection rate across skin tone groups

			This approach can be adapted to our bruise detection task with ALS imaging as an additional input channel.
			</div>
			""", unsafe_allow_html=True)

		with f_tab3:
			st.markdown('<div class="sub-header">Fairness Monitoring Dashboard</div>', unsafe_allow_html=True)

			st.markdown("""
			For the EAS-ID platform, I would develop a comprehensive fairness monitoring dashboard that tracks performance across demographic groups over time. This would enable:

			1. Real-time monitoring of fairness metrics
			2. Early detection of performance drift
			3. Transparent reporting for stakeholders
			4. Identification of areas for model improvement
			""")

			# Create a mock dashboard
			st.markdown('<div class="section-header">Fairness Dashboard Preview</div>', unsafe_allow_html=True)

			# Create tabs for different dashboard sections
			dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Performance Overview", "Detailed Metrics", "Failure Analysis"])

			with dash_tab1:
				# Generate some sample data
				np.random.seed(42)

				# Dates for time series
				dates = pd.date_range(start='2025-01-01', periods=12, freq='W')

				# Performance data
				performance_data = {
					"Date": list(dates) * 3,
					"Skin Tone": ["Type I-II"] * 12 + ["Type III-IV"] * 12 + ["Type V-VI"] * 12,
					"Detection Rate": np.clip(
						np.concatenate([
							0.92 + np.random.normal(0, 0.01, 12),  # Type I-II
							0.90 + np.random.normal(0, 0.01, 12),  # Type III-IV
							0.88 + np.random.normal(0, 0.015, 12)  # Type V-VI
						]),
						0, 1
					),
					"False Positive Rate": np.clip(
						np.concatenate([
							0.03 + np.random.normal(0, 0.005, 12),  # Type I-II
							0.04 + np.random.normal(0, 0.005, 12),  # Type III-IV
							0.05 + np.random.normal(0, 0.01, 12)    # Type V-VI
						]),
						0, 1
					),
					"Precision": np.clip(
						np.concatenate([
							0.94 + np.random.normal(0, 0.01, 12),  # Type I-II
							0.92 + np.random.normal(0, 0.01, 12),  # Type III-IV
							0.89 + np.random.normal(0, 0.015, 12)  # Type V-VI
						]),
						0, 1
					)
				}

				performance_df = pd.DataFrame(performance_data)

				# Detection rate over time
				fig1 = px.line(
					performance_df,
					x="Date",
					y="Detection Rate",
					color="Skin Tone",
					title="Bruise Detection Rate by Skin Tone (Over Time)",
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig1.update_layout(
					height=300,
					margin=dict(l=20, r=20, t=40, b=20),
					yaxis=dict(range=[0.85, 0.95])
				)

				st.plotly_chart(fig1, use_container_width=True)

				# Fairness gap over time
				fairness_gap_data = []

				for date in dates:
					date_subset = performance_df[performance_df["Date"] == date]
					max_rate = date_subset["Detection Rate"].max()
					min_rate = date_subset["Detection Rate"].min()
					gap = max_rate - min_rate
					fairness_gap_data.append({
						"Date": date,
						"Fairness Gap": gap
					})

				fairness_gap_df = pd.DataFrame(fairness_gap_data)

				fig2 = px.line(
					fairness_gap_df,
					x="Date",
					y="Fairness Gap",
					title="Fairness Gap Over Time (Detection Rate)",
					color_discrete_sequence=["#FF9800"]
				)

				# Add threshold line
				fig2.add_hline(
					y=0.05,
					line_dash="dash",
					line_color="red",
					annotation_text="Threshold (0.05)",
					annotation_position="bottom right"
				)

				fig2.update_layout(
					height=250,
					margin=dict(l=20, r=20, t=40, b=20),
					yaxis=dict(range=[0, 0.1])
				)

				st.plotly_chart(fig2, use_container_width=True)

				# Summary metrics
				st.markdown("""
				<div class="highlight-text">
				<b>Fairness Summary Metrics:</b>

				- Current Detection Rate Gap: 0.04 (Below threshold of 0.05) ✓
				- Current False Positive Rate Gap: 0.02 (Below threshold of 0.03) ✓
				- Current Precision Gap: 0.05 (At threshold of 0.05) ⚠️

				<b>Trend Analysis:</b> Fairness gap has decreased by 38% since initial deployment, showing continued improvement in equitable performance.
				</div>
				""", unsafe_allow_html=True)

			with dash_tab2:
				# ROC curves for different skin tones
				st.markdown("### ROC Curves by Skin Tone")

				# Generate ROC curve data
				def generate_roc_data(base_tpr, noise):
					fpr = np.linspace(0, 1, 100)
					tpr = np.clip(base_tpr * fpr + (1 - base_tpr) * fpr**2 + np.random.normal(0, noise, 100), 0, 1)
					return fpr, tpr

				fpr1, tpr1 = generate_roc_data(0.95, 0.01)  # Type I-II
				fpr2, tpr2 = generate_roc_data(0.92, 0.015)  # Type III-IV
				fpr3, tpr3 = generate_roc_data(0.88, 0.02)  # Type V-VI

				fig = go.Figure()

				fig.add_trace(go.Scatter(x=fpr1, y=tpr1, name="Type I-II", line=dict(color="#4CAF50", width=2)))
				fig.add_trace(go.Scatter(x=fpr2, y=tpr2, name="Type III-IV", line=dict(color="#2196F3", width=2)))
				fig.add_trace(go.Scatter(x=fpr3, y=tpr3, name="Type V-VI", line=dict(color="#F44336", width=2)))
				fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="grey", width=2, dash="dash")))

				fig.update_layout(
					title="ROC Curves by Skin Tone",
					xaxis_title="False Positive Rate",
					yaxis_title="True Positive Rate",
					height=400,
					margin=dict(l=20, r=20, t=40, b=20),
					legend=dict(x=0.7, y=0.2)
				)

				st.plotly_chart(fig, use_container_width=True)

				# Confusion matrices for different skin tones
				st.markdown("### Confusion Matrices by Skin Tone")

				# Create columns for confusion matrices
				cm_col1, cm_col2, cm_col3 = st.columns(3)

				# Sample confusion matrix data
				def generate_cm(tp, fp, fn, tn):
					return np.array([
						[tp, fp],
						[fn, tn]
					])

				cm1 = generate_cm(92, 8, 6, 94)  # Type I-II
				cm2 = generate_cm(90, 10, 9, 91)  # Type III-IV
				cm3 = generate_cm(87, 13, 11, 89)  # Type V-VI

				# Plot confusion matrices
				with cm_col1:
					fig = px.imshow(
						cm1,
						text_auto=True,
						title="Type I-II",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Greens"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				with cm_col2:
					fig = px.imshow(
						cm2,
						text_auto=True,
						title="Type III-IV",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Blues"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				with cm_col3:
					fig = px.imshow(
						cm3,
						text_auto=True,
						title="Type V-VI",
						labels=dict(x="Predicted", y="Actual"),
						x=["Bruise", "No Bruise"],
						y=["Bruise", "No Bruise"],
						color_continuous_scale="Reds"
					)
					fig.update_layout(height=250)
					st.plotly_chart(fig, use_container_width=True)

				# Detection performance by bruise age
				st.markdown("### Detection Performance by Bruise Age")

				# Sample data for bruise age vs detection rate
				bruise_age_data = {
					"Bruise Age (days)": [1, 3, 5, 7, 10, 14],
					"Type I-II": [0.95, 0.93, 0.91, 0.88, 0.84, 0.80],
					"Type III-IV": [0.92, 0.90, 0.87, 0.84, 0.80, 0.76],
					"Type V-VI": [0.88, 0.85, 0.82, 0.78, 0.74, 0.70]
				}

				df = pd.DataFrame(bruise_age_data)
				df_melted = pd.melt(
					df,
					id_vars=["Bruise Age (days)"],
					value_vars=["Type I-II", "Type III-IV", "Type V-VI"],
					var_name="Skin Tone",
					value_name="Detection Rate"
				)

				fig = px.line(
					df_melted,
					x="Bruise Age (days)",
					y="Detection Rate",
					color="Skin Tone",
					title="Detection Rate by Bruise Age and Skin Tone",
					markers=True,
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig.update_layout(
					height=350,
					margin=dict(l=20, r=20, t=40, b=20)
				)

				st.plotly_chart(fig, use_container_width=True)

			with dash_tab3:
				st.markdown("### Failure Analysis")

				# Sample failure mode data
				failure_modes = {
					"Failure Mode": [
						"Low Contrast",
						"Bruise Near Hair",
						"Dark Skin + Old Bruise",
						"Multiple Overlapping Bruises",
						"Tattoo Interference",
						"Skin Discoloration",
						"Shadow Effects"
					],
					"Type I-II": [15, 10, 5, 22, 18, 12, 18],
					"Type III-IV": [18, 12, 10, 20, 15, 15, 10],
					"Type V-VI": [35, 8, 22, 15, 12, 5, 3]
				}

				failure_df = pd.DataFrame(failure_modes)

				# Melt the dataframe for plotting
				failure_melted = pd.melt(
					failure_df,
					id_vars=["Failure Mode"],
					value_vars=["Type I-II", "Type III-IV", "Type V-VI"],
					var_name="Skin Tone",
					value_name="Percentage"
				)

				fig = px.bar(
					failure_melted,
					x="Failure Mode",
					y="Percentage",
					color="Skin Tone",
					title="Failure Mode Analysis by Skin Tone",
					barmode="group",
					color_discrete_sequence=["#4CAF50", "#2196F3", "#F44336"]
				)

				fig.update_layout(
					height=400,
					margin=dict(l=20, r=20, t=40, b=20)
				)

				st.plotly_chart(fig, use_container_width=True)

				# Recommendations based on failure analysis
				st.markdown("""
				<div class="highlight-text">
				<b>Failure Analysis Insights:</b>

				1. <b>Key Issue:</b> Low contrast bruises on Type V-VI skin is the dominant failure mode (35%)

				2. <b>Recommendations:</b>
				   - Enhance ALS preprocessing specifically for dark skin
				   - Collect additional data focusing on low-contrast bruises
				   - Implement specialized detection model for this specific case
				   - Consider dual-wavelength ALS imaging (415nm + 450nm)

				3. <b>Secondary Priority:</b> Improve detection of old bruises on dark skin

				4. <b>Note:</b> Tattoo interference affects all skin tones similarly, suggesting this is not a fairness issue but a general detection challenge
				</div>
				""", unsafe_allow_html=True)

				# Sample misclassification examples
				st.markdown("### Misclassification Examples")

				# Create two columns for examples
				miss_col1, miss_col2 = st.columns(2)

				with miss_col1:
					st.markdown("#### Example 1: Low Contrast Bruise (Type V)")

					# Generate a synthetic "failure case"
					img_size = 300
					dark_skin = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([90, 60, 50], dtype=np.uint8)

					# Add some natural skin texture
					texture = np.random.normal(0, 5, (img_size, img_size, 3))
					dark_skin = np.clip(dark_skin + texture, 0, 255).astype(np.uint8)

					# Add a very faint bruise
					center_x, center_y = img_size // 2, img_size // 2
					radius = 60

					bruise_mask = np.zeros((img_size, img_size))
					for i in range(img_size):
						for j in range(img_size):
							dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
							if dist < radius:
								# Gradient falloff for natural appearance
								bruise_mask[i, j] = max(0, 1 - (dist / radius) ** 2)

					# Apply very subtle bruise coloration
					bruised_img = dark_skin.copy()
					visibility_factor = 0.2

					bruised_img[:, :, 0] = np.clip(dark_skin[:, :, 0] - 10 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)
					bruised_img[:, :, 1] = np.clip(dark_skin[:, :, 1] - 5 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)
					bruised_img[:, :, 2] = np.clip(dark_skin[:, :, 2] - 5 * bruise_mask * visibility_factor, 0, 255).astype(np.uint8)

					st.image(bruised_img, use_container_width=True)

					st.markdown("""
					**Issue:** Minimal contrast between bruise and surrounding tissue.

					**Solution:** Dual-wavelength ALS imaging with specialized channel enhancement.
					""")

				with miss_col2:
					st.markdown("#### Example 2: Tattoo Misclassification")

					# Generate a synthetic "tattoo" case
					img_size = 300
					medium_skin = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([180, 140, 120], dtype=np.uint8)

					# Add some natural skin texture
					texture = np.random.normal(0, 8, (img_size, img_size, 3))
					medium_skin = np.clip(medium_skin + texture, 0, 255).astype(np.uint8)

					# Add a tattoo-like pattern that could be confused with a bruise
					tattoo_img = medium_skin.copy()

					# Draw a simple tribal-like tattoo pattern
					for i in range(100, 200):
						for j in range(100, 200):
							# Create a pattern that might be confused with a bruise
							if ((i - 150)**2 + (j - 150)**2 < 40**2) and not ((i - 150)**2 + (j - 150)**2 < 25**2):
								tattoo_img[i, j] = [40, 40, 40]  # Dark tattoo ink

					st.image(tattoo_img, use_container_width=True)

					st.markdown("""
					**Issue:** Tattoo pattern misclassified as bruise.

					**Solution:** Enhanced model training with tattoo examples and spectral analysis (tattoos and bruises have different spectral signatures under ALS).
					""")

			# Dashboard guidance
			st.markdown("""
			### Integrating the Fairness Dashboard

			This fairness monitoring dashboard would be integrated into the EAS-ID platform's development and deployment workflow:

			1. **During Development**: Track fairness metrics across iterative model improvements
			2. **In Clinical Testing**: Monitor performance across different clinical sites and patient demographics
			3. **Post-Deployment**: Continuous monitoring for performance drift or emergent bias
			4. **For Stakeholders**: Transparent reporting of system performance to clinical and community partners

			The dashboard would support the NIH AIM-AHEAD initiative's goals for AI health equity by providing transparent monitoring of algorithm performance across diverse populations.
			""")
