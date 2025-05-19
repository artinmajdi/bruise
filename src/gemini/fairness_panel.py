
import streamlit as st
import pandas as pd
import numpy as np

class FairnessPanel:
	"""
	Handles the display and logic for the Fairness in AI section of the dashboard.
	Focuses on metrics for equitable performance.
	"""

	def _calculate_fairness_metrics(self, tp_group1, fn_group1, fp_group1, tn_group1,
								   tp_group2, fn_group2, fp_group2, tn_group2):
		"""
		Calculates various fairness metrics based on confusion matrix components for two groups.
		Returns a dictionary of metrics.
		"""
		metrics = {}

		# Helper to avoid division by zero
		def safe_div(num, den):
			return num / den if den != 0 else 0

		# --- Metrics for Group 1 ---
		p1 = tp_group1 + fn_group1  # Total actual positives for group 1
		n1 = fp_group1 + tn_group1  # Total actual negatives for group 1
		pp1 = tp_group1 + fp_group1 # Total predicted positives for group 1

		metrics['tpr_g1'] = safe_div(tp_group1, p1) # True Positive Rate (Recall/Sensitivity)
		metrics['fpr_g1'] = safe_div(fp_group1, n1) # False Positive Rate
		metrics['ppv_g1'] = safe_div(tp_group1, pp1) # Positive Predictive Value (Precision)
		metrics['accuracy_g1'] = safe_div(tp_group1 + tn_group1, p1 + n1)
		metrics['selection_rate_g1'] = safe_div(pp1, p1 + n1) # Proportion of group predicted positive

		# --- Metrics for Group 2 ---
		p2 = tp_group2 + fn_group2  # Total actual positives for group 2
		n2 = fp_group2 + tn_group2  # Total actual negatives for group 2
		pp2 = tp_group2 + fp_group2 # Total predicted positives for group 2

		metrics['tpr_g2'] = safe_div(tp_group2, p2) # True Positive Rate (Recall/Sensitivity)
		metrics['fpr_g2'] = safe_div(fp_group2, n2) # False Positive Rate
		metrics['ppv_g2'] = safe_div(tp_group2, pp2) # Positive Predictive Value (Precision)
		metrics['accuracy_g2'] = safe_div(tp_group2 + tn_group2, p2 + n2)
		metrics['selection_rate_g2'] = safe_div(pp2, p2 + n2) # Proportion of group predicted positive

		# --- Group Fairness Metrics (Differences and Ratios) ---
		# Demographic Parity (Statistical Parity)
		# Aims for equal selection rates across groups.
		metrics['demographic_parity_difference'] = metrics['selection_rate_g1'] - metrics['selection_rate_g2']
		metrics['demographic_parity_ratio'] = safe_div(metrics['selection_rate_g1'], metrics['selection_rate_g2'])

		# Equal Opportunity
		# Aims for equal True Positive Rates across groups.
		metrics['equal_opportunity_difference'] = metrics['tpr_g1'] - metrics['tpr_g2']
		metrics['equal_opportunity_ratio'] = safe_div(metrics['tpr_g1'], metrics['tpr_g2'])

		# Equalized Odds
		# Aims for equal TPR and equal FPR across groups.
		# We already have TPR difference (Equal Opportunity). Let's add FPR difference.
		metrics['fpr_difference'] = metrics['fpr_g1'] - metrics['fpr_g2']
		# Equalized Odds is satisfied if both equal_opportunity_difference and fpr_difference are close to 0.

		# Predictive Rate Parity
		# Aims for equal Positive Predictive Values (PPV) across groups.
		metrics['predictive_rate_parity_difference'] = metrics['ppv_g1'] - metrics['ppv_g2']
		metrics['predictive_rate_parity_ratio'] = safe_div(metrics['ppv_g1'], metrics['ppv_g2'])

		# Accuracy Parity
		metrics['accuracy_difference'] = metrics['accuracy_g1'] - metrics['accuracy_g2']


		return metrics

	def display(self):
		"""Displays the fairness content."""
		st.header("⚖️ Fairness in AI: Ensuring Equitable Performance")
		st.markdown("""
		**Challenge:** "Describe a metric you would report to show equitable performance."

		**What they're looking for:** Awareness of bias & reporting (e.g., Demographic Parity Difference, Equalized Odds).
		The EAS-ID project specifically aims for equitable detection across all skin tones.
		""")
		st.markdown("---")

		st.subheader("Why Fairness Metrics Matter in Bruise Detection")
		st.markdown("""
		If an AI model for bruise detection performs differently for individuals with varying skin tones, it could lead to:
		- **Under-detection in certain groups:** Missed injuries, leading to lack of care or justice.
		- **Over-detection/False positives in other groups:** Unnecessary investigations or distress.
		- **Erosion of trust:** If the tool is perceived as biased.

		Therefore, it's crucial to measure and report metrics that specifically assess performance equity across relevant demographic groups (e.g., Fitzpatrick skin types).
		""")

		st.subheader("Interactive Fairness Metric Calculator")
		st.markdown("""
		Let's simulate a scenario with two groups (e.g., Lighter Skin Tones vs. Darker Skin Tones).
		Input the number of True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN) for each group based on a hypothetical model's performance.
		The dashboard will then calculate common fairness metrics.
		""")

		col1, col2 = st.columns(2)

		with col1:
			st.markdown("#### Group 1 (e.g., Lighter Skin Tones)")
			st.markdown("Total samples for Group 1:")
			total_g1 = st.number_input("Total Samples G1", min_value=10, value=1000, key="total_g1", help="Total number of individuals in Group 1.")
			actual_pos_g1 = st.slider("Actual Bruises in Group 1", 0, total_g1, int(total_g1*0.2), key="ap_g1", help="Number of individuals in Group 1 who actually have bruises.")
			actual_neg_g1 = total_g1 - actual_pos_g1
			st.caption(f"Actual Negatives (No Bruise) in Group 1: {actual_neg_g1}")


			tp_g1 = st.slider("True Positives (TP) for Group 1", 0, actual_pos_g1, int(actual_pos_g1*0.9), key="tp_g1", help="Correctly detected bruises for Group 1.")
			fn_g1 = actual_pos_g1 - tp_g1
			st.caption(f"False Negatives (FN) for Group 1 (Missed Bruises): {fn_g1}")

			fp_g1 = st.slider("False Positives (FP) for Group 1", 0, actual_neg_g1, int(actual_neg_g1*0.05), key="fp_g1", help="Incorrectly identified bruises when none exist for Group 1.")
			tn_g1 = actual_neg_g1 - fp_g1
			st.caption(f"True Negatives (TN) for Group 1 (Correctly no bruise): {tn_g1}")


		with col2:
			st.markdown("#### Group 2 (e.g., Darker Skin Tones)")
			st.markdown("Total samples for Group 2:")
			total_g2 = st.number_input("Total Samples G2", min_value=10, value=1000, key="total_g2", help="Total number of individuals in Group 2.")
			actual_pos_g2 = st.slider("Actual Bruises in Group 2", 0, total_g2, int(total_g2*0.2), key="ap_g2", help="Number of individuals in Group 2 who actually have bruises.")
			actual_neg_g2 = total_g2 - actual_pos_g2
			st.caption(f"Actual Negatives (No Bruise) in Group 2: {actual_neg_g2}")

			tp_g2 = st.slider("True Positives (TP) for Group 2", 0, actual_pos_g2, int(actual_pos_g2*0.7), key="tp_g2", help="Correctly detected bruises for Group 2.")
			fn_g2 = actual_pos_g2 - tp_g2
			st.caption(f"False Negatives (FN) for Group 2 (Missed Bruises): {fn_g2}")

			fp_g2 = st.slider("False Positives (FP) for Group 2", 0, actual_neg_g2, int(actual_neg_g2*0.1), key="fp_g2", help="Incorrectly identified bruises when none exist for Group 2.")
			tn_g2 = actual_neg_g2 - fp_g2
			st.caption(f"True Negatives (TN) for Group 2 (Correctly no bruise): {tn_g2}")

		if st.button("Calculate Fairness Metrics"):
			if (tp_g1 + fn_g1 + fp_g1 + tn_g1 == 0) or \
			   (tp_g2 + fn_g2 + fp_g2 + tn_g2 == 0):
				st.error("Please ensure all input values (TP, FN, FP, TN) sum up to positive sample sizes for both groups.")
			else:
				results = self._calculate_fairness_metrics(tp_g1, fn_g1, fp_g1, tn_g1, tp_g2, fn_g2, fp_g2, tn_g2)

				st.markdown("---")
				st.subheader("Calculated Metrics")

				res_col1, res_col2 = st.columns(2)
				with res_col1:
					st.markdown("##### Performance - Group 1")
					st.write(f"Accuracy: {results['accuracy_g1']:.3f}")
					st.write(f"True Positive Rate (Sensitivity/Recall): {results['tpr_g1']:.3f}")
					st.write(f"False Positive Rate: {results['fpr_g1']:.3f}")
					st.write(f"Positive Predictive Value (Precision): {results['ppv_g1']:.3f}")
					st.write(f"Selection Rate: {results['selection_rate_g1']:.3f}")

				with res_col2:
					st.markdown("##### Performance - Group 2")
					st.write(f"Accuracy: {results['accuracy_g2']:.3f}")
					st.write(f"True Positive Rate (Sensitivity/Recall): {results['tpr_g2']:.3f}")
					st.write(f"False Positive Rate: {results['fpr_g2']:.3f}")
					st.write(f"Positive Predictive Value (Precision): {results['ppv_g2']:.3f}")
					st.write(f"Selection Rate: {results['selection_rate_g2']:.3f}")

				st.markdown("---")
				st.markdown("##### Group Fairness Metrics (Ideal values are close to 0 for differences, 1 for ratios)")

				st.markdown(f"**Demographic Parity Difference (Selection Rate):** `{results['demographic_parity_difference']:.3f}`")
				st.caption("Measures if both groups have an equal probability of being classified as positive (bruise detected). A value near 0 is desired.")

				st.markdown(f"**Equal Opportunity Difference (TPR):** `{results['equal_opportunity_difference']:.3f}`")
				st.caption("Measures if both groups have an equal True Positive Rate. Crucial for ensuring that actual bruises are detected at similar rates across groups. A value near 0 is desired.")

				st.markdown(f"**Equalized Odds (TPR & FPR differences):**")
				st.caption(f"  - TPR Difference (Equal Opportunity): `{results['equal_opportunity_difference']:.3f}`")
				st.caption(f"  - FPR Difference: `{results['fpr_difference']:.3f}`")
				st.caption("Aims for fairness for both positive and negative instances. Both differences should be close to 0.")

				st.markdown(f"**Predictive Rate Parity Difference (PPV):** `{results['predictive_rate_parity_difference']:.3f}`")
				st.caption("Measures if the likelihood that a positive prediction is correct is the same for both groups. A value near 0 is desired.")

				st.markdown(f"**Accuracy Difference:** `{results['accuracy_difference']:.3f}`")
				st.caption("Difference in overall accuracy between the groups.")


		st.markdown("---")
		st.subheader("Common Fairness Metrics to Report")
		st.markdown("""
		When reporting on equitable performance for a system like EAS-ID, consider these metrics:

		1.  **True Positive Rate (TPR) Parity / Equal Opportunity:**
			* **Definition:** The model should identify actual bruises (true positives) at similar rates across different skin tone groups.
			* **Why it's important:** Ensures that individuals with bruises have an equal chance of their injury being detected, regardless of their skin tone. This is often a primary concern for equity in diagnostic tools.
			* **Reporting:** Report TPR for each group and the difference/ratio between groups.
			* *Example Metric: True Positive Rate Parity Difference (TPR_GroupA - TPR_GroupB)*

		2.  **False Positive Rate (FPR) Parity:**
			* **Definition:** The model should incorrectly identify bruises (false positives) at similar rates across groups.
			* **Why it's important:** Avoids disproportionately subjecting one group to unnecessary follow-up, concern, or incorrect documentation.
			* **Reporting:** Report FPR for each group and the difference/ratio.

		3.  **Equalized Odds:**
			* **Definition:** Satisfied if both TPR Parity and FPR Parity are achieved. This is a stricter condition.
			* **Why it's important:** Ensures the model works fairly for both those with and without the condition (bruises) across groups.

		4.  **Demographic Parity (Statistical Parity):**
			* **Definition:** The proportion of individuals identified as having a bruise should be similar across groups, regardless of whether they actually have one.
			* **Why it's important:** Can be useful, but less so if the actual prevalence of bruises differs significantly between groups for reasons unrelated to bias. For bruise detection, TPR parity is often more critical.
			* **Reporting:** Report selection rates (proportion predicted positive) for each group and the difference/ratio.
			* *Example Metric: Demographic Parity Difference (SelectionRate_GroupA - SelectionRate_GroupB)*

		5.  **Accuracy Parity:**
			* **Definition:** Overall accuracy of the model is similar across groups.
			* **Why it's important:** While intuitive, accuracy can be misleading if class imbalances or the costs of different types of errors vary between groups.

		**For the EAS-ID project, focusing on True Positive Rate Parity (Equal Opportunity) is likely paramount to ensure that bruises are detected equitably. However, monitoring FPR Parity is also crucial to prevent undue burden on any specific group.**
		""")
		st.markdown("<div class='info-box'><h3>Key Takeaway:</h3>No single fairness metric is universally best. The choice depends on the application's context and potential harms of different types of errors. For medical diagnostics like bruise detection, ensuring that those who *need* detection get it (high, equitable TPR) is often a primary goal, while also controlling for false alarms (FPR). Regularly auditing and reporting these metrics across relevant subgroups is essential.</div>", unsafe_allow_html=True)
