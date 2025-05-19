# Fairness Module for Bruise Detection
# Contains classes and functions for fairness metrics and bias evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """
    Implements fairness metrics for evaluating bruise detection across skin tones
    """
    def __init__(self):
        self.supported_metrics = [
            "demographic_parity_difference",
            "equalized_odds_difference",
            "predictive_parity_difference",
            "balanced_accuracy_difference",
            "threshold_stability"
        ]
        self.skin_tone_groups = {
            "Light": [1, 2],     # Fitzpatrick I-II
            "Medium": [3, 4],    # Fitzpatrick III-IV
            "Dark": [5, 6]       # Fitzpatrick V-VI
        }
        
    def demographic_parity_difference(self, predictions, groups):
        """
        Calculate the demographic parity difference across groups
        
        Parameters:
        - predictions: binary predictions (0/1)
        - groups: group labels for each prediction
        
        Returns:
        - dpd: max difference in positive prediction rates between groups
        - group_rates: dictionary of positive prediction rates by group
        """
        group_rates = {}
        
        for group in np.unique(groups):
            group_preds = predictions[groups == group]
            positive_rate = np.mean(group_preds)
            group_rates[group] = positive_rate
        
        # Calculate max difference between any two groups
        rates = list(group_rates.values())
        dpd = max(rates) - min(rates)
        
        return dpd, group_rates
    
    def equalized_odds_difference(self, predictions, labels, groups):
        """
        Calculate the equalized odds difference across groups
        
        Parameters:
        - predictions: binary predictions (0/1)
        - labels: ground truth labels (0/1)
        - groups: group labels for each prediction
        
        Returns:
        - eod_tpr: max difference in true positive rates between groups
        - eod_fpr: max difference in false positive rates between groups
        - group_tpr: dictionary of true positive rates by group
        - group_fpr: dictionary of false positive rates by group
        """
        group_tpr = {}
        group_fpr = {}
        
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            # True positive rate (sensitivity)
            if np.sum(group_labels == 1) > 0:
                tpr = np.sum((group_preds == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
                group_tpr[group] = tpr
            else:
                group_tpr[group] = np.nan
                
            # False positive rate (1 - specificity)
            if np.sum(group_labels == 0) > 0:
                fpr = np.sum((group_preds == 1) & (group_labels == 0)) / np.sum(group_labels == 0)
                group_fpr[group] = fpr
            else:
                group_fpr[group] = np.nan
        
        # Filter out NaN values
        tpr_values = [v for v in group_tpr.values() if not np.isnan(v)]
        fpr_values = [v for v in group_fpr.values() if not np.isnan(v)]
        
        # Calculate max differences
        eod_tpr = max(tpr_values) - min(tpr_values) if tpr_values else np.nan
        eod_fpr = max(fpr_values) - min(fpr_values) if fpr_values else np.nan
        
        return eod_tpr, eod_fpr, group_tpr, group_fpr
    
    def predictive_parity_difference(self, predictions, labels, groups):
        """
        Calculate the predictive parity difference across groups
        
        Parameters:
        - predictions: binary predictions (0/1)
        - labels: ground truth labels (0/1)
        - groups: group labels for each prediction
        
        Returns:
        - ppd: max difference in positive predictive values between groups
        - group_ppv: dictionary of positive predictive values by group
        """
        group_ppv = {}
        
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            # Positive predictive value (precision)
            if np.sum(group_preds == 1) > 0:
                ppv = np.sum((group_preds == 1) & (group_labels == 1)) / np.sum(group_preds == 1)
                group_ppv[group] = ppv
            else:
                group_ppv[group] = np.nan
        
        # Filter out NaN values
        ppv_values = [v for v in group_ppv.values() if not np.isnan(v)]
        
        # Calculate max difference
        ppd = max(ppv_values) - min(ppv_values) if ppv_values else np.nan
        
        return ppd, group_ppv
    
    def balanced_accuracy_difference(self, predictions, labels, groups):
        """
        Calculate the balanced accuracy difference across groups
        
        Parameters:
        - predictions: binary predictions (0/1)
        - labels: ground truth labels (0/1)
        - groups: group labels for each prediction
        
        Returns:
        - bad: max difference in balanced accuracy between groups
        - group_ba: dictionary of balanced accuracy by group
        """
        group_ba = {}
        
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            # True positive rate
            if np.sum(group_labels == 1) > 0:
                tpr = np.sum((group_preds == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
            else:
                tpr = np.nan
                
            # True negative rate
            if np.sum(group_labels == 0) > 0:
                tnr = np.sum((group_preds == 0) & (group_labels == 0)) / np.sum(group_labels == 0)
            else:
                tnr = np.nan
            
            # Balanced accuracy
            if not (np.isnan(tpr) or np.isnan(tnr)):
                ba = (tpr + tnr) / 2
                group_ba[group] = ba
            else:
                group_ba[group] = np.nan
        
        # Filter out NaN values
        ba_values = [v for v in group_ba.values() if not np.isnan(v)]
        
        # Calculate max difference
        bad = max(ba_values) - min(ba_values) if ba_values else np.nan
        
        return bad, group_ba
    
    def threshold_stability(self, scores, labels, groups, thresholds=None):
        """
        Evaluate how stable optimal thresholds are across groups
        
        Parameters:
        - scores: probability scores (0-1)
        - labels: ground truth labels (0/1)
        - groups: group labels for each prediction
        - thresholds: list of thresholds to evaluate (default: 10 values from 0.1 to 0.9)
        
        Returns:
        - threshold_variability: standard deviation of optimal thresholds
        - group_thresholds: dictionary of optimal thresholds by group
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        group_thresholds = {}
        
        for group in np.unique(groups):
            group_mask = (groups == group)
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]
            
            # Find threshold that maximizes balanced accuracy
            best_ba = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                group_preds = (group_scores >= threshold).astype(int)
                
                # Calculate TPR and TNR
                if np.sum(group_labels == 1) > 0:
                    tpr = np.sum((group_preds == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
                else:
                    tpr = 0
                    
                if np.sum(group_labels == 0) > 0:
                    tnr = np.sum((group_preds == 0) & (group_labels == 0)) / np.sum(group_labels == 0)
                else:
                    tnr = 0
                
                # Balanced accuracy
                ba = (tpr + tnr) / 2
                
                if ba > best_ba:
                    best_ba = ba
                    best_threshold = threshold
            
            group_thresholds[group] = best_threshold
        
        # Calculate threshold variability
        threshold_values = list(group_thresholds.values())
        threshold_variability = np.std(threshold_values)
        
        return threshold_variability, group_thresholds
    
    def evaluate_all_metrics(self, predictions, labels, scores, groups):
        """
        Evaluate all supported fairness metrics
        
        Parameters:
        - predictions: binary predictions (0/1)
        - labels: ground truth labels (0/1)
        - scores: probability scores (0-1)
        - groups: group labels for each prediction
        
        Returns:
        - metrics: dictionary of all calculated fairness metrics
        - group_performance: dictionary of group-specific performance metrics
        """
        metrics = {}
        group_performance = {}
        
        # Demographic parity
        dpd, group_rates = self.demographic_parity_difference(predictions, groups)
        metrics["demographic_parity_difference"] = dpd
        
        # Equalized odds
        eod_tpr, eod_fpr, group_tpr, group_fpr = self.equalized_odds_difference(predictions, labels, groups)
        metrics["equalized_odds_tpr_difference"] = eod_tpr
        metrics["equalized_odds_fpr_difference"] = eod_fpr
        
        # Predictive parity
        ppd, group_ppv = self.predictive_parity_difference(predictions, labels, groups)
        metrics["predictive_parity_difference"] = ppd
        
        # Balanced accuracy
        bad, group_ba = self.balanced_accuracy_difference(predictions, labels, groups)
        metrics["balanced_accuracy_difference"] = bad
        
        # Threshold stability
        threshold_variability, group_thresholds = self.threshold_stability(scores, labels, groups)
        metrics["threshold_variability"] = threshold_variability
        
        # Compile group performance
        for group in np.unique(groups):
            group_performance[group] = {
                "positive_rate": group_rates.get(group, np.nan),
                "true_positive_rate": group_tpr.get(group, np.nan),
                "false_positive_rate": group_fpr.get(group, np.nan),
                "positive_predictive_value": group_ppv.get(group, np.nan),
                "balanced_accuracy": group_ba.get(group, np.nan),
                "optimal_threshold": group_thresholds.get(group, np.nan)
            }
        
        return metrics, group_performance
    
    def fairness_report(self, metrics, group_performance, fairness_thresholds=None):
        """
        Generate a formatted fairness report
        
        Parameters:
        - metrics: dictionary of calculated fairness metrics
        - group_performance: dictionary of group-specific performance metrics
        - fairness_thresholds: dictionary of threshold values for each metric
        
        Returns:
        - report: dictionary with formatted report and assessment
        """
        if fairness_thresholds is None:
            fairness_thresholds = {
                "demographic_parity_difference": 0.05,
                "equalized_odds_tpr_difference": 0.05,
                "equalized_odds_fpr_difference": 0.03,
                "predictive_parity_difference": 0.05,
                "balanced_accuracy_difference": 0.05,
                "threshold_variability": 0.10
            }
        
        # Assess each metric
        assessment = {}
        for metric, value in metrics.items():
            if np.isnan(value):
                assessment[metric] = "Unknown"
            elif metric in fairness_thresholds:
                if value <= fairness_thresholds[metric]:
                    assessment[metric] = "Passed"
                else:
                    assessment[metric] = "Failed"
            else:
                assessment[metric] = "No threshold"
        
        # Overall assessment
        passed = sum(1 for status in assessment.values() if status == "Passed")
        failed = sum(1 for status in assessment.values() if status == "Failed")
        unknown = sum(1 for status in assessment.values() if status == "Unknown")
        
        if failed == 0 and unknown == 0:
            overall = "All fairness criteria met"
        elif failed == 0 and unknown > 0:
            overall = "Partial assessment (some metrics unknown)"
        elif failed <= 2:
            overall = "Most fairness criteria met"
        else:
            overall = "Significant fairness issues detected"
        
        # Detailed group performance table
        groups = list(group_performance.keys())
        metrics_list = list(next(iter(group_performance.values())).keys())
        
        performance_table = {}
        for metric in metrics_list:
            performance_table[metric] = {group: group_performance[group][metric] for group in groups}
        
        report = {
            "metrics": metrics,
            "assessment": assessment,
            "overall": overall,
            "performance_table": performance_table,
            "fairness_thresholds": fairness_thresholds,
            "passed_count": passed,
            "failed_count": failed,
            "unknown_count": unknown
        }
        
        return report

def generate_fairness_report(predictions, scores, labels, skin_tones):
    """
    Generate a comprehensive fairness report for bruise detection across skin tones
    
    Parameters:
    - predictions: binary predictions (0/1)
    - scores: probability scores (0-1)
    - labels: ground truth labels (0/1) 
    - skin_tones: Fitzpatrick skin type for each example (1-6)
    
    Returns:
    - report: dictionary with fairness metrics and visualizations
    """
    # Create skin tone groups
    skin_tone_groups = np.zeros_like(skin_tones, dtype=object)
    skin_tone_groups[(skin_tones == 1) | (skin_tones == 2)] = "Type I-II"
    skin_tone_groups[(skin_tones == 3) | (skin_tones == 4)] = "Type III-IV"
    skin_tone_groups[(skin_tones == 5) | (skin_tones == 6)] = "Type V-VI"
    
    # Calculate fairness metrics
    fairness = FairnessMetrics()
    metrics, group_performance = fairness.evaluate_all_metrics(
        predictions, labels, scores, skin_tone_groups
    )
    
    # Generate full report
    fairness_thresholds = {
        "demographic_parity_difference": 0.05,
        "equalized_odds_tpr_difference": 0.05,
        "equalized_odds_fpr_difference": 0.03,
        "predictive_parity_difference": 0.05,
        "balanced_accuracy_difference": 0.05,
        "threshold_variability": 0.10
    }
    
    report = fairness.fairness_report(metrics, group_performance, fairness_thresholds)
    
    # Calculate per-group confusion matrices
    group_confusion_matrices = {}
    for group in np.unique(skin_tone_groups):
        group_mask = (skin_tone_groups == group)
        cm = confusion_matrix(
            labels[group_mask], 
            predictions[group_mask],
            labels=[0, 1]
        )
        group_confusion_matrices[group] = cm
    
    # Add to report
    report["confusion_matrices"] = group_confusion_matrices
    
    # Calculate ROC curve data per group
    # This would normally use sklearn.metrics.roc_curve but we'll simulate for simplicity
    group_roc_data = {}
    for group in np.unique(skin_tone_groups):
        group_mask = (skin_tone_groups == group)
        group_scores = scores[group_mask]
        group_labels = labels[group_mask]
        
        # Simulate ROC curve data
        thresholds = np.linspace(0, 1, 100)
        tpr = []
        fpr = []
        
        for threshold in thresholds:
            group_preds = (group_scores >= threshold).astype(int)
            
            # TPR and FPR
            if np.sum(group_labels == 1) > 0:
                tpr.append(np.sum((group_preds == 1) & (group_labels == 1)) / np.sum(group_labels == 1))
            else:
                tpr.append(0)
                
            if np.sum(group_labels == 0) > 0:
                fpr.append(np.sum((group_preds == 1) & (group_labels == 0)) / np.sum(group_labels == 0))
            else:
                fpr.append(0)
        
        group_roc_data[group] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": np.trapz(tpr, fpr)  # Approximate AUC calculation
        }
    
    # Add to report
    report["roc_data"] = group_roc_data
    
    # Calculate performance by bruise age
    # For simulation, we'll generate some synthetic age data
    np.random.seed(42)
    bruise_ages = np.random.randint(12, 168, size=len(labels))  # 12-168 hours
    
    age_bins = [0, 24, 72, 120, 168]
    age_labels = ["0-24h", "24-72h", "72-120h", "120h+"]
    
    age_bin_indices = np.digitize(bruise_ages, age_bins) - 1
    
    # Performance by age and skin tone
    age_performance = {}
    for age_idx, age_label in enumerate(age_labels):
        age_mask = (age_bin_indices == age_idx)
        
        age_performance[age_label] = {}
        for group in np.unique(skin_tone_groups):
            group_mask = (skin_tone_groups == group) & age_mask
            
            if np.sum(group_mask) > 0:
                # TPR
                if np.sum(labels[group_mask] == 1) > 0:
                    tpr = np.sum((predictions[group_mask] == 1) & (labels[group_mask] == 1)) / np.sum(labels[group_mask] == 1)
                else:
                    tpr = np.nan
                    
                # TNR
                if np.sum(labels[group_mask] == 0) > 0:
                    tnr = np.sum((predictions[group_mask] == 0) & (labels[group_mask] == 0)) / np.sum(labels[group_mask] == 0)
                else:
                    tnr = np.nan
                
                # Balanced accuracy
                if not (np.isnan(tpr) or np.isnan(tnr)):
                    ba = (tpr + tnr) / 2
                else:
                    ba = np.nan
                
                age_performance[age_label][group] = {
                    "tpr": tpr,
                    "tnr": tnr,
                    "balanced_accuracy": ba
                }
            else:
                age_performance[age_label][group] = {
                    "tpr": np.nan,
                    "tnr": np.nan,
                    "balanced_accuracy": np.nan
                }
    
    # Add to report
    report["age_performance"] = age_performance
    
    return report
