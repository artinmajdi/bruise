"""Gemini module for bruise detection project."""


from .app import Dashboard
from .computer_vision_panel import ComputerVisionPanel
from .fairness_panel import FairnessPanel
from .data_engineering_panel import DataEngineeringPanel
from .mobile_deployment_panel import MobileDeploymentPanel
from .leadership_panel import LeadershipPanel
from .funding_impact_panel import FundingImpactPanel


__all__ = [
    "Dashboard",
    "ComputerVisionPanel",
    "FairnessPanel",
    "DataEngineeringPanel",
    "MobileDeploymentPanel",
    "LeadershipPanel",
    "FundingImpactPanel",
]

