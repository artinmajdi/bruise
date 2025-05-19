# Leadership Module for Bruise Detection
# Contains classes and functions for team management and coordination

import pandas as pd
from datetime import datetime, timedelta
import random

class TeamManagement:
    """
    Class for managing interdisciplinary research teams
    """
    def __init__(self):
        self.team_structure = self._initialize_team_structure()
        self.team_challenges = self._initialize_challenges()
        self.meeting_schedule = self._initialize_meeting_schedule()
        self.project_milestones = self._initialize_project_milestones()
    
    def _initialize_team_structure(self):
        """
        Initialize team structure with roles and responsibilities
        """
        team_structure = {
            "leadership": [
                {
                    "role": "Principal Investigator",
                    "discipline": "Nursing Science",
                    "responsibilities": [
                        "Overall project vision",
                        "Stakeholder management",
                        "Grant management",
                        "Clinical expertise"
                    ],
                    "background": "Forensic nursing with expertise in bruise assessment"
                },
                {
                    "role": "Co-Principal Investigator",
                    "discipline": "Health Informatics",
                    "responsibilities": [
                        "Technical vision",
                        "AI strategy",
                        "Research methodology",
                        "Data management"
                    ],
                    "background": "Machine learning and healthcare informatics"
                },
                {
                    "role": "Co-Principal Investigator",
                    "discipline": "Engineering",
                    "responsibilities": [
                        "Imaging technologies",
                        "Hardware integration",
                        "System architecture",
                        "Technical validation"
                    ],
                    "background": "Computer vision and imaging technologies"
                },
                {
                    "role": "Postdoctoral Researcher",
                    "discipline": "Computer Science/Health Informatics",
                    "responsibilities": [
                        "Technical implementation",
                        "Team coordination",
                        "Research execution",
                        "Student supervision"
                    ],
                    "background": "Deep learning, mobile health, and computer vision"
                }
            ],
            "technical_team": [
                {
                    "role": "Computer Vision Engineer",
                    "discipline": "Computer Science",
                    "responsibilities": [
                        "Deep learning models",
                        "Image processing pipeline",
                        "Model evaluation",
                        "Technical documentation"
                    ],
                    "background": "Computer vision and machine learning"
                },
                {
                    "role": "Mobile Developer",
                    "discipline": "Software Engineering",
                    "responsibilities": [
                        "Mobile application",
                        "Hardware integration",
                        "User interface",
                        "Offline functionality"
                    ],
                    "background": "Mobile development and healthcare applications"
                },
                {
                    "role": "Database Specialist",
                    "discipline": "Information Technology",
                    "responsibilities": [
                        "FHIR database design",
                        "Data security",
                        "Clinical integration",
                        "Audit logging"
                    ],
                    "background": "Healthcare IT and secure database design"
                }
            ],
            "clinical_team": [
                {
                    "role": "Forensic Nurse",
                    "discipline": "Nursing",
                    "responsibilities": [
                        "Clinical requirements",
                        "Validation protocol",
                        "Workflow integration",
                        "User testing"
                    ],
                    "background": "Forensic nursing and intimate partner violence care"
                },
                {
                    "role": "UX Researcher",
                    "discipline": "Human-Computer Interaction",
                    "responsibilities": [
                        "User research",
                        "Interface design",
                        "Usability testing",
                        "Accessibility"
                    ],
                    "background": "Healthcare UX and trauma-informed design"
                }
            ],
            "students": [
                {
                    "role": "Graduate Student (CS)",
                    "discipline": "Computer Science",
                    "responsibilities": [
                        "Algorithm implementation",
                        "Performance testing",
                        "Literature review",
                        "Technical documentation"
                    ],
                    "background": "Machine learning and computer vision"
                },
                {
                    "role": "Graduate Student (Nursing)",
                    "discipline": "Nursing Science",
                    "responsibilities": [
                        "Clinical validation",
                        "Workflow analysis",
                        "Literature review",
                        "Data collection"
                    ],
                    "background": "Forensic nursing and injury documentation"
                },
                {
                    "role": "Graduate Student (Engineering)",
                    "discipline": "Engineering",
                    "responsibilities": [
                        "Hardware integration",
                        "Prototype development",
                        "Testing procedures",
                        "Technical documentation"
                    ],
                    "background": "Electrical engineering and imaging systems"
                }
            ]
        }
        return team_structure
    
    def _initialize_challenges(self):
        """
        Initialize common team challenges and mitigation strategies
        """
        challenges = [
            {
                "challenge": "Technical-clinical communication barriers",
                "description": "Technical team members and clinical team members often use different terminology and have different priorities.",
                "impact": "High",
                "strategies": [
                    "Establish shared glossary of terms",
                    "Regular knowledge translation sessions",
                    "Visual communication tools",
                    "Cross-disciplinary pair programming"
                ],
                "metrics": [
                    "Reduction in clarification requests",
                    "Shared understanding verification",
                    "Cross-disciplinary collaboration satisfaction"
                ]
            },
            {
                "challenge": "Different work rhythms across disciplines",
                "description": "Clinical team members often have irregular schedules due to patient care responsibilities, while technical team members typically work standard hours.",
                "impact": "Medium",
                "strategies": [
                    "Flexible core hours",
                    "Asynchronous communication tools",
                    "Buffer time in project schedules",
                    "Recorded knowledge sessions"
                ],
                "metrics": [
                    "Meeting attendance rates",
                    "Communication response times",
                    "Work-life balance satisfaction"
                ]
            },
            {
                "challenge": "Varying technical skill levels among students",
                "description": "Graduate students come from different disciplines and have varying levels of technical expertise, especially in AI and programming.",
                "impact": "Medium",
                "strategies": [
                    "Tiered task assignment",
                    "Peer mentoring system",
                    "Skills assessment and targeted training",
                    "Gradual complexity progression"
                ],
                "metrics": [
                    "Skill improvement metrics",
                    "Task completion rates",
                    "Student confidence surveys"
                ]
            },
            {
                "challenge": "Balancing innovation with clinical requirements",
                "description": "Technical team members may prioritize innovative approaches, while clinical team members focus on practical implementation and reliability.",
                "impact": "High",
                "strategies": [
                    "Clinical validation criteria in technical specs",
                    "Rapid prototype testing with clinicians",
                    "User story mapping",
                    "Prioritization matrix for features"
                ],
                "metrics": [
                    "Clinical acceptance rates",
                    "Technical innovation metrics",
                    "Feature adoption rates"
                ]
            },
            {
                "challenge": "Publication priorities vs. development timeline",
                "description": "Academic focus on publications may compete with product development priorities and timelines.",
                "impact": "Medium",
                "strategies": [
                    "Publication planning aligned with project milestones",
                    "Targeted conference submissions",
                    "Balanced authorship plan",
                    "Research-development parallel tracks"
                ],
                "metrics": [
                    "Publication outputs",
                    "Development milestone adherence",
                    "Team satisfaction surveys"
                ]
            },
            {
                "challenge": "Integration of feedback from diverse stakeholders",
                "description": "Feedback from clinicians, patients, administrators, and technical experts may conflict and be difficult to prioritize.",
                "impact": "Medium",
                "strategies": [
                    "Structured feedback categorization",
                    "Impact-effort prioritization matrix",
                    "Regular stakeholder alignment sessions",
                    "Transparent decision documentation"
                ],
                "metrics": [
                    "Stakeholder satisfaction",
                    "Feedback implementation rate",
                    "Decision consensus metrics"
                ]
            }
        ]
        return challenges
    
    def _initialize_meeting_schedule(self):
        """
        Initialize team meeting schedule
        """
        today = datetime.now()
        start_date = today - timedelta(days=today.weekday())  # Previous Monday
        
        meeting_schedule = []
        
        # Weekly team meeting
        for week in range(4):
            meeting_schedule.append({
                "name": "Full Team Meeting",
                "date": (start_date + timedelta(days=7 * week + 2)).strftime("%Y-%m-%d"),  # Wednesdays
                "time": "10:00 - 11:30",
                "participants": "All team members",
                "agenda": [
                    "Project updates",
                    "Milestone progress",
                    "Blockers and solutions",
                    "Next week priorities"
                ],
                "frequency": "Weekly"
            })
        
        # Technical team meetings
        for week in range(4):
            meeting_schedule.append({
                "name": "Technical Team Sync",
                "date": (start_date + timedelta(days=7 * week + 1)).strftime("%Y-%m-%d"),  # Tuesdays
                "time": "14:00 - 15:00",
                "participants": "Technical team members",
                "agenda": [
                    "Technical challenges",
                    "Implementation progress",
                    "Code review",
                    "Architecture decisions"
                ],
                "frequency": "Weekly"
            })
        
        # Clinical team meetings
        for week in range(4):
            meeting_schedule.append({
                "name": "Clinical Team Sync",
                "date": (start_date + timedelta(days=7 * week + 4)).strftime("%Y-%m-%d"),  # Fridays
                "time": "13:00 - 14:00",
                "participants": "Clinical team members",
                "agenda": [
                    "Clinical validation progress",
                    "User feedback",
                    "Workflow integration",
                    "Clinical requirements refinement"
                ],
                "frequency": "Weekly"
            })
        
        # Leadership meetings
        for week in range(4):
            if week % 2 == 0:  # Bi-weekly
                meeting_schedule.append({
                    "name": "Leadership Strategy Meeting",
                    "date": (start_date + timedelta(days=7 * week)).strftime("%Y-%m-%d"),  # Mondays
                    "time": "11:00 - 12:00",
                    "participants": "Principal Investigators, Postdoc",
                    "agenda": [
                        "Project strategy",
                        "Resource allocation",
                        "Risk management",
                        "External communications"
                    ],
                    "frequency": "Bi-weekly"
                })
        
        # Student mentorship meetings
        for week in range(4):
            meeting_schedule.append({
                "name": "Student Mentorship Session",
                "date": (start_date + timedelta(days=7 * week + 3)).strftime("%Y-%m-%d"),  # Thursdays
                "time": "15:00 - 16:00",
                "participants": "Postdoc, Graduate Students",
                "agenda": [
                    "Research progress",
                    "Technical skills development",
                    "Publication planning",
                    "Career development"
                ],
                "frequency": "Weekly"
            })
        
        # Monthly all-hands meeting
        meeting_schedule.append({
            "name": "Monthly All-Hands Meeting",
            "date": (start_date + timedelta(days=21)).strftime("%Y-%m-%d"),  # Last week of month
            "time": "13:00 - 15:00",
            "participants": "All team members and stakeholders",
            "agenda": [
                "Monthly progress review",
                "Demos and presentations",
                "Stakeholder feedback",
                "Upcoming milestones and goals"
            ],
            "frequency": "Monthly"
        })
        
        return meeting_schedule
    
    def _initialize_project_milestones(self):
        """
        Initialize project milestones
        """
        today = datetime.now()
        start_date = today - timedelta(days=60)  # Project started 2 months ago
        
        milestones = [
            {
                "name": "Project Kickoff",
                "description": "Initiate project, establish team, and define initial requirements",
                "date": start_date.strftime("%Y-%m-%d"),
                "status": "Completed",
                "deliverables": [
                    "Project charter",
                    "Team onboarding",
                    "Requirements document"
                ],
                "dependencies": []
            },
            {
                "name": "Data Collection Protocol",
                "description": "Define and validate data collection protocol for bruise images",
                "date": (start_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                "status": "Completed",
                "deliverables": [
                    "Clinical protocol document",
                    "IRB approval",
                    "Data collection tools"
                ],
                "dependencies": ["Project Kickoff"]
            },
            {
                "name": "Initial Deep Learning Model",
                "description": "Develop baseline model for bruise detection",
                "date": (start_date + timedelta(days=90)).strftime("%Y-%m-%d"),
                "status": "In Progress",
                "deliverables": [
                    "Baseline model architecture",
                    "Training pipeline",
                    "Performance metrics"
                ],
                "dependencies": ["Data Collection Protocol"]
            },
            {
                "name": "Mobile App Prototype",
                "description": "Develop first functional prototype of mobile application",
                "date": (start_date + timedelta(days=120)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "UI/UX design",
                    "Functional prototype",
                    "User testing results"
                ],
                "dependencies": ["Initial Deep Learning Model"]
            },
            {
                "name": "Database Implementation",
                "description": "Implement FHIR-compliant database for bruise data",
                "date": (start_date + timedelta(days=105)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Database schema",
                    "Security implementation",
                    "API documentation"
                ],
                "dependencies": ["Data Collection Protocol"]
            },
            {
                "name": "Fairness Evaluation Framework",
                "description": "Develop framework for evaluating model performance across skin tones",
                "date": (start_date + timedelta(days=150)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Fairness metrics",
                    "Evaluation protocol",
                    "Reporting dashboard"
                ],
                "dependencies": ["Initial Deep Learning Model"]
            },
            {
                "name": "Initial Clinical Validation",
                "description": "Conduct first clinical validation study",
                "date": (start_date + timedelta(days=180)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Validation protocol",
                    "Study results",
                    "Improvement recommendations"
                ],
                "dependencies": ["Mobile App Prototype", "Fairness Evaluation Framework"]
            },
            {
                "name": "First Conference Publication",
                "description": "Submit and present first conference paper",
                "date": (start_date + timedelta(days=210)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Conference paper",
                    "Presentation",
                    "Publication"
                ],
                "dependencies": ["Initial Clinical Validation"]
            },
            {
                "name": "System Integration",
                "description": "Integrate all components into cohesive system",
                "date": (start_date + timedelta(days=240)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Integrated system",
                    "API documentation",
                    "System tests"
                ],
                "dependencies": ["Mobile App Prototype", "Database Implementation"]
            },
            {
                "name": "Phase 1 Completion",
                "description": "Complete first phase of project",
                "date": (start_date + timedelta(days=365)).strftime("%Y-%m-%d"),
                "status": "Planned",
                "deliverables": [
                    "Final report",
                    "Demonstration",
                    "Future plan"
                ],
                "dependencies": ["System Integration", "First Conference Publication"]
            }
        ]
        
        return milestones
    
    def get_team_roles(self):
        """
        Get all team roles as a flat list
        """
        roles = []
        for category in self.team_structure:
            for member in self.team_structure[category]:
                roles.append(member)
        return roles
    
    def get_team_composition_summary(self):
        """
        Get summary of team composition by discipline
        """
        disciplines = {}
        for category in self.team_structure:
            for member in self.team_structure[category]:
                discipline = member["discipline"]
                if discipline not in disciplines:
                    disciplines[discipline] = 0
                disciplines[discipline] += 1
        
        return disciplines
    
    def get_meetings_by_type(self, meeting_type=None):
        """
        Get meetings filtered by type
        """
        if meeting_type is None:
            return self.meeting_schedule
        
        filtered_meetings = []
        for meeting in self.meeting_schedule:
            if meeting_type.lower() in meeting["name"].lower():
                filtered_meetings.append(meeting)
        
        return filtered_meetings
    
    def get_upcoming_milestones(self, days=90):
        """
        Get milestones coming up within specified days
        """
        today = datetime.now()
        upcoming = []
        
        for milestone in self.project_milestones:
            milestone_date = datetime.strptime(milestone["date"], "%Y-%m-%d")
            days_until = (milestone_date - today).days
            
            if 0 <= days_until <= days and milestone["status"] != "Completed":
                milestone_with_days = milestone.copy()
                milestone_with_days["days_until"] = days_until
                upcoming.append(milestone_with_days)
        
        return upcoming
    
    def get_milestone_dependencies(self):
        """
        Get milestone dependencies for visualization
        """
        dependencies = []
        for milestone in self.project_milestones:
            for dependency in milestone["dependencies"]:
                dependencies.append({
                    "from": dependency,
                    "to": milestone["name"]
                })
        
        return dependencies
    
    def get_milestone_timeline_data(self):
        """
        Get milestone data formatted for timeline visualization
        """
        timeline_data = []
        for milestone in self.project_milestones:
            milestone_date = datetime.strptime(milestone["date"], "%Y-%m-%d")
            
            # Estimate duration based on dependencies and next milestones
            duration_days = 30  # Default duration
            
            timeline_data.append({
                "Task": milestone["name"],
                "Start": milestone_date.strftime("%Y-%m-%d"),
                "Duration": duration_days,
                "Status": milestone["status"],
                "Description": milestone["description"]
            })
        
        return timeline_data
    
    def generate_team_charter(self):
        """
        Generate a team charter document
        """
        charter = {
            "title": "EAS-ID Project Team Charter",
            "version": "1.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "mission": "To develop an equitable, accessible, and clinically validated platform for bruise detection across all skin tones, improving care for victims of intimate partner violence.",
            "vision": "Creating technology that addresses healthcare disparities and improves forensic documentation accuracy, ultimately leading to better outcomes for survivors of violence.",
            "values": [
                "Equity in healthcare technology",
                "Scientific rigor and evidence-based approaches",
                "Interdisciplinary collaboration and respect",
                "User-centered design and clinical relevance",
                "Transparency and reproducibility in AI"
            ],
            "goals": [
                "Develop a mobile AI platform that accurately detects bruises across all Fitzpatrick skin types",
                "Validate the platform in diverse clinical settings and populations",
                "Create equitable algorithms with minimal performance disparities",
                "Generate peer-reviewed publications and open-source resources",
                "Build a foundation for future funding and expansion"
            ],
            "roles_responsibilities": {category: self.team_structure[category] for category in self.team_structure},
            "communication_guidelines": [
                "Direct and respectful communication",
                "Regular structured meetings with clear agendas",
                "Written documentation of decisions and action items",
                "Inclusive language and accessibility",
                "Balance of synchronous and asynchronous communication"
            ],
            "decision_making": {
                "process": "Consensus-seeking with escalation path",
                "authority_matrix": {
                    "technical_decisions": "Technical team leads with PI consultation",
                    "clinical_decisions": "Clinical team leads with PI approval",
                    "resource_allocation": "PIs collectively",
                    "timeline_adjustments": "Postdoc with PI approval",
                    "publication_strategy": "PIs with team input"
                }
            },
            "conflict_resolution": [
                "Direct conversation between involved parties",
                "Facilitated discussion if needed",
                "Escalation to PIs for unresolved issues",
                "Focus on issues not personalities",
                "Documentation of resolution and learning"
            ],
            "success_criteria": [
                "Technical performance metrics across skin tones",
                "Clinical validation results",
                "Publication outcomes",
                "User satisfaction metrics",
                "Team collaboration effectiveness"
            ]
        }
        
        return charter
    
    def generate_mentorship_plan(self, student_type):
        """
        Generate a mentorship plan for a specific type of student
        
        Parameters:
        - student_type: Type of student (CS, Nursing, Engineering)
        
        Returns:
        - plan: Dictionary with mentorship plan details
        """
        common_elements = {
            "regular_meetings": [
                {"frequency": "Weekly", "duration": "30 minutes", "focus": "Research progress"},
                {"frequency": "Monthly", "duration": "1 hour", "focus": "Career development"}
            ],
            "feedback_mechanism": [
                "Regular written feedback on deliverables",
                "Mid-term and end-term evaluations",
                "Bi-directional feedback sessions"
            ],
            "publication_opportunities": [
                "Co-authorship on project papers",
                "Conference presentation opportunities",
                "Poster sessions at university events"
            ],
            "professional_development": [
                "Presentation skills development",
                "Networking opportunities with collaborators",
                "Grant writing exposure",
                "CV/resume development"
            ]
        }
        
        if student_type.lower() == "cs":
            plan = {
                "title": "Computer Science Graduate Student Mentorship Plan",
                "technical_skills_development": [
                    "Deep learning model development",
                    "Computer vision techniques",
                    "Software engineering best practices",
                    "Performance evaluation and optimization"
                ],
                "research_skills": [
                    "Algorithm development methodology",
                    "Experimental design for AI systems",
                    "Scientific paper writing for technical audiences",
                    "Literature review in machine learning"
                ],
                "interdisciplinary_exposure": [
                    "Clinical shadowing opportunity",
                    "Participation in clinical team meetings",
                    "Forensic nursing basics workshop",
                    "Healthcare IT standards introduction"
                ],
                "suggested_deliverables": [
                    "Deep learning model implementation",
                    "Technical documentation",
                    "Evaluation report",
                    "Conference paper draft"
                ],
                "growth_trajectory": [
                    "Initial focus on implementation",
                    "Progress to algorithm design",
                    "Advance to research direction setting",
                    "Culminate in independent research components"
                ]
            }
        elif student_type.lower() == "nursing":
            plan = {
                "title": "Nursing Graduate Student Mentorship Plan",
                "technical_skills_development": [
                    "Basic programming concepts",
                    "Data analysis for healthcare",
                    "Mobile application usage and testing",
                    "AI concepts for healthcare professionals"
                ],
                "research_skills": [
                    "Clinical validation methodology",
                    "User testing protocols",
                    "Healthcare technology evaluation",
                    "Qualitative research methods"
                ],
                "interdisciplinary_exposure": [
                    "AI basics workshop",
                    "Participation in technical team meetings",
                    "Software development lifecycle introduction",
                    "Data visualization techniques"
                ],
                "suggested_deliverables": [
                    "Clinical validation protocol",
                    "User feedback analysis",
                    "Workflow integration report",
                    "Practice guideline draft"
                ],
                "growth_trajectory": [
                    "Initial focus on clinical requirements",
                    "Progress to validation design",
                    "Advance to clinical implementation planning",
                    "Culminate in practice guideline development"
                ]
            }
        elif student_type.lower() == "engineering":
            plan = {
                "title": "Engineering Graduate Student Mentorship Plan",
                "technical_skills_development": [
                    "Imaging system integration",
                    "Mobile hardware optimization",
                    "Embedded systems programming",
                    "Testing and validation methodology"
                ],
                "research_skills": [
                    "System design methodology",
                    "Hardware-software integration",
                    "Performance testing protocols",
                    "Technical documentation"
                ],
                "interdisciplinary_exposure": [
                    "Clinical shadowing opportunity",
                    "AI basics workshop",
                    "Forensic imaging introduction",
                    "Healthcare device regulations"
                ],
                "suggested_deliverables": [
                    "Hardware integration implementation",
                    "System performance analysis",
                    "Technical specifications document",
                    "Prototype demonstration"
                ],
                "growth_trajectory": [
                    "Initial focus on system components",
                    "Progress to integration challenges",
                    "Advance to system architecture",
                    "Culminate in full prototype development"
                ]
            }
        else:
            plan = {
                "title": "Graduate Student Mentorship Plan",
                "technical_skills_development": [
                    "Project-specific technical skills",
                    "General research methods",
                    "Data analysis techniques",
                    "Scientific communication"
                ],
                "research_skills": [
                    "Literature review",
                    "Experimental design",
                    "Data collection and analysis",
                    "Scientific writing"
                ],
                "interdisciplinary_exposure": [
                    "Cross-disciplinary workshops",
                    "Participation in diverse team meetings",
                    "Domain knowledge seminars",
                    "Collaborative projects"
                ],
                "suggested_deliverables": [
                    "Research components",
                    "Documentation",
                    "Analysis reports",
                    "Publication drafts"
                ],
                "growth_trajectory": [
                    "Initial focus on core skills",
                    "Progress to independent work",
                    "Advance to collaborative projects",
                    "Culminate in research leadership"
                ]
            }
        
        # Add common elements
        for key, value in common_elements.items():
            plan[key] = value
        
        return plan
    
    def generate_communication_matrix(self):
        """
        Generate a communication matrix for team interactions
        """
        team_groups = list(self.team_structure.keys())
        
        # Define meeting frequency between different team groups
        meeting_frequency = {
            ("leadership", "leadership"): "Weekly",
            ("leadership", "technical_team"): "Weekly",
            ("leadership", "clinical_team"): "Weekly",
            ("leadership", "students"): "Weekly",
            ("technical_team", "technical_team"): "Daily",
            ("technical_team", "clinical_team"): "Weekly",
            ("technical_team", "students"): "Daily",
            ("clinical_team", "clinical_team"): "Weekly",
            ("clinical_team", "students"): "Weekly",
            ("students", "students"): "Daily"
        }
        
        # Define communication channels between different team groups
        communication_channels = {
            ("leadership", "leadership"): ["Meetings", "Email", "Project management tool", "Messaging"],
            ("leadership", "technical_team"): ["Meetings", "Project management tool", "Messaging", "Documentation"],
            ("leadership", "clinical_team"): ["Meetings", "Email", "Messaging"],
            ("leadership", "students"): ["Meetings", "Mentoring sessions", "Project management tool"],
            ("technical_team", "technical_team"): ["Stand-ups", "Code review", "Messaging", "Documentation"],
            ("technical_team", "clinical_team"): ["Meetings", "Demonstrations", "Requirements sessions"],
            ("technical_team", "students"): ["Pair programming", "Code review", "Mentoring sessions"],
            ("clinical_team", "clinical_team"): ["Meetings", "Email", "Clinical documentation"],
            ("clinical_team", "students"): ["Shadowing", "Meetings", "Email"],
            ("students", "students"): ["Peer sessions", "Messaging", "Collaborative workspace"]
        }
        
        # Define information types shared between different team groups
        information_types = {
            ("leadership", "leadership"): ["Strategic decisions", "Resource allocation", "Risk management", "External communications"],
            ("leadership", "technical_team"): ["Project requirements", "Technical direction", "Timeline management", "Resource allocation"],
            ("leadership", "clinical_team"): ["Clinical requirements", "Validation protocols", "IRB matters", "Stakeholder feedback"],
            ("leadership", "students"): ["Project scope", "Research direction", "Mentoring feedback", "Career development"],
            ("technical_team", "technical_team"): ["Technical specifications", "Code standards", "Architecture decisions", "Implementation challenges"],
            ("technical_team", "clinical_team"): ["Technical capabilities", "Clinical workflow integration", "User interface design", "Validation results"],
            ("technical_team", "students"): ["Technical guidance", "Code review feedback", "Development practices", "Technical skills"],
            ("clinical_team", "clinical_team"): ["Clinical protocols", "User testing feedback", "Workflow optimization", "Documentation standards"],
            ("clinical_team", "students"): ["Clinical context", "User requirements", "Testing protocols", "Domain knowledge"],
            ("students", "students"): ["Technical assistance", "Research collaboration", "Learning resources", "Peer feedback"]
        }
        
        # Construct communication matrix
        matrix = []
        
        # Add all group-to-group combinations
        for i, group1 in enumerate(team_groups):
            for j, group2 in enumerate(team_groups):
                if i <= j:  # Include diagonal and upper triangle
                    key = (group1, group2)
                    reverse_key = (group2, group1)
                    
                    # Check if direct or reverse key exists
                    freq = meeting_frequency.get(key, meeting_frequency.get(reverse_key, "As needed"))
                    channels = communication_channels.get(key, communication_channels.get(reverse_key, ["Email", "Messaging"]))
                    info = information_types.get(key, information_types.get(reverse_key, ["Project information"]))
                    
                    matrix.append({
                        "from_group": group1,
                        "to_group": group2,
                        "frequency": freq,
                        "channels": channels,
                        "information_types": info
                    })
        
        return matrix
    
    def get_onboarding_process(self):
        """
        Get onboarding process for new team members
        """
        onboarding = {
            "pre_arrival": [
                {
                    "step": "Send welcome package",
                    "responsibility": "Postdoc",
                    "timeline": "1 week before start",
                    "description": "Email welcome letter, team charter, project overview, and initial readings"
                },
                {
                    "step": "Setup accounts and access",
                    "responsibility": "Technical lead",
                    "timeline": "3 days before start",
                    "description": "Create email, project management, code repository, and documentation access"
                },
                {
                    "step": "Schedule first week",
                    "responsibility": "Postdoc",
                    "timeline": "2 days before start",
                    "description": "Create calendar invites for orientation, introductions, and initial meetings"
                }
            ],
            "first_day": [
                {
                    "step": "Welcome meeting",
                    "responsibility": "PI or Postdoc",
                    "timeline": "9:00 AM",
                    "description": "Introduction to project, team structure, and expectations"
                },
                {
                    "step": "Administrative setup",
                    "responsibility": "Administrative support",
                    "timeline": "10:30 AM",
                    "description": "Complete necessary paperwork, badge access, and university requirements"
                },
                {
                    "step": "Team lunch",
                    "responsibility": "Whole team",
                    "timeline": "12:00 PM",
                    "description": "Informal team lunch for relationship building"
                },
                {
                    "step": "Workspace setup",
                    "responsibility": "Technical lead",
                    "timeline": "2:00 PM",
                    "description": "Set up computer, development environment, and project access"
                },
                {
                    "step": "First day debrief",
                    "responsibility": "Mentor",
                    "timeline": "4:00 PM",
                    "description": "Review first day experience, answer questions, and prepare for day two"
                }
            ],
            "first_week": [
                {
                    "step": "Project deep dive",
                    "responsibility": "Postdoc",
                    "timeline": "Day 2 morning",
                    "description": "Detailed overview of project goals, architecture, and current status"
                },
                {
                    "step": "Team introductions",
                    "responsibility": "New member",
                    "timeline": "Throughout week",
                    "description": "One-on-one meetings with each team member to understand roles and build relationships"
                },
                {
                    "step": "Technical onboarding",
                    "responsibility": "Technical lead",
                    "timeline": "Day 2-3",
                    "description": "Walkthrough of codebase, development practices, and technical documentation"
                },
                {
                    "step": "Clinical context session",
                    "responsibility": "Clinical lead",
                    "timeline": "Day 4",
                    "description": "Introduction to clinical aspects, terminology, and requirements"
                },
                {
                    "step": "First task assignment",
                    "responsibility": "Mentor",
                    "timeline": "Day 5",
                    "description": "Assign first small task with clear scope and support"
                }
            ],
            "first_month": [
                {
                    "step": "30-day check-in",
                    "responsibility": "Postdoc and PI",
                    "timeline": "End of month",
                    "description": "Review first month experience, address concerns, and set goals for next period"
                },
                {
                    "step": "Training completion",
                    "responsibility": "New member",
                    "timeline": "Throughout month",
                    "description": "Complete required training modules (research ethics, data security, etc.)"
                },
                {
                    "step": "Project contribution",
                    "responsibility": "New member",
                    "timeline": "Weeks 2-4",
                    "description": "Complete first meaningful contribution to project"
                },
                {
                    "step": "Team presentation",
                    "responsibility": "New member",
                    "timeline": "Week 4",
                    "description": "Brief presentation on background, interests, and initial project observations"
                }
            ],
            "resources": [
                {
                    "type": "Documentation",
                    "items": [
                        "Project charter",
                        "Technical documentation",
                        "Clinical protocols",
                        "Research publications",
                        "Team structure and contacts"
                    ]
                },
                {
                    "type": "Training",
                    "items": [
                        "Research ethics and compliance",
                        "Data security and privacy",
                        "Technical tools and platforms",
                        "Clinical terminology",
                        "Project management system"
                    ]
                },
                {
                    "type": "Support",
                    "items": [
                        "Assigned mentor",
                        "Technical buddy",
                        "Clinical buddy",
                        "Administrative contact",
                        "PI open door policy"
                    ]
                }
            ]
        }
        
        return onboarding
    
    def get_publication_plan(self):
        """
        Get publication plan for the project
        """
        today = datetime.now()
        start_date = today - timedelta(days=60)  # Project started 2 months ago
        
        publication_plan = {
            "strategy": {
                "focus_areas": [
                    "Algorithmic innovations in multi-spectral bruise detection",
                    "Fairness evaluation across skin tones",
                    "Clinical validation and workflow integration",
                    "Mobile healthcare technology for forensic applications",
                    "Interdisciplinary research methodology"
                ],
                "target_audiences": [
                    "Computer vision and AI researchers",
                    "Healthcare informatics community",
                    "Forensic nursing practitioners",
                    "Health equity researchers",
                    "Mobile health developers"
                ],
                "authorship_guidelines": [
                    "Substantial contribution to research, writing, or revision",
                    "Approval of final version",
                    "Agreement to be accountable for the work",
                    "Role-based author order with exceptions for student development",
                    "Acknowledgment of all contributors"
                ]
            },
            "planned_publications": [
                {
                    "title": "Multi-spectral Deep Learning for Equitable Bruise Detection: System Design and Early Results",
                    "target_venue": "IEEE Journal of Biomedical and Health Informatics",
                    "planned_submission": (start_date + timedelta(days=210)).strftime("%Y-%m-%d"),
                    "target_authors": ["PI", "Co-PI (Health Informatics)", "Co-PI (Engineering)", "Postdoc", "CS Student"],
                    "lead_author": "Postdoc",
                    "status": "Outline",
                    "key_innovations": [
                        "Multi-spectral image processing pipeline",
                        "Model architecture for low-contrast detection",
                        "Preliminary equity evaluation",
                        "System design overview"
                    ]
                },
                {
                    "title": "Clinical Validation of a Mobile Bruise Detection System Across Diverse Skin Tones",
                    "target_venue": "Journal of Forensic Nursing",
                    "planned_submission": (start_date + timedelta(days=270)).strftime("%Y-%m-%d"),
                    "target_authors": ["PI", "Forensic Nurse", "Postdoc", "Nursing Student", "UX Researcher"],
                    "lead_author": "PI",
                    "status": "Planning",
                    "key_innovations": [
                        "Clinical validation methodology",
                        "Performance across Fitzpatrick scale",
                        "Workflow integration findings",
                        "Documentation quality assessment"
                    ]
                },
                {
                    "title": "Fairness-Aware Bruise Detection: Metrics, Methods, and Evaluation",
                    "target_venue": "ACM Conference on Health, Inference, and Learning (CHIL)",
                    "planned_submission": (start_date + timedelta(days=180)).strftime("%Y-%m-%d"),
                    "target_authors": ["Postdoc", "Co-PI (Health Informatics)", "CS Student", "PI"],
                    "lead_author": "Postdoc",
                    "status": "Data Collection",
                    "key_innovations": [
                        "Fairness metrics for medical imaging",
                        "Bias mitigation techniques",
                        "Evaluation across demographic groups",
                        "Transparency and explainability"
                    ]
                },
                {
                    "title": "Mobile Hardware Integration for ALS Bruise Imaging: Technical Challenges and Solutions",
                    "target_venue": "IEEE Conference on Mobile Health (MobiHealth)",
                    "planned_submission": (start_date + timedelta(days=240)).strftime("%Y-%m-%d"),
                    "target_authors": ["Co-PI (Engineering)", "Postdoc", "Engineering Student", "Mobile Developer"],
                    "lead_author": "Co-PI (Engineering)",
                    "status": "Planning",
                    "key_innovations": [
                        "Mobile ALS hardware design",
                        "Camera calibration techniques",
                        "Power and performance optimization",
                        "Field usability considerations"
                    ]
                },
                {
                    "title": "User-Centered Design of a Trauma-Informed Bruise Documentation System",
                    "target_venue": "International Conference on Pervasive Computing Technologies for Healthcare",
                    "planned_submission": (start_date + timedelta(days=300)).strftime("%Y-%m-%d"),
                    "target_authors": ["UX Researcher", "Forensic Nurse", "PI", "Postdoc", "Nursing Student"],
                    "lead_author": "UX Researcher",
                    "status": "Planning",
                    "key_innovations": [
                        "Trauma-informed design principles",
                        "User research with diverse stakeholders",
                        "Iterative testing methodology",
                        "Accessibility considerations"
                    ]
                }
            ],
            "conference_targets": [
                {
                    "name": "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
                    "deadline": (today + timedelta(days=120)).strftime("%Y-%m-%d"),
                    "focus": "Computer vision innovations",
                    "planned_submission": "No - too early for results"
                },
                {
                    "name": "ACM Conference on Health, Inference, and Learning (CHIL)",
                    "deadline": (today + timedelta(days=60)).strftime("%Y-%m-%d"),
                    "focus": "Healthcare AI and fairness",
                    "planned_submission": "Yes - abstract submitted"
                },
                {
                    "name": "International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)",
                    "deadline": (today + timedelta(days=90)).strftime("%Y-%m-%d"),
                    "focus": "Medical imaging and intervention",
                    "planned_submission": "Considering - pending preliminary results"
                },
                {
                    "name": "International Association of Forensic Nurses Annual Conference",
                    "deadline": (today + timedelta(days=150)).strftime("%Y-%m-%d"),
                    "focus": "Forensic nursing applications",
                    "planned_submission": "Yes - clinical validation results"
                },
                {
                    "name": "IEEE Conference on Mobile Health (MobiHealth)",
                    "deadline": (today + timedelta(days=180)).strftime("%Y-%m-%d"),
                    "focus": "Mobile health technology",
                    "planned_submission": "Yes - hardware integration paper"
                }
            ]
        }
        
        return publication_plan
    
    def get_conflict_resolution_plan(self):
        """
        Get comprehensive conflict resolution plan for team issues
        """
        resolution_plan = {
            "conflict_types": [
                {
                    "type": "Technical Disagreement",
                    "examples": [
                        "Choice of AI architecture",
                        "Technology stack selection",
                        "Performance vs. accuracy trade-offs"
                    ],
                    "resolution_steps": [
                        "Document all perspectives with technical justification",
                        "Conduct proof-of-concept testing where feasible",
                        "Present evidence-based arguments",
                        "Seek external technical expert opinion if needed",
                        "Make decision based on project requirements",
                        "Document decision and rationale for future reference"
                    ],
                    "prevention_strategies": [
                        "Clear technical requirements from project start",
                        "Regular architecture review meetings",
                        "Established technical decision-making process"
                    ]
                },
                {
                    "type": "Resource Allocation",
                    "examples": [
                        "Computing resources distribution",
                        "Budget allocation between teams",
                        "Time allocation for different tasks"
                    ],
                    "resolution_steps": [
                        "Clearly articulate resource needs with justification",
                        "Map resources to project priorities",
                        "Consider compromise solutions",
                        "Document resource allocation decisions",
                        "Review allocation effectiveness periodically"
                    ],
                    "prevention_strategies": [
                        "Transparent resource planning",
                        "Regular resource review meetings",
                        "Clear priority framework"
                    ]
                },
                {
                    "type": "Authorship and Credit",
                    "examples": [
                        "Paper authorship order",
                        "Contribution acknowledgment",
                        "Presentation opportunities"
                    ],
                    "resolution_steps": [
                        "Review contribution documentation",
                        "Apply established authorship guidelines",
                        "Consider discipline-specific norms",
                        "Facilitate discussion between parties",
                        "Document final agreements"
                    ],
                    "prevention_strategies": [
                        "Clear authorship policy from start",
                        "Regular contribution tracking",
                        "Transparent decision processes"
                    ]
                },
                {
                    "type": "Interpersonal Conflicts",
                    "examples": [
                        "Communication style differences",
                        "Cultural misunderstandings",
                        "Personal disagreements"
                    ],
                    "resolution_steps": [
                        "Private initial discussion with each party",
                        "Identify underlying issues",
                        "Facilitate structured dialogue",
                        "Focus on professional behavior",
                        "Monitor progress and follow up",
                        "Escalate to HR if necessary"
                    ],
                    "prevention_strategies": [
                        "Team building activities",
                        "Clear communication guidelines",
                        "Cultural sensitivity training"
                    ]
                },
                {
                    "type": "Work-Life Balance",
                    "examples": [
                        "Excessive workload concerns",
                        "Meeting scheduling conflicts",
                        "Deadline pressures"
                    ],
                    "resolution_steps": [
                        "Assess current workload distribution",
                        "Identify critical vs. optional tasks",
                        "Adjust timelines where possible",
                        "Implement flexible working arrangements",
                        "Regular check-ins on team wellbeing"
                    ],
                    "prevention_strategies": [
                        "Realistic project planning",
                        "Regular workload assessments",
                        "Flexible work policies"
                    ]
                }
            ],
            "escalation_path": [
                {
                    "level": 1,
                    "resolver": "Direct Discussion",
                    "scope": "Between involved parties",
                    "timeline": "Within 1 week"
                },
                {
                    "level": 2,
                    "resolver": "Team Lead/Postdoc",
                    "scope": "Mediated discussion",
                    "timeline": "Within 2 weeks"
                },
                {
                    "level": 3,
                    "resolver": "Principal Investigators",
                    "scope": "Formal review and decision",
                    "timeline": "Within 3 weeks"
                },
                {
                    "level": 4,
                    "resolver": "Department Chair/HR",
                    "scope": "Official institutional process",
                    "timeline": "As per institutional policy"
                }
            ],
            "documentation_requirements": [
                "Written summary of conflict",
                "Attempted resolution steps",
                "Participant perspectives",
                "Final resolution and agreements",
                "Follow-up actions and timeline"
            ],
            "resolution_principles": [
                "Focus on project goals and team success",
                "Maintain professional respect",
                "Seek win-win solutions",
                "Document decisions for clarity",
                "Learn from conflicts to prevent recurrence",
                "Preserve team cohesion and morale"
            ]
        }
        
        return resolution_plan
    
    def get_risk_management_plan(self):
        """
        Get comprehensive risk management plan for the project
        """
        risk_plan = {
            "risk_categories": [
                {
                    "category": "Technical Risks",
                    "risks": [
                        {
                            "risk": "AI model performance below requirements",
                            "impact": "High",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Iterative model development with regular benchmarking",
                                "Multiple model architectures in parallel",
                                "External expert consultation",
                                "Fallback to proven approaches"
                            ],
                            "contingency_plan": "Extend timeline for model development, seek additional expertise"
                        },
                        {
                            "risk": "Integration challenges between components",
                            "impact": "Medium",
                            "likelihood": "High",
                            "mitigation_strategies": [
                                "Early integration testing",
                                "Well-defined APIs",
                                "Regular system integration meetings",
                                "Modular architecture design"
                            ],
                            "contingency_plan": "Allocate additional integration sprint, simplify architecture if needed"
                        },
                        {
                            "risk": "Performance issues on mobile devices",
                            "impact": "High",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Early device testing",
                                "Progressive optimization approach",
                                "Multiple model sizes",
                                "Hardware acceleration utilization"
                            ],
                            "contingency_plan": "Implement cloud-based processing option, optimize model further"
                        }
                    ]
                },
                {
                    "category": "Clinical Risks",
                    "risks": [
                        {
                            "risk": "Low clinical adoption",
                            "impact": "High",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Early clinician involvement",
                                "User-centered design process",
                                "Pilot testing in clinical settings",
                                "Comprehensive training materials"
                            ],
                            "contingency_plan": "Redesign interface based on feedback, increase training efforts"
                        },
                        {
                            "risk": "Regulatory compliance issues",
                            "impact": "High",
                            "likelihood": "Low",
                            "mitigation_strategies": [
                                "Early regulatory consultation",
                                "Compliance-first design approach",
                                "Regular compliance audits",
                                "Expert legal consultation"
                            ],
                            "contingency_plan": "Adjust system design for compliance, seek regulatory guidance"
                        }
                    ]
                },
                {
                    "category": "Research Risks",
                    "risks": [
                        {
                            "risk": "Insufficient data diversity",
                            "impact": "High",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Multiple data collection sites",
                                "Targeted recruitment strategies",
                                "Data augmentation techniques",
                                "Synthetic data generation"
                            ],
                            "contingency_plan": "Extend data collection period, partner with additional institutions"
                        },
                        {
                            "risk": "Publication delays",
                            "impact": "Medium",
                            "likelihood": "High",
                            "mitigation_strategies": [
                                "Parallel paper preparation",
                                "Regular writing workshops",
                                "Clear authorship agreements",
                                "Multiple publication venues"
                            ],
                            "contingency_plan": "Adjust publication strategy, consider preprint servers"
                        }
                    ]
                },
                {
                    "category": "Team Risks",
                    "risks": [
                        {
                            "risk": "Key team member departure",
                            "impact": "High",
                            "likelihood": "Low",
                            "mitigation_strategies": [
                                "Knowledge documentation",
                                "Cross-training initiatives",
                                "Competitive retention packages",
                                "Succession planning"
                            ],
                            "contingency_plan": "Rapid knowledge transfer, temporary consultant hiring"
                        },
                        {
                            "risk": "Team burnout",
                            "impact": "Medium",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Regular workload monitoring",
                                "Flexible work arrangements",
                                "Mental health resources",
                                "Team building activities"
                            ],
                            "contingency_plan": "Temporary workload reduction, additional resources allocation"
                        }
                    ]
                },
                {
                    "category": "External Risks",
                    "risks": [
                        {
                            "risk": "Funding interruption",
                            "impact": "High",
                            "likelihood": "Low",
                            "mitigation_strategies": [
                                "Diversified funding sources",
                                "Regular grant applications",
                                "Budget reserves",
                                "Industry partnerships"
                            ],
                            "contingency_plan": "Prioritize critical activities, seek emergency funding"
                        },
                        {
                            "risk": "Competing technology release",
                            "impact": "Medium",
                            "likelihood": "Medium",
                            "mitigation_strategies": [
                                "Regular competitive analysis",
                                "Unique value proposition",
                                "Rapid development cycle",
                                "Strong IP protection"
                            ],
                            "contingency_plan": "Differentiation strategy, accelerate unique features"
                        }
                    ]
                }
            ],
            "risk_assessment_matrix": {
                "description": "5x5 risk matrix (Impact x Likelihood)",
                "impact_levels": ["Negligible", "Minor", "Moderate", "Major", "Catastrophic"],
                "likelihood_levels": ["Rare", "Unlikely", "Possible", "Likely", "Almost Certain"],
                "action_thresholds": {
                    "low": "Monitor and review periodically",
                    "medium": "Active mitigation required",
                    "high": "Immediate action and contingency planning",
                    "critical": "Executive attention and resource allocation"
                }
            },
            "monitoring_process": [
                {
                    "frequency": "Weekly",
                    "scope": "Technical and operational risks",
                    "responsible": "Project Manager/Postdoc",
                    "actions": "Update risk register, implement mitigations"
                },
                {
                    "frequency": "Monthly",
                    "scope": "All risk categories",
                    "responsible": "Leadership Team",
                    "actions": "Review risk landscape, adjust strategies"
                },
                {
                    "frequency": "Quarterly",
                    "scope": "Strategic risks",
                    "responsible": "Principal Investigators",
                    "actions": "Strategic risk assessment, resource allocation"
                }
            ],
            "risk_communication_plan": [
                "Regular risk updates in team meetings",
                "Monthly risk dashboard for stakeholders",
                "Immediate escalation for critical risks",
                "Quarterly risk report for funders",
                "Post-incident reviews for lessons learned"
            ]
        }
        
        return risk_plan
    
    def calculate_team_metrics(self):
        """
        Calculate various team performance and collaboration metrics
        """
        today = datetime.now()
        
        metrics = {
            "team_size": {
                "total": sum(len(members) for members in self.team_structure.values()),
                "by_category": {category: len(members) for category, members in self.team_structure.items()},
                "by_discipline": self.get_team_composition_summary()
            },
            "meeting_metrics": {
                "total_scheduled": len(self.meeting_schedule),
                "by_type": {},
                "total_hours_per_week": 0,
                "participation_load": {}
            },
            "milestone_metrics": {
                "total": len(self.project_milestones),
                "completed": len([m for m in self.project_milestones if m["status"] == "Completed"]),
                "in_progress": len([m for m in self.project_milestones if m["status"] == "In Progress"]),
                "planned": len([m for m in self.project_milestones if m["status"] == "Planned"]),
                "completion_rate": 0,
                "upcoming_30_days": len(self.get_upcoming_milestones(30)),
                "overdue": 0
            },
            "collaboration_intensity": {
                "cross_team_meetings": 0,
                "interdisciplinary_interactions": 0
            },
            "workload_distribution": {},
            "publication_metrics": {}
        }
        
        # Calculate meeting metrics
        meeting_types = {}
        for meeting in self.meeting_schedule:
            meeting_type = meeting["frequency"]
            if meeting_type not in meeting_types:
                meeting_types[meeting_type] = 0
            meeting_types[meeting_type] += 1
            
            # Extract duration (assuming format "HH:MM - HH:MM")
            time_parts = meeting["time"].split(" - ")
            if len(time_parts) == 2:
                start_time = datetime.strptime(time_parts[0], "%H:%M")
                end_time = datetime.strptime(time_parts[1], "%H:%M")
                duration_hours = (end_time - start_time).seconds / 3600
                
                if meeting["frequency"] == "Weekly":
                    metrics["meeting_metrics"]["total_hours_per_week"] += duration_hours
                elif meeting["frequency"] == "Bi-weekly":
                    metrics["meeting_metrics"]["total_hours_per_week"] += duration_hours / 2
                elif meeting["frequency"] == "Monthly":
                    metrics["meeting_metrics"]["total_hours_per_week"] += duration_hours / 4
        
        metrics["meeting_metrics"]["by_type"] = meeting_types
        
        # Calculate milestone metrics
        completed = metrics["milestone_metrics"]["completed"]
        total = metrics["milestone_metrics"]["total"]
        metrics["milestone_metrics"]["completion_rate"] = round(completed / total * 100, 1) if total > 0 else 0
        
        # Count overdue milestones
        for milestone in self.project_milestones:
            if milestone["status"] != "Completed":
                milestone_date = datetime.strptime(milestone["date"], "%Y-%m-%d")
                if milestone_date < today:
                    metrics["milestone_metrics"]["overdue"] += 1
        
        # Calculate collaboration intensity
        comm_matrix = self.generate_communication_matrix()
        for entry in comm_matrix:
            if entry["from_group"] != entry["to_group"]:
                metrics["collaboration_intensity"]["cross_team_meetings"] += 1
                if entry["frequency"] in ["Daily", "Weekly"]:
                    metrics["collaboration_intensity"]["interdisciplinary_interactions"] += 1
        
        # Calculate workload distribution (simplified)
        team_categories = list(self.team_structure.keys())
        for category in team_categories:
            # Estimate based on meeting participation and milestones
            workload_score = 0
            
            # Meeting participation
            for meeting in self.meeting_schedule:
                if category.lower() in meeting["participants"].lower() or "all" in meeting["participants"].lower():
                    if meeting["frequency"] == "Weekly":
                        workload_score += 4
                    elif meeting["frequency"] == "Bi-weekly":
                        workload_score += 2
                    elif meeting["frequency"] == "Monthly":
                        workload_score += 1
            
            # Milestone involvement (simplified)
            for milestone in self.project_milestones:
                if category.lower() in milestone["description"].lower():
                    workload_score += 5
            
            metrics["workload_distribution"][category] = workload_score
        
        # Publication metrics
        pub_plan = self.get_publication_plan()
        if "planned_publications" in pub_plan:
            metrics["publication_metrics"] = {
                "total_planned": len(pub_plan["planned_publications"]),
                "by_status": {},
                "by_venue_type": {"journal": 0, "conference": 0},
                "lead_author_distribution": {}
            }
            
            for pub in pub_plan["planned_publications"]:
                status = pub.get("status", "Unknown")
                if status not in metrics["publication_metrics"]["by_status"]:
                    metrics["publication_metrics"]["by_status"][status] = 0
                metrics["publication_metrics"]["by_status"][status] += 1
                
                if "Journal" in pub.get("target_venue", ""):
                    metrics["publication_metrics"]["by_venue_type"]["journal"] += 1
                else:
                    metrics["publication_metrics"]["by_venue_type"]["conference"] += 1
                
                lead = pub.get("lead_author", "Unknown")
                if lead not in metrics["publication_metrics"]["lead_author_distribution"]:
                    metrics["publication_metrics"]["lead_author_distribution"][lead] = 0
                metrics["publication_metrics"]["lead_author_distribution"][lead] += 1
        
        return metrics
    
    def generate_progress_report(self):
        """
        Generate a comprehensive progress report for stakeholders
        """
        today = datetime.now()
        start_date = today - timedelta(days=60)  # Assuming project started 2 months ago
        
        # Calculate metrics
        metrics = self.calculate_team_metrics()
        
        report = {
            "report_metadata": {
                "title": "EAS-ID Project Progress Report",
                "date": today.strftime("%Y-%m-%d"),
                "period": f"{start_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}",
                "prepared_by": "Project Leadership Team"
            },
            "executive_summary": {
                "project_status": "On Track",  # This would be calculated based on metrics
                "key_achievements": [
                    "Completed initial data collection protocol",
                    "Established interdisciplinary team of 12+ members",
                    "Achieved baseline model performance benchmarks",
                    "Initiated clinical validation planning"
                ],
                "major_challenges": [
                    "Integration complexity between mobile and AI components",
                    "Data diversity requirements for equitable performance",
                    "Balancing clinical requirements with technical innovation"
                ],
                "upcoming_milestones": [
                    f"{m['name']} ({m['days_until']} days)" 
                    for m in self.get_upcoming_milestones(90)[:3]
                ]
            },
            "team_status": {
                "composition": metrics["team_size"],
                "recent_additions": [],  # Would be tracked in real system
                "training_completed": ["IRB training", "HIPAA compliance", "Git workflow"],
                "team_satisfaction": "High (based on recent survey)"  # Would be from actual survey
            },
            "technical_progress": {
                "development_status": {
                    "deep_learning_model": "80% complete",
                    "mobile_application": "60% complete",
                    "database_implementation": "40% complete",
                    "system_integration": "30% complete"
                },
                "performance_metrics": {
                    "model_accuracy": {
                        "fitzpatrick_1_2": "92%",
                        "fitzpatrick_3_4": "88%",
                        "fitzpatrick_5_6": "82%"
                    },
                    "processing_time": "250ms per image",
                    "mobile_app_size": "45MB"
                },
                "technical_risks": [
                    "Model optimization for mobile deployment",
                    "Real-time performance requirements",
                    "Cross-platform compatibility"
                ]
            },
            "clinical_progress": {
                "protocol_development": "Complete",
                "irb_status": "Approved",
                "recruitment_status": {
                    "target": 500,
                    "enrolled": 0,  # Would be actual number
                    "demographics": "Planning diverse recruitment"
                },
                "clinical_sites": ["GMU Health Center", "Partner Hospital A", "Community Clinic B"],
                "validation_readiness": "Preparing for pilot study"
            },
            "milestone_summary": {
                "overall_progress": f"{metrics['milestone_metrics']['completion_rate']}%",
                "completed": metrics["milestone_metrics"]["completed"],
                "in_progress": metrics["milestone_metrics"]["in_progress"],
                "upcoming": metrics["milestone_metrics"]["upcoming_30_days"],
                "at_risk": metrics["milestone_metrics"]["overdue"]
            },
            "resource_utilization": {
                "budget_status": {
                    "allocated": "$500,000",
                    "spent": "$125,000",
                    "committed": "$75,000",
                    "available": "$300,000"
                },
                "compute_resources": {
                    "gpu_hours_used": 1200,
                    "storage_used_gb": 850,
                    "monthly_cloud_costs": "$3,500"
                },
                "human_resources": {
                    "total_fte": 8.5,
                    "by_category": {
                        "research": 4.0,
                        "development": 3.0,
                        "clinical": 1.5
                    }
                }
            },
            "collaboration_metrics": metrics["collaboration_intensity"],
            "publication_status": {
                "papers_planned": metrics["publication_metrics"]["total_planned"],
                "papers_submitted": 0,  # Would be tracked
                "papers_accepted": 0,
                "presentations_given": 2,
                "upcoming_submissions": [
                    pub["title"][:50] + "..." 
                    for pub in self.get_publication_plan()["planned_publications"][:2]
                ]
            },
            "stakeholder_engagement": {
                "funder_meetings": 2,
                "advisory_board_meetings": 1,
                "community_presentations": 3,
                "media_coverage": ["University news feature", "Local health magazine"]
            },
            "risks_and_mitigations": {
                "high_priority_risks": [
                    {
                        "risk": "AI model performance on darker skin tones",
                        "mitigation": "Focused data collection and model optimization",
                        "status": "Actively managing"
                    },
                    {
                        "risk": "Clinical adoption barriers",
                        "mitigation": "User-centered design and extensive training",
                        "status": "Preventive measures in place"
                    }
                ],
                "risk_summary": "3 high, 5 medium, 8 low priority risks identified"
            },
            "next_period_priorities": [
                "Complete mobile app prototype",
                "Achieve target model performance across all skin tones",
                "Begin pilot clinical validation",
                "Submit first conference paper",
                "Expand team with UX researcher"
            ],
            "recommendations": [
                "Allocate additional resources to model optimization",
                "Accelerate clinical site partnerships",
                "Consider early industry engagement for deployment planning",
                "Plan team retreat for strategic alignment"
            ],
            "appendices": {
                "detailed_milestone_gantt": "See attached visualization",
                "full_risk_register": "Available upon request",
                "team_roster": "See attached document",
                "technical_specifications": "Available in project repository"
            }
        }
        
        return report
    
    def get_team_development_activities(self):
        """
        Get team development and training activities
        """
        activities = {
            "scheduled_trainings": [
                {
                    "title": "Deep Learning for Healthcare Applications",
                    "date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                    "duration": "4 hours",
                    "target_audience": ["Technical team", "Students"],
                    "instructor": "External expert",
                    "objectives": [
                        "Understand healthcare-specific ML challenges",
                        "Learn about fairness in medical AI",
                        "Hands-on with medical imaging models"
                    ]
                },
                {
                    "title": "Forensic Photography Techniques",
                    "date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"),
                    "duration": "3 hours",
                    "target_audience": ["Clinical team", "Technical team"],
                    "instructor": "Forensic nurse specialist",
                    "objectives": [
                        "Understand bruise documentation standards",
                        "Learn about lighting and photography",
                        "Practice with ALS techniques"
                    ]
                },
                {
                    "title": "FHIR Standards and Healthcare Interoperability",
                    "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "duration": "2 hours",
                    "target_audience": ["Technical team", "Database specialist"],
                    "instructor": "Healthcare IT consultant",
                    "objectives": [
                        "Understand FHIR resource model",
                        "Learn about healthcare data standards",
                        "Implementation best practices"
                    ]
                }
            ],
            "team_building_events": [
                {
                    "event": "Monthly Team Lunch",
                    "frequency": "Monthly",
                    "next_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "purpose": "Informal networking and relationship building"
                },
                {
                    "event": "Quarterly Team Retreat",
                    "frequency": "Quarterly",
                    "next_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                    "purpose": "Strategic planning and team cohesion"
                },
                {
                    "event": "Innovation Workshop",
                    "frequency": "Bi-monthly",
                    "next_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
                    "purpose": "Creative problem solving and ideation"
                }
            ],
            "skill_development_programs": [
                {
                    "program": "Technical Writing for Researchers",
                    "duration": "6 weeks",
                    "format": "Online self-paced",
                    "participants": ["Graduate students", "Postdoc"],
                    "skills_developed": [
                        "Scientific paper writing",
                        "Grant proposal writing",
                        "Technical documentation"
                    ]
                },
                {
                    "program": "Leadership Development for Scientists",
                    "duration": "3 months",
                    "format": "Hybrid workshop series",
                    "participants": ["Postdoc", "Senior team members"],
                    "skills_developed": [
                        "Team management",
                        "Conflict resolution",
                        "Strategic thinking"
                    ]
                },
                {
                    "program": "Cross-functional Collaboration",
                    "duration": "Ongoing",
                    "format": "Peer mentoring",
                    "participants": ["All team members"],
                    "skills_developed": [
                        "Interdisciplinary communication",
                        "Cultural competency",
                        "Collaborative problem solving"
                    ]
                }
            ],
            "knowledge_sharing_initiatives": [
                {
                    "initiative": "Weekly Tech Talks",
                    "description": "Team members present on their expertise area",
                    "schedule": "Every Friday 3-4 PM",
                    "recent_topics": [
                        "Introduction to Computer Vision",
                        "Forensic Nursing Practices",
                        "Mobile App Architecture"
                    ]
                },
                {
                    "initiative": "Journal Club",
                    "description": "Review and discuss relevant research papers",
                    "schedule": "Bi-weekly Wednesdays",
                    "recent_papers": [
                        "Fairness in Medical AI Systems",
                        "Bruise Dating Techniques",
                        "Mobile Health Interventions"
                    ]
                },
                {
                    "initiative": "Documentation Days",
                    "description": "Dedicated time for knowledge documentation",
                    "schedule": "First Monday of each month",
                    "outputs": [
                        "Technical wikis",
                        "Process documents",
                        "Lesson learned reports"
                    ]
                }
            ],
            "professional_development_support": {
                "conference_attendance": {
                    "budget": "$20,000 annually",
                    "supported_conferences": [
                        "Computer Vision conferences",
                        "Healthcare informatics conferences", 
                        "Forensic nursing conferences"
                    ],
                    "requirements": "Present research or bring back learnings"
                },
                "certification_support": {
                    "budget": "$5,000 annually",
                    "supported_certifications": [
                        "Cloud platform certifications",
                        "Healthcare IT certifications",
                        "Project management certifications"
                    ]
                },
                "continuing_education": {
                    "budget": "$10,000 annually",
                    "supported_programs": [
                        "Online courses",
                        "Workshop attendance",
                        "Degree program support"
                    ]
                }
            }
        }
        
        return activities
