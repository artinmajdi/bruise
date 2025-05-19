
import streamlit as st

class LeadershipPanel:
	"""
	Handles the display and logic for the Leadership section.
	Focuses on coordinating a heterogeneous team.
	"""

	def display(self):
		"""Displays the leadership content."""
		st.header("🤝 Leadership & Teamwork: Coordinating a Heterogeneous Team")
		st.markdown("""
		**Challenge:** "Tell us about a time you coordinated a heterogeneous team." (Or, "How would you approach supervising students and clinicians as part of this project?")

		**What they're looking for:** Readiness to supervise students & clinicians, communication skills, project management awareness, ability to foster collaboration.
		The Postdoc is expected to supervise graduate students and programmers.
		""")
		st.markdown("---")

		st.subheader("The EAS-ID Team Context")
		st.markdown("""
		The EAS-ID project is inherently multidisciplinary, involving:
		- **Principal Investigators (PIs):** Experts in nursing, AI/health informatics, engineering.
		- **Postdoctoral Researcher (You):** Bridging deep learning engineering with clinical application, supervising.
		- **Graduate Students:** Likely from Computer Science, Engineering, Health Informatics.
		- **Programmers:** Potentially research staff or contractors.
		- **Clinicians (Nurses, Forensic Specialists):** End-users, domain experts, providing feedback and validation data.
		- **Potentially Undergrad Students:** Assisting with tasks.
		""")

		st.subheader("Strategies for Effective Coordination & Supervision")
		st.markdown("""
		Coordinating such a diverse team requires a proactive and adaptable leadership style. Here are key strategies:
		""")

		tabs = st.tabs([
			"Clear Communication",
			"Goal Setting & Task Delegation",
			"Fostering Collaboration",
			"Mentorship & Skill Development",
			"Conflict Resolution & Feedback"
		])

		with tabs[0]:
			st.markdown("#### 1. Establishing Clear and Consistent Communication")
			st.markdown("""
			- **Regular Team Meetings:**
				- **Frequency:** Weekly or bi-weekly, depending on project phase.
				- **Agenda:** Structured agenda covering progress, roadblocks, next steps, and a dedicated time for Q&A or open discussion.
				- **Format:** Mix of full team meetings and smaller subgroup meetings (e.g., tech team, clinical advisory).
			- **Multiple Channels:**
				- **Project Management Tools:** (e.g., Asana, Trello, Jira) for task tracking and progress visibility.
				- **Shared Documentation:** (e.g., Confluence, Google Workspace, GitHub Wiki) for protocols, design documents, meeting minutes.
				- **Communication Platforms:** (e.g., Slack, Microsoft Teams) for quick queries and informal discussions.
				- **Email:** For formal communication and external correspondence.
			- **Tailored Communication:** Adapt your communication style to the audience. Explain technical concepts clearly to clinicians, and clinical needs clearly to engineers/programmers. Avoid jargon where possible or explain it.
			- **Active Listening:** Encourage team members to share their perspectives and listen attentively to understand their challenges and ideas.
			""")

		with tabs[1]:
			st.markdown("#### 2. Clear Goal Setting and Task Delegation")
			st.markdown("""
			- **Shared Vision:** Ensure everyone understands the overall project goals (e.g., equitable bruise detection) and how their work contributes.
			- **SMART Goals:** Define Specific, Measurable, Achievable, Relevant, and Time-bound goals for individuals and sub-teams.
			- **Role Clarity:** Clearly define roles, responsibilities, and expectations for each team member, especially for students and programmers you supervise.
			- **Delegation based on Strengths & Development Needs:** Assign tasks that leverage individual strengths while also providing opportunities for growth. For students, this is particularly important.
			- **Prioritization:** Help the team prioritize tasks based on project milestones and dependencies.
			""")

		with tabs[2]:
			st.markdown("#### 3. Fostering a Collaborative and Inclusive Environment")
			st.markdown("""
			- **Encourage Cross-Disciplinary Interaction:** Create opportunities for engineers, students, and clinicians to interact and learn from each other (e.g., joint workshops, presentations where clinicians explain workflow challenges to tech team).
			- **Value Diverse Perspectives:** Actively solicit input from all team members. Recognize that clinicians bring invaluable real-world insights, and students may offer fresh perspectives.
			- **Psychological Safety:** Create an environment where team members feel safe to ask questions, admit mistakes, and propose unconventional ideas without fear of negative repercussions.
			- **Celebrate Successes:** Acknowledge individual and team achievements to build morale and reinforce positive collaboration.
			""")

		with tabs[3]:
			st.markdown("#### 4. Mentorship and Skill Development (Especially for Students/Programmers)")
			st.markdown("""
			- **Regular Check-ins:** One-on-one meetings with supervisees to discuss progress, challenges, and provide guidance.
			- **Constructive Feedback:** Offer specific, actionable, and timely feedback on their work. Balance positive reinforcement with areas for improvement.
			- **Skill Enhancement:** Identify learning opportunities, suggest relevant resources (papers, courses, workshops), and encourage them to develop new skills.
			- **Career Guidance (for students):** Offer advice on academic progress, research directions, and career paths.
			- **Empowerment:** Give students and programmers ownership of their tasks while providing necessary support and oversight. Avoid micromanagement but ensure they are on track.
			""")

		with tabs[4]:
			st.markdown("#### 5. Proactive Problem Solving and Conflict Resolution")
			st.markdown("""
			- **Identify Roadblocks Early:** Encourage open reporting of issues.
			- **Facilitate Solutions:** Work with the team to brainstorm and implement solutions to technical or logistical challenges.
			- **Address Conflicts Constructively:** If disagreements arise (e.g., different technical approaches, differing interpretations of clinical needs), address them promptly and professionally. Focus on understanding different viewpoints and finding common ground or a well-reasoned path forward.
			- **Regular Feedback Loops:** Not just top-down, but also solicit feedback on your own leadership and the team's processes.
			""")

		st.markdown("---")
		st.subheader("Example Scenario: Supervising a Graduate Student on Model Development")
		st.markdown("""
		*Imagine a graduate student is tasked with improving the bruise segmentation model. My approach would be:*

		1.  **Initial Planning:**
			* Meet to clearly define the specific objective (e.g., "Improve Dice score for faint bruises on Fitzpatrick V-VI skin tones by 10% within 3 months").
			* Discuss potential approaches (e.g., new architectures, loss functions, data augmentation strategies).
			* Break down the task into manageable sub-tasks with timelines.
			* Ensure they have access to necessary resources (data, compute, relevant papers).

		2.  **Ongoing Support & Mentorship:**
			* **Weekly 1-on-1s:** Review progress, discuss experimental results, troubleshoot issues (e.g., code bugs, unexpected model behavior).
			* **Code Reviews:** Provide feedback on their code quality, efficiency, and documentation.
			* **Guidance on Research:** Help them interpret results, suggest next experiments, and guide them in literature review.
			* **Encourage Independence:** While providing support, encourage them to take initiative and problem-solve independently.

		3.  **Integration with Team:**
			* Ensure the student presents their progress and findings to the wider tech team periodically.
			* Facilitate discussions between the student and clinicians if their work requires specific domain input (e.g., "Are these segmented regions clinically plausible?").

		4.  **Professional Development:**
			* Encourage them to write up their findings for potential publication or conference presentation.
			* Help them develop presentation skills.
		""")

		st.markdown("<div class='info-box'><h3>Key Takeaway:</h3>Effective leadership in this context means being a skilled communicator, a supportive mentor (especially for students), a good project manager, and a facilitator of interdisciplinary collaboration. It's about enabling each team member to contribute their best work towards the shared project goals, bridging the gap between technical development and clinical relevance.</div>", unsafe_allow_html=True)
