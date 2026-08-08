from typing import Dict, Any, Optional, List

class SessionState:
    MIN_QUESTIONS: int = 10
    MAX_QUESTIONS: int = 15

    def __init__(self, session_id: str, candidate: Dict[str, Any], profile_analysis: Dict[str, Any]):
        self.session_id: str = session_id
        self.candidate: Dict[str, Any] = candidate
        self.profile_analysis: Dict[str, Any] = profile_analysis
        self.question_number: int = 0
        self.extended_session: bool = False
        self.user_finished_at_10: bool = False
        self.custom_notes: str = ""
        
        self.questions_asked: List[Dict[str, Any]] = []
        self.answers: List[str] = []
        self.evaluations: List[Dict[str, Any]] = []
        self.covered_days: List[int] = []
        self.covered_topics: List[str] = []
        self.current_difficulty: str = "medium"
        self.status: str = "in_progress"  # "in_progress", "awaiting_choice", "completed"
        self.final_feedback: Optional[Dict[str, Any]] = None

    def update_profile_customization(
        self,
        name: Optional[str] = None,
        job_role: Optional[str] = None,
        years_experience: Optional[int] = None,
        difficulty_override: Optional[str] = None,
        custom_notes: Optional[str] = None
    ):
        """Allow candidate and session settings customization during interview."""
        member = self.candidate.get("member", {})
        if name:
            member["name"] = name
            self.profile_analysis["name"] = name
        if job_role:
            member["jobRole"] = job_role
            self.profile_analysis["jobRole"] = job_role
        if years_experience is not None:
            member["yearsExperience"] = years_experience
            self.profile_analysis["yearsExperience"] = years_experience
        if difficulty_override in ["easy", "medium", "hard"]:
            self.current_difficulty = difficulty_override
        if custom_notes is not None:
            self.custom_notes = custom_notes
            self.profile_analysis["custom_notes"] = custom_notes

    def add_question(self, question_data: Dict[str, Any]):
        if self.question_number >= self.MAX_QUESTIONS:
            raise ValueError(f"Cannot add question. Hard limit of {self.MAX_QUESTIONS} questions reached.")
        
        self.question_number += 1
        day = question_data.get("curriculumDay")
        topic = question_data.get("topic")
        if day and day not in self.covered_days:
            self.covered_days.append(day)
        if topic and topic not in self.covered_topics:
            self.covered_topics.append(topic)
        self.questions_asked.append(question_data)

    def record_answer_and_eval(self, answer: str, evaluation: Dict[str, Any]):
        self.answers.append(answer)
        self.evaluations.append(evaluation)
        if self.questions_asked:
            self.questions_asked[-1]["answer"] = answer
            self.questions_asked[-1]["evaluation"] = evaluation

        # Update difficulty based on evaluation recommendation unless manually overridden
        next_diff = evaluation.get("recommended_difficulty") or evaluation.get("nextDifficulty")
        if next_diff and next_diff in ["easy", "medium", "hard"]:
            self.current_difficulty = next_diff

    @property
    def total_questions_asked(self) -> int:
        return len(self.questions_asked)

    @property
    def total_answers_received(self) -> int:
        return len(self.answers)

    @property
    def total_days_covered(self) -> int:
        return len(self.covered_days)

    def should_show_completion_choice(self) -> bool:
        return (
            self.total_answers_received == self.MIN_QUESTIONS and
            not self.extended_session and
            not self.user_finished_at_10
        )

    def is_interview_complete(self) -> bool:
        if self.user_finished_at_10:
            return True
        if self.total_answers_received >= self.MAX_QUESTIONS:
            return True
        return False

    def get_live_performance_metrics(self) -> Optional[Dict[str, int]]:
        """Calculate real performance averages from turn evaluations."""
        if not self.evaluations:
            return None
        
        c_list = [e.get("correctness", 7) for e in self.evaluations]
        d_list = [e.get("technical_depth", 7) for e in self.evaluations]
        r_list = [e.get("reasoning", 7) for e in self.evaluations]
        p_list = [e.get("practical_understanding", 7) for e in self.evaluations]
        m_list = [e.get("communication", 7) for e in self.evaluations]

        tech_acc = int(sum(c_list + d_list) / (len(c_list) * 2) * 10)
        comm = int(sum(m_list) / len(m_list) * 10)
        prob_solve = int(sum(r_list + p_list) / (len(r_list) * 2) * 10)
        conf = int((tech_acc * 0.4) + (comm * 0.3) + (prob_solve * 0.3))

        return {
            "technicalAccuracy": min(100, max(0, tech_acc)),
            "communication": min(100, max(0, comm)),
            "problemSolving": min(100, max(0, prob_solve)),
            "confidence": min(100, max(0, conf))
        }

    def get_conversation_turns(self) -> List[Dict[str, Any]]:
        """Return full conversational stream for UI rendering."""
        stream = []
        for i, q in enumerate(self.questions_asked):
            # AI Question Turn
            stream.append({
                "id": f"q-{i+1}",
                "sender": "interviewer",
                "text": q.get("question", ""),
                "day": q.get("curriculumDay", 7),
                "topic": q.get("topic", "General"),
                "difficulty": q.get("difficulty", "medium"),
                "questionNumber": i + 1
            })
            # Candidate Answer Turn (if answered)
            if i < len(self.answers):
                eval_data = self.evaluations[i] if i < len(self.evaluations) else None
                stream.append({
                    "id": f"a-{i+1}",
                    "sender": "candidate",
                    "text": self.answers[i],
                    "evaluation": eval_data,
                    "questionNumber": i + 1
                })
        return stream


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def create_session(self, session_id: str, candidate: Dict[str, Any], profile_analysis: Dict[str, Any]) -> SessionState:
        session = SessionState(session_id, candidate, profile_analysis)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()
