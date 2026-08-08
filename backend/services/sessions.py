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
        self.questions_asked: List[Dict[str, Any]] = []
        self.answers: List[str] = []
        self.evaluations: List[Dict[str, Any]] = []
        self.covered_days: List[int] = []
        self.covered_topics: List[str] = []
        self.current_difficulty: str = "medium"
        self.status: str = "in_progress"  # "in_progress", "awaiting_choice", "completed"
        self.final_feedback: Optional[Dict[str, Any]] = None

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

        # Update difficulty based on evaluation recommendation
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
        """
        At Question 10 answer submission, if user has not yet decided to extend or finish,
        show the completion choice UI.
        """
        return (
            self.total_answers_received == self.MIN_QUESTIONS and
            not self.extended_session and
            not self.user_finished_at_10
        )

    def is_interview_complete(self) -> bool:
        """
        Interview is complete if:
        1. User explicitly finished at Q10 choice, OR
        2. Total answers received reaches MAX_QUESTIONS (15).
        """
        if self.user_finished_at_10:
            return True
        if self.total_answers_received >= self.MAX_QUESTIONS:
            return True
        return False

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
