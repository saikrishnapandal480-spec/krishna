import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("session_manager")

class SessionState:
    MIN_QUESTIONS = 10
    MAX_QUESTIONS = 15

    def __init__(self, session_id: str, candidate: Dict[str, Any], profile_analysis: Dict[str, Any]):
        self.session_id = session_id
        self.candidate = candidate
        self.profile_analysis = profile_analysis

        self.questions_asked: List[Dict[str, Any]] = []
        self.answers: List[str] = []
        self.evaluations: List[Dict[str, Any]] = []
        self.covered_days: List[int] = []
        self.covered_topics: List[str] = []

        self.current_difficulty = "medium"
        self.custom_notes = ""
        self.status = "initialized"  # initialized, in_progress, awaiting_choice, completed
        self.user_finished_at_10 = False
        self.extended_session = False
        self.final_feedback: Optional[Dict[str, Any]] = None
        self.created_at = time.time()

    def update_profile_customization(
        self,
        name: Optional[str] = None,
        job_role: Optional[str] = None,
        years_experience: Optional[int] = None,
        difficulty_override: Optional[str] = None,
        custom_notes: Optional[str] = None
    ):
        """Update active session candidate customization profile."""
        member = self.candidate.get("member", {})
        if name:
            member["name"] = name.strip()
        if job_role:
            member["jobRole"] = job_role.strip()
        if years_experience is not None:
            member["yearsExperience"] = years_experience

        if difficulty_override and difficulty_override in ["easy", "medium", "hard"]:
            self.current_difficulty = difficulty_override

        if custom_notes is not None:
            self.custom_notes = custom_notes.strip()

    def add_question(self, question_data: Dict[str, Any]):
        self.questions_asked.append(question_data)
        day = question_data.get("curriculumDay")
        topic = question_data.get("topic")

        if day and day not in self.covered_days:
            self.covered_days.append(day)
        if topic and topic not in self.covered_topics:
            self.covered_topics.append(topic)

    def record_answer_and_eval(self, answer: str, evaluation: Dict[str, Any]):
        self.answers.append(answer.strip())
        self.evaluations.append(evaluation)

        # Update difficulty based on recommendation if valid
        rec_diff = evaluation.get("recommended_difficulty")
        if rec_diff in ["easy", "medium", "hard"]:
            self.current_difficulty = rec_diff

    @property
    def question_number(self) -> int:
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
        
        valid_evals = [e for e in self.evaluations if e and not e.get("evaluation_error")]
        if not valid_evals:
            return None
        
        c_list = [e.get("correctness", e.get("score", 0)) for e in valid_evals]
        d_list = [e.get("technical_depth", e.get("technical_accuracy", 0)) for e in valid_evals]
        r_list = [e.get("reasoning", e.get("problem_solving", 0)) for e in valid_evals]
        p_list = [e.get("practical_understanding", e.get("relevance", 0)) for e in valid_evals]
        m_list = [e.get("communication", 0) for e in valid_evals]

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
                score_val = None
                if eval_data and not eval_data.get("evaluation_error"):
                    if "score" in eval_data and eval_data["score"] is not None:
                        score_val = eval_data["score"]
                    elif "overall" in eval_data and eval_data["overall"] is not None:
                        score_val = eval_data["overall"]

                stream.append({
                    "id": f"a-{i+1}",
                    "sender": "candidate",
                    "text": self.answers[i],
                    "evaluation": eval_data,
                    "score": score_val,
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

# Singleton Session Manager
session_manager = SessionManager()
