import logging
from typing import Dict, Any, List, Optional
from backend.services.ai_provider import get_ai_provider
from backend.services.curriculum import curriculum_service
from backend.services.sessions import SessionState

logger = logging.getLogger("interviewer_agent")

class InterviewPlanner:
    """
    Interview Planner responsible for topic transitions, curriculum coverage tracking,
    difficulty adaptation, and turn execution.
    """
    MINIMUM_QUESTIONS = 8
    MINIMUM_CURRICULUM_DAYS = 4

    def select_target_curriculum_day(self, session: SessionState) -> Dict[str, Any]:
        """
        Dynamically determine the next curriculum day to evaluate based on candidate profile and covered days.
        """
        covered_days = session.covered_days
        profile = session.profile_analysis
        all_days = curriculum_service.get_all_days()

        # Candidate completed missions days
        completed_mission_days = [m["day"] for m in profile.get("strong_missions", []) + profile.get("weak_missions", [])]
        uncovered_completed_days = [d for d in completed_mission_days if d not in covered_days]

        if uncovered_completed_days and len(covered_days) < self.MINIMUM_CURRICULUM_DAYS:
            target_day_num = uncovered_completed_days[0]
            day_obj = curriculum_service.get_day(target_day_num)
            if day_obj:
                return day_obj

        # Pick any uncovered curriculum day if completed missions fully covered
        for d_obj in all_days:
            if d_obj["day"] not in covered_days:
                return d_obj

        # Fallback to repeating covered day
        fallback_day = covered_days[session.question_number % len(covered_days)] if covered_days else 7
        return curriculum_service.get_day(fallback_day) or all_days[0]

    async def generate_initial_turn(self, session: SessionState) -> Optional[Dict[str, Any]]:
        """
        Generate question #1 for turn 0 using Groq.
        """
        target_day = self.select_target_curriculum_day(session)
        provider = get_ai_provider()
        return await provider.generate_question(
            candidate_profile=session.profile_analysis,
            curriculum_context=curriculum_service.curriculum_data,
            target_day=target_day,
            previous_question=None,
            previous_answer=None,
            previous_evaluation=None,
            covered_days=session.covered_days,
            covered_topics=session.covered_topics,
            current_difficulty=session.current_difficulty,
            question_number=1
        )

    async def process_turn(self, session: SessionState, user_answer: str) -> Optional[Dict[str, Any]]:
        """
        Process candidate answer turn: Evaluates previous answer AND generates next question in 1 combined Groq call.
        Returns dict with "evaluation" and "next_question".
        """
        target_day = self.select_target_curriculum_day(session)
        previous_question = session.questions_asked[-1] if session.questions_asked else {"question": "", "curriculumDay": 7, "topic": "General AI"}
        next_q_num = session.question_number + 1

        provider = get_ai_provider()
        
        # Check if provider supports single combined call
        if hasattr(provider, "process_interview_turn"):
            turn_result = await provider.process_interview_turn(
                candidate_profile=session.profile_analysis,
                curriculum_context=curriculum_service.curriculum_data,
                target_day=target_day,
                previous_question=previous_question,
                previous_answer=user_answer,
                covered_days=session.covered_days,
                covered_topics=session.covered_topics,
                current_difficulty=session.current_difficulty,
                question_number=next_q_num
            )
            return turn_result
        else:
            # Fallback sequential
            eval_data = await provider.evaluate_answer(
                session.profile_analysis, previous_question, user_answer, session.questions_asked
            )
            next_diff = eval_data.get("recommended_difficulty", session.current_difficulty)
            q_data = await provider.generate_question(
                session.profile_analysis, curriculum_service.curriculum_data, target_day,
                previous_question, user_answer, eval_data,
                session.covered_days, session.covered_topics, next_diff, next_q_num
            )
            if not q_data:
                return None
            return {
                "evaluation": eval_data,
                "next_question": q_data
            }

interviewer_agent = InterviewPlanner()
