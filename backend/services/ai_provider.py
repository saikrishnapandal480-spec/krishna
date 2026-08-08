import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.services.groq_service import groq_service

logger = logging.getLogger("ai_provider")

SYSTEM_PROMPT = """You are a senior AI technical interviewer conducting a live engineering assessment.
Evaluate technical correctness strictly and formulate ONE concise follow-up question at a time.

STRICT MARKING RULES:
- If the candidate states "I don't know", "idk", "no idea", "pass", "skip", or an evasive greeting (e.g. "hii"), assign 0 for correctness, depth, reasoning, understanding, and an overall score of 0.0 / 10.
- Award marks strictly based on verified technical accuracy and engineering trade-offs provided."""

def is_non_answer(answer_text: str) -> bool:
    """Check if the candidate response is 'I don't know', 'idk', 'pass', 'hii', or a non-technical phrase."""
    if not answer_text or not answer_text.strip():
        return True
    
    clean = answer_text.strip().lower()
    clean_alpha = "".join(ch for ch in clean if ch.isalnum() or ch.isspace()).strip()
    
    if len(clean_alpha) < 3:
        return True
    
    non_answer_exact = {
        "i dont know", "i dont know", "dont know", "don't know",
        "idk", "no idea", "not sure", "no clue", "pass", "skip",
        "hii", "hi", "hello", "hey", "test", "na", "n/a", "none",
        "i have no idea", "no answer", "dunno", "nothing", "bye"
    }
    
    if clean_alpha in non_answer_exact:
        return True
        
    for phrase in ["i dont know", "dont know", "idk", "no idea", "no clue", "have no idea"]:
        if clean_alpha.startswith(phrase) and len(clean_alpha) < 30:
            return True
            
    return False

ZERO_EVALUATION = {
    "correctness": 0,
    "technical_depth": 0,
    "reasoning": 0,
    "practical_understanding": 0,
    "communication": 1,
    "overall": 0.0,
    "strengths": [],
    "weaknesses": ["Candidate explicitly stated they do not know the answer or provided a non-response."],
    "recommended_difficulty": "easy"
}

class BaseAIProvider(ABC):
    @abstractmethod
    async def evaluate_answer(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def generate_question(
        self,
        candidate_profile: Dict[str, Any],
        curriculum_context: Dict[str, Any],
        target_day: Dict[str, Any],
        previous_question: Optional[Dict[str, Any]],
        previous_answer: Optional[str],
        previous_evaluation: Optional[Dict[str, Any]],
        covered_days: List[int],
        covered_topics: List[str],
        current_difficulty: str,
        question_number: int
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def process_interview_turn(
        self,
        candidate_profile: Dict[str, Any],
        curriculum_context: Dict[str, Any],
        target_day: Dict[str, Any],
        previous_question: Dict[str, Any],
        previous_answer: str,
        covered_days: List[int],
        covered_topics: List[str],
        current_difficulty: str,
        question_number: int
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def generate_final_feedback(
        self,
        candidate_profile: Dict[str, Any],
        questions_and_evaluations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        pass


class GroqAIProvider(BaseAIProvider):
    """
    Groq AI Provider optimized for ultra-fast response latency.
    Uses compact prompts and single-pass turn generation for sub-second execution.
    """

    async def evaluate_answer(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if is_non_answer(answer):
            return ZERO_EVALUATION.copy()

        eval_prompt = f"""Evaluate technical response.
Candidate Role: {candidate_profile.get('jobRole', 'Software Engineer')}
Question: "{question.get('question')}"
Answer: "{answer}"

Return ONLY JSON:
{{
  "correctness": 8,
  "technical_depth": 7,
  "reasoning": 8,
  "practical_understanding": 7,
  "communication": 8,
  "overall": 7.6,
  "strengths": ["Good explanation"],
  "weaknesses": ["Omitted parameters"],
  "recommended_difficulty": "medium"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt}
        ]

        result = await groq_service.generate_json(messages, temperature=0.2, max_tokens=200)
        if not result or not isinstance(result, dict):
            return {
                "correctness": 5,
                "technical_depth": 4,
                "reasoning": 5,
                "practical_understanding": 4,
                "communication": 5,
                "overall": 4.6,
                "strengths": ["Attempted response"],
                "weaknesses": ["Needs technical precision"],
                "recommended_difficulty": "easy"
            }
        return result

    async def generate_question(
        self,
        candidate_profile: Dict[str, Any],
        curriculum_context: Dict[str, Any],
        target_day: Dict[str, Any],
        previous_question: Optional[Dict[str, Any]],
        previous_answer: Optional[str],
        previous_evaluation: Optional[Dict[str, Any]],
        covered_days: List[int],
        covered_topics: List[str],
        current_difficulty: str,
        question_number: int
    ) -> Optional[Dict[str, Any]]:
        day_num = target_day.get("day", 7)
        day_title = target_day.get("title", "")

        prompt = f"""Generate opening interview question #{question_number}.
ROLE: {candidate_profile.get('jobRole', 'Software Engineer')}
QUESTION NUMBER: {question_number}
FOCUS TOPIC: Day {day_num}: {day_title} ({current_difficulty})

TASK: Formulate opening technical interview question #{question_number}.

Return ONLY JSON:
{{
  "question": "Opening question text...",
  "curriculum_day": {day_num},
  "topic": "{day_title}",
  "difficulty": "{current_difficulty}"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        res = await groq_service.generate_json(messages, temperature=0.3, max_tokens=250)
        if not res or "question" not in res:
            return None

        return {
            "questionNumber": question_number,
            "curriculumDay": res.get("curriculum_day", day_num),
            "topic": res.get("topic", day_title),
            "question": res["question"],
            "difficulty": res.get("difficulty", current_difficulty)
        }

    async def process_interview_turn(
        self,
        candidate_profile: Dict[str, Any],
        curriculum_context: Dict[str, Any],
        target_day: Dict[str, Any],
        previous_question: Dict[str, Any],
        previous_answer: str,
        covered_days: List[int],
        covered_topics: List[str],
        current_difficulty: str,
        question_number: int
    ) -> Optional[Dict[str, Any]]:
        day_num = target_day.get("day", 7)
        day_title = target_day.get("title", "")

        # Check if candidate explicitly stated "I don't know" or non-answer
        if is_non_answer(previous_answer):
            logger.info("[INTERVIEW] Candidate response is non-answer / 'I don't know'. Applying zero evaluation.")
            eval_data = ZERO_EVALUATION.copy()
            next_q = await self.generate_question(
                candidate_profile, curriculum_context, target_day,
                previous_question, previous_answer, eval_data,
                covered_days, covered_topics, "easy", question_number
            )
            if not next_q:
                return None
            return {
                "evaluation": eval_data,
                "next_question": next_q
            }

        prompt = f"""ROLE:
{candidate_profile.get('jobRole', 'Software Engineer')}

QUESTION NUMBER:
{question_number}

PREVIOUS QUESTION:
"{previous_question.get('question', '')}"

CANDIDATE ANSWER:
"{previous_answer}"

TARGET TOPIC:
Day {day_num}: {day_title} ({current_difficulty})

TASK:
Evaluate candidate's answer internally (0-10) and generate ONE follow-up technical interview question.

Return ONLY JSON:
{{
  "evaluation": {{
    "correctness": 8, "technical_depth": 7, "reasoning": 8, "practical_understanding": 7, "communication": 8, "overall": 7.6,
    "strengths": ["Good explanation"], "weaknesses": ["Missed edge cases"], "recommended_difficulty": "{current_difficulty}"
  }},
  "next_question": {{
    "question": "Follow-up question text...",
    "curriculum_day": {day_num},
    "topic": "{day_title}",
    "difficulty": "{current_difficulty}"
  }}
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        res = await groq_service.generate_json(messages, temperature=0.3, max_tokens=300)
        if not res or "evaluation" not in res or "next_question" not in res:
            eval_data = await self.evaluate_answer(candidate_profile, previous_question, previous_answer, [])
            q_data = await self.generate_question(
                candidate_profile, curriculum_context, target_day,
                previous_question, previous_answer, eval_data,
                covered_days, covered_topics, current_difficulty, question_number
            )
            if not q_data:
                return None
            return {
                "evaluation": eval_data,
                "next_question": q_data
            }

        next_q = res["next_question"]
        next_q["questionNumber"] = question_number
        if "curriculum_day" in next_q:
            next_q["curriculumDay"] = next_q.pop("curriculum_day")

        return {
            "evaluation": res["evaluation"],
            "next_question": next_q
        }

    async def generate_final_feedback(
        self,
        candidate_profile: Dict[str, Any],
        questions_and_evaluations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        history_summary = []
        for q in questions_and_evaluations:
            history_summary.append({
                "questionNumber": q.get("questionNumber"),
                "curriculumDay": q.get("curriculumDay"),
                "topic": q.get("topic"),
                "question": q.get("question"),
                "candidateAnswer": q.get("answer"),
                "evaluation": q.get("evaluation")
            })

        cand_name = candidate_profile.get("name", "Candidate")
        cand_role = candidate_profile.get("jobRole", "Technical Role")

        feedback_prompt = f"""Synthesize a LENGTHY, COMPREHENSIVE FINAL EVALUATION REPORT for candidate {cand_name} ({cand_role}).

Full Interview History:
{json.dumps(history_summary, indent=2)}

Return ONLY JSON:
{{
  "summary": "Multi-paragraph comprehensive executive summary paragraph 1...\\n\\nExecutive summary paragraph 2...",
  "strengths": [
    "Demonstrated thorough understanding of curriculum concepts.",
    "Correctly articulated architecture trade-offs.",
    "Communicated technical decisions clearly."
  ],
  "gaps": [
    "Exhibited knowledge gaps on advanced indexing parameters.",
    "Omitted observability latency metrics.",
    "Failed to address rate-limiting resilience."
  ],
  "next": [
    "Conduct deep-dive study on vector search optimization.",
    "Build a production pipeline with monitoring dashboards.",
    "Practice multi-turn technical interview scenarios."
  ]
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": feedback_prompt}
        ]

        result = await groq_service.generate_json(messages, temperature=0.3, max_tokens=500)
        
        if not result or not isinstance(result, dict) or "summary" not in result:
            return {
                "summary": f"Comprehensive Technical Assessment for {cand_name} ({cand_role}):\n\nThe candidate completed a technical assessment evaluating core AI engineering competencies. Throughout the interview, performance varied based on topic depth and familiarity with production engineering patterns.\n\nWhile demonstrating foundational knowledge in AI tools and workflow setup, targeted practical practice on system resilience and query optimization is strongly recommended.",
                "strengths": [
                    "Demonstrated foundational understanding of curriculum concepts.",
                    "Engaged in multi-turn technical dialogue."
                ],
                "gaps": [
                    "Exhibited knowledge gaps on specific technical questions.",
                    "Needs deeper understanding of production tuning parameters."
                ],
                "next": [
                    "Implement hands-on vector database optimization projects.",
                    "Study enterprise rate-limiting patterns and fallback mechanisms."
                ]
            }

        return result


def get_ai_provider() -> BaseAIProvider:
    return GroqAIProvider()
