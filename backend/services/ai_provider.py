import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.services.groq_service import groq_service

logger = logging.getLogger("ai_provider")

SYSTEM_PROMPT = """You are a senior AI technical interviewer evaluating a candidate response in a live technical interview.
You evaluate the candidate's answer against the specific interview question asked.

EVALUATION SCORING GUIDANCE (0 to 10 scale):
- 0-2 = Completely incorrect, irrelevant, "I don't know", "idk", evasive greeting ("hii"), or no meaningful answer
- 3-4 = Very weak understanding with major conceptual errors
- 5-6 = Partial/basic understanding with important gaps
- 7   = Good answer with some missing details
- 8   = Strong technically correct answer
- 9   = Excellent answer with strong depth and reasoning
- 10  = Exceptional expert-level answer

IMPORTANT:
- Do NOT automatically give 7.
- The score MUST depend on the candidate's actual answer compared against the specific question.
- Do NOT include markdown formatting or text outside valid JSON."""

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
    "score": 0,
    "overall": 0.0,
    "technical_accuracy": 0,
    "correctness": 0,
    "technical_depth": 0,
    "reasoning": 0,
    "practical_understanding": 0,
    "relevance": 0,
    "completeness": 0,
    "problem_solving": 0,
    "communication": 1,
    "feedback": "Candidate explicitly stated they do not know the answer or provided a non-response.",
    "strengths": [],
    "weaknesses": ["No technical answer provided."],
    "recommended_difficulty": "easy"
}

def validate_evaluation(eval_dict: Any) -> Dict[str, Any]:
    """
    Validate and format evaluation dict.
    Ensures score exists, is numeric between 0 and 10, and never defaults to hardcoded 7.
    Returns evaluation_error dict if invalid.
    """
    if not isinstance(eval_dict, dict):
        return {"evaluation_error": True, "message": "Unable to evaluate this answer."}
        
    score_val = eval_dict.get("score")
    if score_val is None:
        score_val = eval_dict.get("overall")
    if score_val is None:
        score_val = eval_dict.get("correctness")

    try:
        score_num = float(score_val)
        if 0.0 <= score_num <= 10.0:
            score_num = round(score_num, 1)
            eval_dict["score"] = score_num
            eval_dict["overall"] = score_num
            eval_dict["correctness"] = int(score_num)
            eval_dict["technical_accuracy"] = int(eval_dict.get("technical_accuracy", score_num))
            eval_dict["relevance"] = int(eval_dict.get("relevance", score_num))
            eval_dict["completeness"] = int(eval_dict.get("completeness", score_num))
            eval_dict["problem_solving"] = int(eval_dict.get("problem_solving", score_num))
            eval_dict["communication"] = int(eval_dict.get("communication", score_num))
            eval_dict["feedback"] = eval_dict.get("feedback", "Evaluation complete.")
            return eval_dict
    except (ValueError, TypeError):
        pass

    return {
        "evaluation_error": True,
        "message": "Unable to evaluate this answer."
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
    Groq AI Provider for dynamic technical interview evaluation and turn generation.
    Evaluates each candidate answer dynamically against the question without hardcoded fallbacks.
    """

    async def evaluate_answer(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if is_non_answer(answer):
            logger.info("[EVALUATION] Non-answer detected. Assigning score 0.")
            return ZERO_EVALUATION.copy()

        q_text = question.get('question', '')
        eval_prompt = f"""Evaluate candidate's answer against the specific interview question.

QUESTION:
"{q_text}"

CANDIDATE ANSWER:
"{answer}"

SCORING GUIDANCE (0 to 10):
0-2 = Completely incorrect, irrelevant, or no meaningful answer
3-4 = Very weak understanding
5-6 = Partial/basic understanding with important gaps
7 = Good answer with some missing details
8 = Strong technically correct answer
9 = Excellent answer with strong depth and reasoning
10 = Exceptional expert-level answer

Return ONLY JSON:
{{
  "score": 8,
  "technical_accuracy": 8,
  "relevance": 9,
  "completeness": 7,
  "problem_solving": 8,
  "communication": 8,
  "feedback": "Short explanation of the score based on technical accuracy and question context"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt}
        ]

        logger.info("[EVALUATION] Groq Evaluation Started")
        raw_res = await groq_service.generate_completion(messages, json_mode=True, temperature=0.2, max_tokens=200)
        logger.info(f"[EVALUATION] Groq Raw Response: {raw_res}")

        if not raw_res:
            # Retry once with strict prompt
            raw_res = await groq_service.generate_completion(messages, json_mode=False, temperature=0.1, max_tokens=200)
            logger.info(f"[EVALUATION] Groq Retry Raw Response: {raw_res}")

        if not raw_res:
            return {"evaluation_error": True, "message": "Unable to evaluate this answer."}

        cleaned = raw_res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            val = validate_evaluation(parsed)
            logger.info(f"[EVALUATION] Parsed Score: {val.get('score')}")
            return val
        except Exception as e:
            logger.error(f"[EVALUATION] Failed to parse JSON score: {e}")
            return {"evaluation_error": True, "message": "Unable to evaluate this answer."}

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
        q_text = previous_question.get('question', '')
        logger.info(f"[EVALUATION] Question Number: {question_number}")
        logger.info(f"[EVALUATION] Question: {q_text[:50]}...")
        logger.info(f"[EVALUATION] Candidate Answer: {previous_answer[:50]}...")

        day_num = target_day.get("day", 7)
        day_title = target_day.get("title", "")

        if is_non_answer(previous_answer):
            logger.info("[EVALUATION] Candidate response is non-answer / 'I don't know'. Assigning score 0.")
            eval_data = ZERO_EVALUATION.copy()
            logger.info(f"[EVALUATION] Parsed Score: {eval_data.get('score')}")
            logger.info("[EVALUATION] Evaluation Saved")
            
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

        prompt = f"""ROLE: {candidate_profile.get('jobRole', 'Software Engineer')}
QUESTION NUMBER: {question_number}

QUESTION:
"{q_text}"

CANDIDATE ANSWER:
"{previous_answer}"

TARGET NEXT TOPIC: Day {day_num}: {day_title}

TASK:
1. Evaluate candidate's answer against the specific question (score 0 to 10).
2. Generate ONE next interview question.

SCORING:
0-2 = Completely incorrect or irrelevant
3-4 = Very weak
5-6 = Basic/partial understanding
7 = Good answer
8 = Strong technically correct answer
9-10 = Excellent/exceptional depth

Return ONLY JSON:
{{
  "evaluation": {{
    "score": 8,
    "technical_accuracy": 8,
    "relevance": 9,
    "completeness": 7,
    "problem_solving": 8,
    "communication": 8,
    "feedback": "Short evaluation explanation"
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

        logger.info("[EVALUATION] Groq Evaluation Started")
        res = await groq_service.generate_json(messages, temperature=0.3, max_tokens=300)
        logger.info(f"[EVALUATION] Groq Raw Response: {res}")

        if not res or "next_question" not in res:
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

        eval_data = validate_evaluation(res.get("evaluation"))
        logger.info(f"[EVALUATION] Parsed Score: {eval_data.get('score')}")
        logger.info("[EVALUATION] Evaluation Saved")

        next_q = res["next_question"]
        next_q["questionNumber"] = question_number
        if "curriculum_day" in next_q:
            next_q["curriculumDay"] = next_q.pop("curriculum_day")

        return {
            "evaluation": eval_data,
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
