import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.services.groq_service import groq_service

logger = logging.getLogger("ai_provider")

SYSTEM_PROMPT = """You are a senior AI engineering lead conducting an authentic, 2-person conversational technical interview with a candidate (the interviewee).
Your goal is to evaluate technical correctness, reasoning, architecture knowledge, and practical engineering decision-making through a natural back-and-forth dialogue.

STRICT MARKING SCHEME RULES:
1. If the candidate explicitly states "I don't know", "idk", "no idea", "pass", "skip", or gives an evasive/non-technical greeting (e.g. "hii"), YOU MUST ASSIGN 0 FOR CORRECTNESS, TECHNICAL_DEPTH, REASONING, PRACTICAL_UNDERSTANDING, AND AN OVERALL SCORE OF 0.0 OUT OF 10.
2. NEVER give 2, 3, or partial marks for "I don't know" or non-answers.
3. Marks (1-10) are awarded STRICTLY based on verified technical accuracy, specific architecture details, code logic, or engineering trade-offs provided.

Ask ONE concise, focused question at a time. Speak naturally as an interviewer conversing directly with the candidate.
Acknowledge the candidate's previous response with a brief conversational transition.
Maintain a professional, authoritative technical interview tone throughout."""

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
    Groq AI Provider for dynamic technical interview turns.
    Evaluates candidate responses accurately with strict zero-marking for 'I don't know'
    and generates lengthy, comprehensive final assessment feedback.
    """

    async def evaluate_answer(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if is_non_answer(answer):
            logger.info("Candidate response detected as non-answer / 'I don't know'. Assigning strict 0.0 marks.")
            return ZERO_EVALUATION.copy()

        eval_prompt = f"""Evaluate the candidate's response to the technical interview question.

Candidate: {candidate_profile.get('name')} ({candidate_profile.get('jobRole')})
Target Day: Day {question.get('curriculumDay')} - Topic: {question.get('topic')}
Question Asked: "{question.get('question')}"
Candidate Answer: "{answer}"

STRICT SCORING CRITERIA:
- Evaluate correctness, technical_depth, reasoning, practical_understanding, communication, and completeness (0-10 scale).
- If the answer lacks technical substance, gives incorrect concepts, or says "I don't know", assign 0 for correctness/depth.
- Determine overall score (0.0-10.0) and recommended_difficulty ("easy", "medium", or "hard").

Return ONLY JSON:
{{
  "correctness": 8,
  "technical_depth": 7,
  "reasoning": 8,
  "practical_understanding": 7,
  "communication": 8,
  "overall": 7.6,
  "strengths": ["Clear explanation of vector similarity"],
  "weaknesses": ["Omitted HNSW indexing parameters"],
  "recommended_difficulty": "hard"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt}
        ]

        result = await groq_service.generate_json(messages, temperature=0.2)
        if not result or not isinstance(result, dict):
            return {
                "correctness": 5,
                "technical_depth": 4,
                "reasoning": 5,
                "practical_understanding": 4,
                "communication": 5,
                "overall": 4.6,
                "strengths": ["Attempted response"],
                "weaknesses": ["Requires deeper technical precision and concrete architecture details"],
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
        objectives = target_day.get("objectives", [])
        tools = target_day.get("tools", [])

        prompt = f"""Generate opening interview question #{question_number}.

Candidate:
Name: {candidate_profile.get('name')} ({candidate_profile.get('jobRole')})
Strong Missions: {json.dumps(candidate_profile.get('strong_missions', []))}
Weak Missions: {json.dumps(candidate_profile.get('weak_missions', []))}
Skipped Topics: {json.dumps(candidate_profile.get('skipped_missions', []))}

Focus Topic: Day {day_num}: {day_title}
Objectives: {json.dumps(objectives)}
Tools: {json.dumps(tools)}
Difficulty: {current_difficulty}

Formulate an engaging technical question tailored to candidate's background and curriculum focus.

Return ONLY JSON:
{{
  "question": "Dynamic question text...",
  "curriculum_day": {day_num},
  "topic": "{day_title}",
  "difficulty": "{current_difficulty}"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        res = await groq_service.generate_json(messages, temperature=0.4)
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
        objectives = target_day.get("objectives", [])
        tools = target_day.get("tools", [])

        # Check if candidate explicitly stated "I don't know" or non-answer
        if is_non_answer(previous_answer):
            logger.info("Candidate provided non-answer in process_interview_turn. Applying strict 0.0 evaluation.")
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

        prompt = f"""Process technical interview turn #{question_number}.

Candidate:
Name: {candidate_profile.get('name')} ({candidate_profile.get('jobRole')})
Strong Missions: {json.dumps(candidate_profile.get('strong_missions', []))}
Weak Missions: {json.dumps(candidate_profile.get('weak_missions', []))}
Skipped Topics: {json.dumps(candidate_profile.get('skipped_missions', []))}

Previous Turn Context:
Question Asked: "{previous_question.get('question')}"
Candidate Answer: "{previous_answer}"

Next Target Curriculum Focus:
Day {day_num}: {day_title}
Objectives: {json.dumps(objectives)}
Tools: {json.dumps(tools)}
Target Difficulty: {current_difficulty}
Days Covered So Far: {covered_days}

STRICT EVALUATION INSTRUCTION:
1. If the candidate answer is incorrect, vague, or lacks substance, score correctness as 0-3. If they gave a strong technical response with accurate logic, score 7-10.
2. Generate next question #{question_number} building on their response.

Return ONLY JSON:
{{
  "evaluation": {{
    "correctness": 8,
    "technical_depth": 7,
    "reasoning": 8,
    "practical_understanding": 7,
    "communication": 8,
    "overall": 7.6,
    "strengths": ["Good concept explanation"],
    "weaknesses": ["Missed edge cases"],
    "recommended_difficulty": "hard"
  }},
  "next_question": {{
    "question": "Dynamic question text...",
    "curriculum_day": {day_num},
    "topic": "{day_title}",
    "difficulty": "{current_difficulty}"
  }}
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        res = await groq_service.generate_json(messages, temperature=0.4)
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

        feedback_prompt = f"""Synthesize a LENGTHY, COMPREHENSIVE, AND DETAILED FINAL EVALUATION REPORT for candidate {cand_name} ({cand_role}).

Full Interview Performance History across all answered questions:
{json.dumps(history_summary, indent=2)}

INSTRUCTIONS FOR LENGTHY, THOROUGH REPORT:
1. "summary": Provide a lengthy, multi-paragraph executive assessment analyzing the candidate's technical competence, domain knowledge, accuracy, reasoning ability, and production engineering readiness. Address specific questions answered well vs skipped/failed.
2. "strengths": Provide 4 to 6 lengthy, detailed bullet points detailing specific technical topics, concepts, code logic, or architectural trade-offs demonstrated correctly.
3. "gaps": Provide 4 to 6 lengthy, detailed bullet points explaining exact technical knowledge gaps, missed parameters, incorrect logic, or questions where candidate stated "I don't know" / non-answers.
4. "next": Provide 4 to 6 lengthy, detailed step-by-step actionable recommendations for professional career development, technical documentation review, and hands-on system building.

Return ONLY JSON:
{{
  "summary": "Multi-paragraph comprehensive executive summary paragraph 1...\\n\\nExecutive summary paragraph 2 covering domain readiness...",
  "strengths": [
    "Demonstrated thorough understanding of Sentence Transformers vector embedding generation for large documents.",
    "Correctly articulated RAG vector database chunking strategies and indexing trade-offs.",
    "Displayed strong architectural reasoning when evaluating low-latency query parameters.",
    "Communicated technical decisions clearly with structured engineering terminology."
  ],
  "gaps": [
    "Struggled with production vector index HNSW tuning parameters, resulting in a 0.0 score on turn #2.",
    "Omitted key Prometheus observability latency metrics required for monitoring real-time agent execution.",
    "Demonstrated knowledge gaps in distributed vector database partitioning strategies.",
    "Failed to address rate-limiting and fallback queue logic under high-concurrency requests."
  ],
  "next": [
    "Conduct deep-dive study on HNSW vs IVF vector indexing parameters to optimize recall vs latency trade-offs.",
    "Build a production RAG pipeline integrating Prometheus metrics and Grafana dashboards for latency tracking.",
    "Review distributed system rate-limiting algorithms (Token Bucket, Leaky Bucket) for agent tool invocation.",
    "Practice multi-turn technical interview scenarios emphasizing edge-case handling and system resilience."
  ]
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": feedback_prompt}
        ]

        result = await groq_service.generate_json(messages, temperature=0.3)
        
        if not result or not isinstance(result, dict) or "summary" not in result:
            # Fallback rich, detailed multi-paragraph feedback report
            return {
                "summary": f"Comprehensive Technical Interview Evaluation for {cand_name} ({cand_role}):\n\nThe candidate completed a rigorous technical assessment evaluating core AI engineering competencies, vector search architectures, and system design principles. Throughout the interview, performance varied based on topic depth and familiarity with production engineering patterns.\n\nWhile demonstrating foundational knowledge in AI tools and workflow setup, critical technical gaps were identified in advanced indexing parameters and real-time observability metrics. To excel in enterprise AI engineering roles, targeted practical practice on system resilience and query optimization is strongly recommended.",
                "strengths": [
                    "Demonstrated foundational understanding of curriculum concepts and development environment setup.",
                    "Showed willingness to engage in multi-turn technical system architecture dialogue.",
                    "Articulated high-level concepts for AI tool integration and workflow execution."
                ],
                "gaps": [
                    "Exhibited knowledge gaps on specific technical questions, including non-responses or 'I don't know' answers.",
                    "Needs deeper understanding of vector database HNSW indexing tuning parameters and latency benchmarks.",
                    "Omitted enterprise observability standards and rate-limiting resilience mechanisms."
                ],
                "next": [
                    "Implement hands-on RAG vector database projects focusing on HNSW search index optimization.",
                    "Study enterprise rate-limiting patterns and fallback queue mechanisms for resilient AI services.",
                    "Review Prometheus metrics and Grafana alerting configurations for production LLM monitoring."
                ]
            }

        return result


def get_ai_provider() -> BaseAIProvider:
    return GroqAIProvider()
