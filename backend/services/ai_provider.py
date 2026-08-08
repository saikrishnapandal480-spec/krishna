import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.services.groq_service import groq_service

logger = logging.getLogger("ai_provider")

SYSTEM_PROMPT = """You are a senior AI engineering lead conducting an authentic, 2-person conversational technical interview with a candidate (the interviewee).
Your goal is to evaluate technical correctness, reasoning, architecture knowledge, and practical engineering decision-making through a natural back-and-forth dialogue.
Use ONLY the provided curriculum and candidate profile as the basis for assessment.
Ask ONE concise, focused question at a time. Speak naturally as an interviewer conversing directly with the candidate.
Acknowledge the candidate's previous response with a brief conversational transition (e.g., "Thanks for that explanation...", "That's a solid point on X...", or "Building on what you said about Y...").
Never repeat a question.
If the candidate gives a strong answer, increase difficulty with deeper follow-ups.
If the candidate struggles, simplify and probe fundamentals.
Prefer practical engineering scenarios, trade-offs, debugging, and system architecture over simple textbook definitions.
Respect skipped topics. Do NOT test skipped missions as known material.
Maintain a natural, professional, conversational interview tone throughout."""

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
    Evaluates candidate responses accurately and synthesizes non-contradictory feedback.
    """

    async def evaluate_answer(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        eval_prompt = f"""Evaluate the candidate's response to the technical interview question.

Candidate: {candidate_profile.get('name')} ({candidate_profile.get('jobRole')})
Target Day: Day {question.get('curriculumDay')} - Topic: {question.get('topic')}
Question Asked: "{question.get('question')}"
Candidate Answer: "{answer}"

Evaluate correctness, technical_depth, reasoning, practical_understanding, communication, and completeness (1-10 scale).
Determine overall score (1.0-10.0) and recommended_difficulty ("easy", "medium", or "hard").

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
                "correctness": 6,
                "technical_depth": 6,
                "reasoning": 6,
                "practical_understanding": 6,
                "communication": 6,
                "overall": 6.0,
                "strengths": ["Attempted response"],
                "weaknesses": ["Need deeper technical detail"],
                "recommended_difficulty": "medium"
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

Formulate an engaging opening technical question tailored to candidate's background and completed curriculum topics.

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

        res = await groq_service.generate_json(messages, temperature=0.5)
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
Topics Covered So Far: {covered_topics}

Instructions:
1. Evaluate candidate's previous answer (score 1-10, strengths, weaknesses, recommended_difficulty).
2. Generate next question #{question_number}. Use previous answer to build an intelligent follow-up (e.g. "You mentioned X... How would you...").

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

        feedback_prompt = f"""Synthesize complete interview performance into final assessment feedback.

Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Full Interview Performance History (Questions, Answers, and Evaluations):
{json.dumps(history_summary, indent=2)}

Strict Accuracy Requirements:
1. Provide an executive summary summarizing performance across all answered questions.
2. STRENGTHS: List what the candidate answered correctly with specific technical details from their actual answers.
3. GAPS: List what was incorrect or missing in their answers (e.g. missing evaluation metrics, missing rate limiting).
4. RECOMMENDED NEXT STEPS: Provide clear, actionable recommendations on how the candidate can improve.
5. Do NOT provide contradictory statements (e.g. calling a topic both a strength and gap without distinction).
6. If an area was unassessed or data is uncertain, state it clearly.

Return ONLY JSON:
{{
  "summary": "Executive evaluation summary paragraph...",
  "strengths": [
    "Accurately explained vector embeddings using Sentence Transformers",
    "Solid understanding of multi-agent routing"
  ],
  "gaps": [
    "Omitted vector database HNSW indexing tuning parameters",
    "Needs deeper knowledge of Prometheus observability latency metrics"
  ],
  "next": [
    "Practice benchmarking vector search recall vs latency trade-offs",
    "Implement Grafana dashboard alerts for LLM latency tracking"
  ]
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": feedback_prompt}
        ]

        return await groq_service.generate_json(messages, temperature=0.3)


def get_ai_provider() -> BaseAIProvider:
    return GroqAIProvider()
