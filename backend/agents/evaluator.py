from typing import Dict, Any, List
from backend.services.ai_provider import get_ai_provider

class EvaluatorAgent:
    def __init__(self):
        self.ai_provider = get_ai_provider()

    async def evaluate(
        self,
        candidate_profile: Dict[str, Any],
        question: Dict[str, Any],
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate candidate's technical response on correctness, depth, reasoning,
        practical understanding, and communication clarity.
        """
        provider = get_ai_provider()
        return await provider.evaluate_answer(candidate_profile, question, answer, history)

evaluator_agent = EvaluatorAgent()
