from typing import Dict, Any, List
from backend.services.ai_provider import get_ai_provider

class FeedbackAgent:
    def __init__(self):
        pass

    async def generate_feedback(
        self,
        candidate_profile: Dict[str, Any],
        questions_and_evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        provider = get_ai_provider()
        return await provider.generate_final_feedback(candidate_profile, questions_and_evaluations)

feedback_agent = FeedbackAgent()
