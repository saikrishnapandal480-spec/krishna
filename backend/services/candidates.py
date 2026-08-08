import json
from pathlib import Path
from typing import Dict, Any, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"

class CandidateService:
    def __init__(self):
        self.candidates_data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        file_path = DATA_DIR / "candidates.json"
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("candidates", [])

    def get_all_candidates(self) -> List[Dict[str, Any]]:
        return self.candidates_data

    def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        for c in self.candidates_data:
            member = c.get("member", {})
            if member.get("id") == candidate_id:
                return c
        return None

    def analyze_profile(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze candidate missions and signals to personalize the interview.
        Categorize into strong_days, weak_days, and skipped_days.
        """
        member = candidate.get("member", {})
        missions = candidate.get("missions", [])
        signals = candidate.get("signals", {})

        strong_missions = []
        weak_missions = []
        skipped_missions = []

        for m in missions:
            day = m.get("day")
            title = m.get("title")
            passed = m.get("passed", False)
            attempts = m.get("attempts", 1)
            skipped = m.get("skipped", False)

            if skipped:
                skipped_missions.append({"day": day, "title": title})
            elif passed and attempts <= 2:
                strong_missions.append({"day": day, "title": title, "attempts": attempts})
            else:  # attempts > 2 or not passed
                weak_missions.append({"day": day, "title": title, "attempts": attempts, "passed": passed})

        return {
            "id": member.get("id", "UNKNOWN"),
            "name": member.get("name", "Candidate"),
            "jobRole": member.get("jobRole", "Engineer"),
            "yearsExperience": member.get("yearsExperience", 0),
            "education": member.get("education", ""),
            "strong_missions": strong_missions,
            "weak_missions": weak_missions,
            "skipped_missions": skipped_missions,
            "signals": signals
        }

candidate_service = CandidateService()
