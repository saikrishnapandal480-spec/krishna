import json
from pathlib import Path
from typing import Dict, Any, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"

class CurriculumService:
    def __init__(self):
        self.curriculum_data = self._load_data()
        self.days = self.curriculum_data.get("days", [])
        self.modules = self.curriculum_data.get("modules", [])
        self._day_map = {d["day"]: d for d in self.days}

    def _load_data(self) -> Dict[str, Any]:
        file_path = DATA_DIR / "curriculum.json"
        if not file_path.exists():
            return {"days": [], "modules": []}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_days(self) -> List[Dict[str, Any]]:
        return self.days

    def get_day(self, day_num: int) -> Optional[Dict[str, Any]]:
        return self._day_map.get(day_num)

    def get_module_for_day(self, day_num: int) -> Optional[Dict[str, Any]]:
        for mod in self.modules:
            day_range = mod.get("days", [])
            if len(day_range) == 2 and day_range[0] <= day_num <= day_range[1]:
                return mod
        return None

    def get_available_days_for_interview(self) -> List[Dict[str, Any]]:
        """Return key curriculum days suitable for questioning."""
        # Prioritize core build/AI days (e.g. days 7, 8, 9, 10, 11, 12, 13, 16, 18, 21, 22, 23, 27, 28, 29, 31)
        core_days = [7, 8, 10, 11, 12, 13, 16, 18, 21, 22, 23, 27, 28, 29, 31]
        return [self._day_map[d] for d in core_days if d in self._day_map]

curriculum_service = CurriculumService()
