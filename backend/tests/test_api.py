import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

# Ensure root workspace is in python path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app
from backend.services.candidates import candidate_service

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Interview Agent" in response.text

def test_get_candidates():
    response = client.get("/api/candidates")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) >= 1

def test_get_curriculum():
    response = client.get("/api/curriculum")
    assert response.status_code == 200
    data = response.json()
    assert "days" in data
    assert len(data["days"]) == 31

def test_groq_health_endpoint():
    response = client.get("/api/groq-health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

@patch("backend.services.groq_service.groq_service.generate_json")
def test_start_interview_session(mock_groq):
    mock_groq.return_value = {
        "question": "Can you explain how embeddings convert textual data into vector representations?",
        "curriculum_day": 7,
        "topic": "Embeddings Explained",
        "difficulty": "medium"
    }

    candidates = candidate_service.get_all_candidates()
    candidate = candidates[0]

    payload = {
        "sessionId": "test-session-groq-001",
        "candidate": candidate
    }

    response = client.post("/api/interview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "reply" in data
    assert "Welcome. Let's begin your interview." in data["reply"]
    assert data["done"] is False
    assert data["questionNumber"] == 1
    assert data["minQuestions"] == 10
    assert data["maxQuestions"] == 15

@patch("backend.services.groq_service.groq_service.generate_json")
def test_state_machine_q10_choice_and_finish(mock_groq):
    """
    Test 10-question state machine:
    - Turns 1 to 10 answer submissions
    - At Answer 10, verifies showChoice=True
    - Action 'finish' triggers final Groq feedback (done=True)
    """
    import re

    def groq_side_effect(messages, **kwargs):
        prompt_str = str(messages)
        if "Generate final interview assessment" in prompt_str:
            return {
                "summary": "Sarah Johnson completed 10 questions with solid reasoning.",
                "strengths": ["RAG Architecture", "Vector Search"],
                "gaps": ["Observability"],
                "next": ["Practice Prometheus metrics"]
            }
        elif "opening interview question" in prompt_str.lower():
            return {
                "question": "Opening question regarding embeddings?",
                "curriculum_day": 7,
                "topic": "Embeddings Explained",
                "difficulty": "medium"
            }
        else:
            match = re.search(r"Day (\d+):", prompt_str)
            day_num = int(match.group(1)) if match else 7
            return {
                "evaluation": {
                    "correctness": 8, "technical_depth": 8, "reasoning": 8,
                    "practical_understanding": 8, "communication": 8, "overall": 8.0,
                    "strengths": ["Good answer"], "weaknesses": [], "recommended_difficulty": "medium"
                },
                "next_question": {
                    "question": f"Question for day {day_num}?",
                    "curriculum_day": day_num,
                    "topic": f"Topic {day_num}",
                    "difficulty": "medium"
                }
            }

    mock_groq.side_effect = groq_side_effect

    candidates = candidate_service.get_all_candidates()
    candidate = candidates[0]
    session_id = "test-state-machine-q10-finish"

    # Start Session
    start_resp = client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})
    assert start_resp.status_code == 200

    # Submit Answers 1 to 9
    for i in range(1, 10):
        resp = client.post("/api/interview", json={"sessionId": session_id, "message": f"Answer {i}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False
        assert data["showChoice"] is False

    # Submit Answer 10 -> Should trigger showChoice=True
    resp10 = client.post("/api/interview", json={"sessionId": session_id, "message": "Answer 10"})
    assert resp10.status_code == 200
    data10 = resp10.json()
    assert data10["done"] is False
    assert data10["showChoice"] is True
    assert "Your 10-question compulsory interview is complete" in data10["reply"]

    # Candidate chooses 'finish'
    finish_resp = client.post("/api/interview", json={"sessionId": session_id, "action": "finish"})
    assert finish_resp.status_code == 200
    fdata = finish_resp.json()
    assert fdata["done"] is True
    assert fdata["reply"] == "Interview completed."
    assert "feedback" in fdata

@patch("backend.services.groq_service.groq_service.generate_json")
def test_state_machine_continue_to_q15_hard_stop(mock_groq):
    """
    Test 15-question hard limit state machine:
    - Answers 1 to 10 -> Choice -> Candidate chooses 'continue'
    - Answers 11 to 15 -> At Answer 15, automatically triggers completion (done=True)
    """
    import re

    def groq_side_effect(messages, **kwargs):
        prompt_str = str(messages)
        if "Generate final interview assessment" in prompt_str:
            return {
                "summary": "Sarah Johnson completed 15 questions with comprehensive knowledge.",
                "strengths": ["RAG Architecture", "Vector Search", "Multi-Agent"],
                "gaps": ["Observability"],
                "next": ["Review Kubernetes deployments"]
            }
        elif "opening interview question" in prompt_str.lower():
            return {
                "question": "Opening question regarding embeddings?",
                "curriculum_day": 7,
                "topic": "Embeddings Explained",
                "difficulty": "medium"
            }
        else:
            match = re.search(r"Day (\d+):", prompt_str)
            day_num = int(match.group(1)) if match else 7
            return {
                "evaluation": {
                    "correctness": 8, "technical_depth": 8, "reasoning": 8,
                    "practical_understanding": 8, "communication": 8, "overall": 8.0,
                    "strengths": ["Good answer"], "weaknesses": [], "recommended_difficulty": "medium"
                },
                "next_question": {
                    "question": f"Question for day {day_num}?",
                    "curriculum_day": day_num,
                    "topic": f"Topic {day_num}",
                    "difficulty": "medium"
                }
            }

    mock_groq.side_effect = groq_side_effect

    candidates = candidate_service.get_all_candidates()
    candidate = candidates[0]
    session_id = "test-state-machine-q15-hardstop"

    # Start Session
    client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})

    # Answers 1 to 9
    for i in range(1, 10):
        client.post("/api/interview", json={"sessionId": session_id, "message": f"Answer {i}"})
    
    # Answer 10 -> Choice
    resp10 = client.post("/api/interview", json={"sessionId": session_id, "message": "Answer 10"})
    assert resp10.json()["showChoice"] is True

    # Candidate chooses 'continue'
    cont_resp = client.post("/api/interview", json={"sessionId": session_id, "action": "continue"})
    assert cont_resp.status_code == 200
    cdata = cont_resp.json()
    assert cdata["done"] is False
    assert cdata["questionNumber"] == 11

    # Answers 11 to 14
    for i in range(11, 15):
        r = client.post("/api/interview", json={"sessionId": session_id, "message": f"Answer {i}"})
        assert r.status_code == 200
        assert r.json()["done"] is False

    # Answer 15 -> HARD STOP (automatically triggers done=True)
    resp15 = client.post("/api/interview", json={"sessionId": session_id, "message": "Answer 15"})
    assert resp15.status_code == 200
    data15 = resp15.json()
    assert data15["done"] is True
    assert data15["reply"] == "Interview completed."
    assert "feedback" in data15
