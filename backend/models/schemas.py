from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MissionSchema(BaseModel):
    day: int
    title: str
    passed: Optional[bool] = True
    attempts: Optional[int] = 1
    skipped: Optional[bool] = False

class SignalsSchema(BaseModel):
    commitDays: int = 0
    missionsCompleted: int = 0
    missionsFirstTry: int = 0

class MemberSchema(BaseModel):
    id: str
    name: str
    jobRole: str
    yearsExperience: int
    education: str
    status: Optional[str] = "COMPLETED"

class CandidateSchema(BaseModel):
    member: MemberSchema
    missions: List[MissionSchema] = []
    signals: Optional[SignalsSchema] = None

class FeedbackSchema(BaseModel):
    summary: str
    strengths: List[str]
    gaps: List[str]
    next: List[str]

class SessionCustomizeRequest(BaseModel):
    name: Optional[str] = None
    jobRole: Optional[str] = None
    yearsExperience: Optional[int] = None
    difficultyOverride: Optional[str] = None
    customNotes: Optional[str] = None

class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    action: Optional[str] = None  # "finish" or "continue"

class InterviewResponse(BaseModel):
    reply: str
    done: bool = False
    showChoice: bool = False
    feedback: Optional[FeedbackSchema] = None
    questionNumber: Optional[int] = None
    minQuestions: int = 10
    maxQuestions: int = 15
    currentTopic: Optional[str] = None
    curriculumDay: Optional[int] = None
    coveredDaysCount: Optional[int] = None
    difficulty: Optional[str] = None
    conversationTurns: Optional[List[Dict[str, Any]]] = None
    liveMetrics: Optional[Dict[str, int]] = None
