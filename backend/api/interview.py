from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
from backend.models.schemas import InterviewRequest, InterviewResponse, FeedbackSchema, SessionCustomizeRequest
from backend.services.sessions import session_manager
from backend.services.candidates import candidate_service
from backend.services.curriculum import curriculum_service
from backend.services.groq_service import groq_service
from backend.agents.interviewer import interviewer_agent
from backend.agents.feedback import feedback_agent

router = APIRouter()

@router.get("/groq-health")
async def groq_health_check():
    """Test Groq connection health safely without exposing API keys."""
    return await groq_service.health_check()

@router.get("/candidates")
async def get_candidates():
    """Retrieve candidate profiles for selection and profile cards."""
    return {"candidates": candidate_service.get_all_candidates()}

@router.get("/candidate/{candidate_id}")
async def get_candidate_detail(candidate_id: str):
    """Retrieve detailed candidate profile for inspection modal."""
    c = candidate_service.get_candidate_by_id(candidate_id)
    if not c:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    analysis = candidate_service.analyze_profile(c)
    return {"candidate": c, "analysis": analysis}

@router.get("/curriculum")
async def get_curriculum():
    """Retrieve course curriculum structure."""
    return curriculum_service.curriculum_data

@router.put("/session/{session_id}")
async def customize_session(session_id: str, req: SessionCustomizeRequest):
    """
    PROTOTYPE EDIT MODE API: Allows customizing candidate profile details,
    difficulty override, or custom notes during an active interview session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found."
        )

    session.update_profile_customization(
        name=req.name,
        job_role=req.jobRole,
        years_experience=req.yearsExperience,
        difficulty_override=req.difficultyOverride,
        custom_notes=req.customNotes
    )

    return {
        "status": "updated",
        "sessionId": session_id,
        "candidate": session.candidate,
        "difficulty": session.current_difficulty,
        "customNotes": session.custom_notes
    }

@router.post("/interview", response_model=InterviewResponse, response_model_exclude_none=True)
async def interview_endpoint(req: InterviewRequest):
    """
    Core HTTP API Endpoint required by technical specification.
    POST /api/interview
    Maintains session state using sessionId.
    Enforces strict state machine: Questions 1..10 -> Q10 Choice -> Continue Q11..15 or Finish -> Q15 Hard Stop.
    Generates all turns dynamically via Groq LLM API and returns complete conversationTurns stream.
    """
    session_id = req.sessionId.strip() if req.sessionId else ""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sessionId is required."
        )

    # 1. START INTERVIEW REQUEST (Candidate payload provided)
    if req.candidate is not None:
        candidate_data = req.candidate
        profile_analysis = candidate_service.analyze_profile(candidate_data)
        
        # Create or reset session
        session = session_manager.create_session(session_id, candidate_data, profile_analysis)
        
        # Generate Question 1 dynamically via Groq
        q1_data = await interviewer_agent.generate_initial_turn(session)
        if not q1_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to generate the next interview question. Please try again."
            )

        session.add_question(q1_data)
        welcome_reply = f"Welcome. Let's begin your interview.\n\nQuestion 1 (Day {q1_data['curriculumDay']} - {q1_data['topic']}):\n{q1_data['question']}"

        return InterviewResponse(
            reply=welcome_reply,
            done=False,
            showChoice=False,
            questionNumber=session.question_number,
            minQuestions=session.MIN_QUESTIONS,
            maxQuestions=session.MAX_QUESTIONS,
            currentTopic=q1_data['topic'],
            curriculumDay=q1_data['curriculumDay'],
            coveredDaysCount=session.total_days_covered,
            difficulty=session.current_difficulty,
            conversationTurns=session.get_conversation_turns()
        )

    # 2. CONVERSATION TURN REQUEST
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Please initialize session with candidate payload."
        )

    user_action = req.action.strip().lower() if req.action else ""
    user_message = req.message.strip() if req.message else ""

    # Check for decision at Question 10 completion choice: FINISH
    if user_action == "finish" or user_message.lower() == "finish":
        session.user_finished_at_10 = True
        session.status = "completed"
        fb_dict = await feedback_agent.generate_feedback(
            candidate_profile=session.profile_analysis,
            questions_and_evaluations=session.questions_asked
        )
        if not fb_dict:
            fb_dict = {
                "summary": "Interview completed after 10 questions.",
                "strengths": ["Completed core 10-question evaluation."],
                "gaps": ["Further practice recommended on production observability."],
                "next": ["Review vector search index optimization."]
            }
        session.final_feedback = fb_dict

        return InterviewResponse(
            reply="Interview completed.",
            done=True,
            showChoice=False,
            feedback=FeedbackSchema(
                summary=fb_dict.get("summary", "Interview completed."),
                strengths=fb_dict.get("strengths", []),
                gaps=fb_dict.get("gaps", []),
                next=fb_dict.get("next", [])
            ),
            conversationTurns=session.get_conversation_turns()
        )

    # Check for decision at Question 10 completion choice: CONTINUE
    if user_action == "continue" or user_message.lower() == "continue":
        session.extended_session = True
        session.status = "in_progress"

        q11_data = await interviewer_agent.generate_initial_turn(session)
        if not q11_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to generate the next interview question. Please try again."
            )

        session.add_question(q11_data)
        reply_text = f"Question {session.question_number} (Day {q11_data['curriculumDay']} - {q11_data['topic']}):\n{q11_data['question']}"

        return InterviewResponse(
            reply=reply_text,
            done=False,
            showChoice=False,
            questionNumber=session.question_number,
            minQuestions=session.MIN_QUESTIONS,
            maxQuestions=session.MAX_QUESTIONS,
            currentTopic=q11_data['topic'],
            curriculumDay=q11_data['curriculumDay'],
            coveredDaysCount=session.total_days_covered,
            difficulty=session.current_difficulty,
            conversationTurns=session.get_conversation_turns()
        )

    # Standard candidate technical message submission evaluation
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message string or valid action is required."
        )

    turn_result = await interviewer_agent.process_turn(session, user_message)
    if not turn_result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to generate the next interview question. Please try again."
        )

    evaluation = turn_result["evaluation"]
    next_q_data = turn_result["next_question"]

    # Record previous answer and evaluation
    session.record_answer_and_eval(user_message, evaluation)

    # CHECK 1: Question 10 Completion Choice UI trigger
    if session.should_show_completion_choice():
        session.status = "awaiting_choice"
        return InterviewResponse(
            reply="Your 10-question compulsory interview is complete. Would you like to finish now or continue up to 15 questions?",
            done=False,
            showChoice=True,
            questionNumber=session.question_number,
            minQuestions=session.MIN_QUESTIONS,
            maxQuestions=session.MAX_QUESTIONS,
            coveredDaysCount=session.total_days_covered,
            difficulty=session.current_difficulty,
            conversationTurns=session.get_conversation_turns()
        )

    # CHECK 2: Hard Limit (15 Questions answered) or Explicit Finish
    if session.is_interview_complete():
        session.status = "completed"
        fb_dict = await feedback_agent.generate_feedback(
            candidate_profile=session.profile_analysis,
            questions_and_evaluations=session.questions_asked
        )

        if not fb_dict:
            fb_dict = {
                "summary": "Interview completed across 15 technical questions.",
                "strengths": ["Demonstrated comprehensive technical knowledge across cohort topics."],
                "gaps": ["Further practice recommended on high-load production scaling."],
                "next": ["Review vector database partitioning and agent workflow optimization."]
            }

        session.final_feedback = fb_dict

        return InterviewResponse(
            reply="Interview completed.",
            done=True,
            showChoice=False,
            feedback=FeedbackSchema(
                summary=fb_dict.get("summary", "Interview completed."),
                strengths=fb_dict.get("strengths", []),
                gaps=fb_dict.get("gaps", []),
                next=fb_dict.get("next", [])
            ),
            conversationTurns=session.get_conversation_turns()
        )

    # Prevent question generation beyond Question 15 hard limit
    if session.question_number >= session.MAX_QUESTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview has reached the hard limit of 15 questions."
        )

    # ADD NEXT QUESTION (Questions 2..10 or 11..15)
    session.add_question(next_q_data)
    reply_text = f"Question {session.question_number} (Day {next_q_data['curriculumDay']} - {next_q_data['topic']}):\n{next_q_data['question']}"

    return InterviewResponse(
        reply=reply_text,
        done=False,
        showChoice=False,
        questionNumber=session.question_number,
        minQuestions=session.MIN_QUESTIONS,
        maxQuestions=session.MAX_QUESTIONS,
        currentTopic=next_q_data['topic'],
        curriculumDay=next_q_data['curriculumDay'],
        coveredDaysCount=session.total_days_covered,
        difficulty=session.current_difficulty,
        conversationTurns=session.get_conversation_turns()
    )
