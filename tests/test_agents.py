import pytest
from agents.state import MathMentorState
from agents.intent_router_agent import intent_router_agent
from agents.guardrail_agent import guardrail_agent
from config.settings import ALLOWED_TOPICS

def test_intent_router_agent():
    state: MathMentorState = {"topic": "algebra", "agent_trace": []}
    new_state = intent_router_agent(state)
    assert new_state["in_scope"] == True
    assert new_state["route"] == "solve"
    assert new_state["hitl_required"] == False

    state2: MathMentorState = {"topic": "cooking", "agent_trace": []}
    new_state2 = intent_router_agent(state2)
    assert new_state2["in_scope"] == False
    assert new_state2["route"] == "hitl"
    assert new_state2["hitl_required"] == True

def test_guardrail_agent():
    # Test passed
    state: MathMentorState = {
        "topic": "algebra",
        "explanation": "To solve $x-2=0$, we add 2 to both sides: $$x = 2$$. This confirms the root is 2.",
        "solution": "x=2",
        "agent_trace": [],
        "parsed_problem": {"problem_text": "x-2=0"},
        "retrieved_context": [],
        "verification_result": {"confidence": 95, "is_correct": True}
    }
    res = guardrail_agent(state)
    assert res["guardrail_passed"] == True
    
    # Test bad topic
    state2: MathMentorState = {
        "topic": "cooking",
        "explanation": "Here is the math: $x = 2$",
        "solution": "x=2",
        "agent_trace": [],
        "parsed_problem": {},
        "retrieved_context": [],
        "verification_result": {}
    }
    res2 = guardrail_agent(state2)
    assert res2["guardrail_passed"] == False
    assert any("cooking" in issue.lower() for issue in res2["guardrail_issues"])

    # Test bad latex
    state3: MathMentorState = {
        "topic": "algebra",
        "explanation": "Here is the math: $x = 2",
        "solution": "x=2",
        "agent_trace": [],
        "parsed_problem": {},
        "retrieved_context": [],
        "verification_result": {}
    }
    res3 = guardrail_agent(state3)
    assert res3["guardrail_passed"] == False
    
    # Test empty solution
    state4: MathMentorState = {
        "topic": "algebra",
        "explanation": "$x = 2$",
        "solution": "",
        "agent_trace": [],
        "parsed_problem": {},
        "retrieved_context": [],
        "verification_result": {}
    }
    res4 = guardrail_agent(state4)
    assert res4["guardrail_passed"] == False
