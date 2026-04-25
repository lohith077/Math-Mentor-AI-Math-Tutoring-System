"""
Math Mentor — Streamlit UI
Full implementation with all features.
"""
import streamlit as st
import tempfile, os, json, io
from dotenv import load_dotenv

# Load environment before anything else for tracing
load_dotenv()

from agents.graph import math_mentor_graph
from agents.state import MathMentorState
from memory.memory_store import (
    initialize_db, save_solved_problem, save_feedback, 
    save_hitl_correction, 
    get_flagged_problems, get_hitl_history
)
from memory.similarity_search import add_problem_to_memory
from rag.indexer import index_knowledge_base
from langsmith import traceable
from agents.feedback_agent import feedback_agent
import datetime

# ── App config ──
st.set_page_config(page_title="Math Mentor", page_icon="🧮", layout="wide")


# ── Initialize on first run ──
@st.cache_resource
def initialize():
    initialize_db()
    try:
        index_knowledge_base()
    except Exception as e:
        st.warning(f"Knowledge base indexing: {e}")
    return True

initialize()

# ── Session state ──
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False

# ── Header ──
st.title("🧮 Math Mentor")
st.caption("JEE-style math problem solver · RAG + Multi-Agent · HITL")
st.divider()

# ── Input Section ──
st.subheader("📥 Input")
input_mode = st.radio("Input mode", ["Text", "Image", "Audio"], horizontal=True)

raw_input = None
input_type = input_mode.lower()
    
if input_mode == "Text":
    raw_input = st.text_area("Type your math problem", height=140,
                              placeholder="e.g. Solve 2x² + 3x - 5 = 0")
elif input_mode == "Image":
    from streamlit_paste_button import paste_image_button as pbi
    
    img_tab1, img_tab2, img_tab3 = st.tabs(["📋 Paste from Clipboard", "📁 Upload File", "📷 Camera"])
    
    with img_tab1:
        paste_result = pbi("📋 Click here, then Ctrl+V / ⌘+V to paste image")
        if paste_result.image_data is not None:
            st.image(paste_result.image_data, caption="Pasted image", use_container_width=True)
            buf = io.BytesIO()
            paste_result.image_data.save(buf, format="PNG")
            raw_input = buf.getvalue()
    
    with img_tab2:
        uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
            raw_input = uploaded.read()
    
    with img_tab3:
        camera_img = st.camera_input("Take a photo of your math problem")
        if camera_img:
            raw_input = camera_img.read()
else:  # Audio
    from input_handlers.asr_handler import transcribe_audio
    audio_file = st.audio_input("Record your math problem")
    
    if audio_file:
        raw_input = audio_file.read()
        with st.spinner("🎙️ Transcribing..."):
            res = transcribe_audio(raw_input)
            st.success(f"**Transcription:** {res['text']}")
            st.caption(f"Confidence: {res['confidence']}%")
    
    with st.expander("Or upload an existing audio file"):
        uploaded = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"])
        if uploaded:
            st.audio(uploaded)
            raw_input = uploaded.read()
            if not audio_file:
                with st.spinner("📑 Transcribing upload..."):
                    res = transcribe_audio(raw_input)
                    st.success(f"**Transcription:** {res['text']}")

solve_btn = st.button("🚀 Solve", type="primary", use_container_width=True, disabled=not raw_input)

# ── HITL Panel ──
if st.session_state.hitl_pending and st.session_state.pipeline_state:
    state = st.session_state.pipeline_state
    hitl_msg = state.get("hitl_response", {}).get("friendly_message", "Please review and correct the input.")
    
    st.warning(f"⚠️ **Human Review Needed** ({state.get('hitl_trigger_agent', 'System')})")
    
    # Show the specific reason/error
    reason = state.get('hitl_reason', '')
    if reason:
        st.error(f"**Issue:** {reason}")
        
    # Show the tutor's friendly message
    st.info(hitl_msg)
    
    # QUIZ MODE: Smart Suggestions
    options = state.get("hitl_response", {}).get("options")
    if options:
        st.subheader("💡 Suggestions (Quick Fix)")
        cols = st.columns(len(options))
        for i, opt in enumerate(options):
            if cols[i].button(opt, key=f"quiz_opt_{i}"):
                # User chose a suggestion
                save_hitl_correction(
                    state.get("hitl_trigger_agent", ""), 
                    state.get("hitl_reason", ""),
                    state.get("extracted_text", ""), 
                    opt, 
                    "suggestion_click"
                )
                
                # Implicit Feedback: Analyze WHY HITL was needed
                try:
                    analysis = feedback_agent(
                        comment=f"HITL Suggestion Clicked: '{opt}' chosen over '{state.get('extracted_text','')}'",
                        problem_text=opt,
                        solution="N/A (HITL Phase)"
                    )
                    save_feedback(
                        problem_id=0, # No problem ID yet as it hasn't finished
                        feedback_type="hitl_implicit",
                        comment=f"Correction from {state.get('hitl_trigger_agent')} loop.",
                        category=analysis.get("category", "Technical"),
                        sentiment="Negative", # HITL usually implies a failure to parse/verify correctly
                        is_actionable=True
                    )
                except:
                    pass
                new_state = {**state, "extracted_text": opt, "hitl_required": False,
                             "hitl_response": {"status": "approved"}}
                
                with st.spinner(f"🧠 Applying suggestion: '{opt}'..."):
                    try:
                        config = {"run_name": "Math Mentor Graph Resume", "metadata": {"status": "hitl_suggestion"}}
                        final_state = math_mentor_graph.invoke(new_state, config=config)
                        st.session_state.pipeline_state = final_state
                        st.session_state.hitl_pending = final_state.get("hitl_required", False)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resuming: {e}")

    with st.expander("✏️ Edit & Continue"):
        edited_text = st.text_area("Correct the extracted text:",
                                    value=state.get("extracted_text", ""),
                                    key="hitl_edit")
        col_a, col_b, col_c = st.columns(3)
        
        if col_a.button("✅ Approve & Continue"):
            save_hitl_correction(
                state.get("hitl_trigger_agent", ""), 
                state.get("hitl_reason", ""),
                state.get("extracted_text", ""), 
                edited_text, 
                "approve"
            )

            # Implicit Feedback: Analyze manual correction
            try:
                analysis = feedback_agent(
                    comment=f"Manual HITL Correction: '{edited_text}' replacing '{state.get('extracted_text','')}'",
                    problem_text=edited_text,
                    solution="N/A (HITL Phase)"
                )
                save_feedback(
                    problem_id=0,
                    feedback_type="hitl_implicit",
                    comment=f"Manual edit during {state.get('hitl_trigger_agent')} phase.",
                    category=analysis.get("category", "Technical"),
                    sentiment="Negative",
                    is_actionable=True
                )
            except:
                pass
            new_state = {**state, "extracted_text": edited_text, "hitl_required": False,
                         "hitl_response": {"status": "approved"}}
            
            with st.spinner("🧠 Resuming Math Mentor pipeline..."):
                try:
                    # Explicitly pass config for LangSmith to capture the graph as a primary run
                    config = {"run_name": "Math Mentor Graph Resume", "metadata": {"status": "hitl_resumed"}}
                    final_state = math_mentor_graph.invoke(new_state, config=config)
                    st.session_state.pipeline_state = final_state
                    
                    if final_state.get("hitl_required"):
                        st.session_state.hitl_pending = True
                    else:
                        st.session_state.hitl_pending = False
                        
                        # Save the run outcome (Success or Failure)
                        problem_id = save_solved_problem({
                            "input_type": final_state.get("input_type", "text"),
                            "original_input": str(final_state.get("raw_input", ""))[:500],
                            "parsed_problem": final_state.get("parsed_problem", {}),
                            "topic": final_state.get("topic", ""),
                            "solution": final_state.get("solution", ""),
                            "solution_steps": final_state.get("solution_steps", []),
                            "tools_used": final_state.get("tools_used", []),
                            "verification_confidence": final_state.get("verification_result", {}).get("confidence", 0),
                            "is_correct": final_state.get("verification_result", {}).get("is_correct", True),
                            "explanation": final_state.get("explanation", "")
                        })
                        
                        # Only add to memory (for future cache hits) if it was actually correct
                        if final_state.get("solution") and final_state.get("verification_result", {}).get("is_correct"):
                            problem_text = final_state.get("parsed_problem", {}).get("problem_text", "")
                            add_problem_to_memory(problem_id, problem_text, final_state.get("topic",""), final_state.get("solution",""))
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error upon resume: {e}")
        
        if col_b.button("❌ Reject"):
            st.session_state.hitl_pending = False
            st.session_state.pipeline_state = None
            st.rerun()
        
        if col_c.button("🔄 Re-run from scratch"):
            st.session_state.hitl_pending = False
            st.session_state.pipeline_state = None
            st.rerun()

# ── Run pipeline ──
if solve_btn and raw_input:
    initial_state: MathMentorState = {
        "raw_input": raw_input,
        "input_type": input_type,
        "extracted_text": "",
        "extraction_confidence": 100.0,
        "parsed_problem": {},
        "topic": "",
        "needs_clarification": False,
        "route": "",
        "in_scope": True,
        "memory_hit": False,
        "memory_matches": [],
        "retrieved_context": [],
        "solution": "",
        "solution_steps": [],
        "tools_used": [],
        "verification_result": {},
        "explanation": "",
        "guardrail_passed": False,
        "guardrail_issues": [],
        "hitl_required": False,
        "hitl_reason": "",
        "hitl_trigger_agent": "",
        "hitl_response": {},
        "agent_trace": [],
        "problem_id": None,
        "feedback": {},
        "chat_history": [],
        "is_followup": False,
        "followup_query": "",
        "followup_suggestions": [],
        "error": None
    }
    
    with st.spinner("🧠 Running Math Mentor pipeline..."):
        try:
            # Native LangGraph invoke call handling environment-based tracing
            # Passing run_name in config helps LangSmith identify the graph execution
            config = {"run_name": "Math Mentor Graph Initial", "metadata": {"input_mode": input_mode}}
            final_state = math_mentor_graph.invoke(initial_state, config=config)
            st.session_state.pipeline_state = final_state
            
            if final_state.get("hitl_required"):
                st.session_state.hitl_pending = True
            else:
                st.session_state.hitl_pending = False
                
                # Save the run outcome (Success or Failure)
                problem_id = save_solved_problem({
                    "input_type": input_type,
                    "original_input": str(raw_input)[:500],
                    "parsed_problem": final_state.get("parsed_problem", {}),
                    "topic": final_state.get("topic", ""),
                    "solution": final_state.get("solution", ""),
                    "solution_steps": final_state.get("solution_steps", []),
                    "tools_used": final_state.get("tools_used", []),
                    "verification_confidence": final_state.get("verification_result", {}).get("confidence", 0),
                    "is_correct": final_state.get("verification_result", {}).get("is_correct", True),
                    "explanation": final_state.get("explanation", "")
                })
                
                # Only add to memory (for future cache hits) if it was actually correct
                if final_state.get("solution") and final_state.get("verification_result", {}).get("is_correct"):
                    problem_text = final_state.get("parsed_problem", {}).get("problem_text", "")
                    add_problem_to_memory(problem_id, problem_text, final_state.get("topic",""), final_state.get("solution",""))
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# ── Results ──
# ── Results ──
state = st.session_state.pipeline_state
if state and not st.session_state.hitl_pending:
    
    # Extracted text preview
    if state.get("extracted_text"):
        with st.expander("📋 Extracted Input", expanded=False):
            st.text(state["extracted_text"])
            
            # Extraction Confidence (for images/audio)
            if state.get("input_type") in ["image", "audio"]:
                conf = state.get("extraction_confidence", 100)
                st.progress(int(conf)/100, text=f"Extraction confidence: {conf:.1f}%")
            
            # Semantic Confidence (from Parser)
            sem_conf = state.get("semantic_confidence", 0) * 100
            st.progress(int(sem_conf)/100, text=f"Mathematical Clarity (Semantic Confidence): {sem_conf:.1f}%")
    
    # Memory reuse notice
    if state.get("memory_hit"):
        st.info(f"♻️ Found similar solved problem in memory (similarity: {state['memory_matches'][0]['similarity']})")
    
    # Confidence indicator
    verif = state.get("verification_result", {})
    if verif:
        conf = verif.get("confidence", 0)
        st.metric("Verifier Confidence", f"{conf}%", delta="✅ Correct" if verif.get("is_correct") else "⚠️ Issues found")
    
    # Guardrail status
    if not state.get("guardrail_passed"):
        st.error(f"⛔ Guardrail issues: {', '.join(state.get('guardrail_issues', []))}")
    
    # Raw solution output
    if state.get("solution"):
        st.subheader("💡 Answer")
        st.info(state["solution"])
    elif state.get("explanation"):
        pass
    else:
        st.warning("⚠️ Pipeline finished but no specific solution was generated.")
    
    # Final explanation
    if state.get("explanation"):
        st.subheader("📖 Step-by-Step Explanation")
        st.markdown(state["explanation"])
        
    # Consolidated Feedback Section (always visible when solution exists)
    if state.get("solution") or state.get("explanation"):
        st.divider()
        st.subheader("📣 Feedback")
        st.write("Was this solution helpful? (Optional: Tell us more)")
        
        sentiment = st.radio("How was the experience?", ["Helpful", "Issues / Need Live Tutor"], horizontal=True, label_visibility="collapsed")
        
        user_comment = st.text_area("Additional details (optional)", height=70, placeholder="e.g. 'Step 2 was confusing' or 'Too much technical jargon'")
        
        if st.button("📤 Submit Feedback", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing your feedback..."):
                fb_type = "positive" if sentiment == "Helpful" else "negative"
                
                if user_comment:
                    parsed = state.get("parsed_problem", {})
                    problem_text = parsed.get("problem_text", "") if isinstance(parsed, dict) else str(parsed)
                    analysis = feedback_agent(user_comment, problem_text, state.get("solution", ""))
                    category = analysis.get("category", "General")
                    sent_val = analysis.get("sentiment", "Neutral")
                    is_act = analysis.get("is_actionable", False)
                else:
                    user_comment = "No comment provided."
                    category = "General"
                    sent_val = "Positive" if fb_type == "positive" else "Negative"
                    is_act = False if fb_type == "positive" else True
                
                save_feedback(
                    problem_id=state.get("problem_id", 0),
                    feedback_type=fb_type,
                    comment=user_comment,
                    category=category,
                    sentiment=sent_val,
                    is_actionable=is_act
                )
                
                if fb_type == "negative":
                    st.info("Our team and a **live math tutor** have been notified to help you!")
                st.success("Thank you! Your feedback helps us improve.")

    # RAG Sources
    if state.get("retrieved_context"):
        with st.expander(f"📚 Knowledge Sources ({len(state['retrieved_context'])} retrieved)"):
            for chunk in state["retrieved_context"]:
                st.caption(f"**[{chunk['source']}]** — relevance: {chunk.get('relevance_score', 0):.3f}")
                st.text(chunk["text"][:300] + "...")
                st.divider()

    # ── Follow-up Chat Section ──
    st.divider()
    st.subheader("💬 Guided Follow-up")
    st.write("Ask a question about the solution or dive deeper into the concept.")

    # Display Chat History
    chat_history = state.get("chat_history", [])
    for role, text in chat_history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").markdown(text)

    # Fixed follow-up categories (same for all questions)
    suggestions = [
        "📐 What formulas/identities were used?",
        "📖 Give a more detailed explanation",
        "🔗 Explain the logical flow step by step",
        "🔄 Try a similar problem from memory",
        "⚡ Any JEE shortcuts or tips for this?"
    ]
    
    st.write("**Suggested Next Steps:**")
    s_cols = st.columns(3)
    for i, s_query in enumerate(suggestions[:3]):
        if s_cols[i].button(s_query, key=f"suggest_{i}"):
            st.session_state.followup_trigger = s_query
    
    s_cols2 = st.columns(2)
    for i, s_query in enumerate(suggestions[3:5]):
        if s_cols2[i].button(s_query, key=f"suggest_{i+3}"):
            st.session_state.followup_trigger = s_query

    # Manual Follow-up Input
    ff_query = st.chat_input("Ask a follow-up question...")
    
    # Logic to trigger follow-up
    selected_query = ff_query or st.session_state.get("followup_trigger")
    
    if selected_query:
        if st.session_state.get("followup_trigger"):
            st.session_state.followup_trigger = None
            
        new_state = {
            **state, 
            "is_followup": True, 
            "followup_query": selected_query
        }
        
        with st.spinner("🧠 Math Mentor is thinking..."):
            try:
                config = {"run_name": "Math Mentor Follow-up", "metadata": {"query": selected_query}}
                final_state = math_mentor_graph.invoke(new_state, config=config)
                final_state["is_followup"] = False 
                st.session_state.pipeline_state = final_state
                st.rerun()
            except Exception as e:
                st.error(f"Follow-up error: {e}")

    # Re-check / HITL Trigger
    if st.button("🔄 Request Live Review / Re-solve"):
        if state:
            new_state = {**state, "hitl_required": True, "hitl_reason": "User requested re-check",
                         "hitl_trigger_agent": "User"}
            from agents.hitl_agent import hitl_agent
            new_state = hitl_agent(new_state)
            st.session_state.pipeline_state = new_state
            st.session_state.hitl_pending = True
            st.rerun()
