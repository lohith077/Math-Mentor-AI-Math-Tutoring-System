"""
① Input Handler Agent
Type: Rule-Based | No LLM | Tools: Whisper, Tesseract
"""
from input_handlers.ocr_handler import extract_text_from_image
from input_handlers.asr_handler import transcribe_audio
from input_handlers.text_handler import normalize_text
from agents.state import MathMentorState
from config.settings import OCR_CONFIDENCE_THRESHOLD
from langsmith import traceable
import datetime

@traceable(run_type="chain", name="Input Handler Agent")
def input_handler_agent(state: MathMentorState) -> MathMentorState:
    print(f"\n[DEBUG] 📥 Input Handler Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    trace.append({"agent": "InputHandler", "action": "start", "timestamp": str(datetime.datetime.now())})
    
    input_type = state["input_type"]
    raw = state["raw_input"]
    
    # Check if we already have extracted text (e.g., from a HITL correction)
    # If so, and it was a media input, we skip re-extraction to avoid overwriting user edits
    if state.get("extracted_text") and input_type in ["image", "audio"]:
        print(f"[DEBUG] 📥 Input Handler Agent: Using existing extracted text (Human-Approved).")
        extracted = state["extracted_text"]
        confidence = 100.0 # Force 100% confidence for user-corrected text
    else:
        if input_type == "image":
            result = extract_text_from_image(raw)
            extracted = result["text"]
            confidence = result["confidence"]
        elif input_type == "audio":
            result = transcribe_audio(raw)
            extracted = result["text"]
            confidence = result["confidence"]
        else:  # text
            extracted = normalize_text(raw)
            confidence = 100.0
    
    hitl_required = confidence < OCR_CONFIDENCE_THRESHOLD
    reason = f"Extraction confidence {confidence:.1f}% < threshold {OCR_CONFIDENCE_THRESHOLD}%" if hitl_required else ""
    
    trace.append({"agent": "InputHandler", "action": f"extracted text (confidence={confidence:.1f}%)", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 📥 Input Handler Agent finished in {duration:.2f}s")
    
    return {
        **state,
        "extracted_text": extracted,
        "extraction_confidence": confidence,
        "hitl_required": hitl_required,
        "hitl_reason": reason,
        "hitl_trigger_agent": "InputHandler" if hitl_required else "",
        "agent_trace": trace
    }
