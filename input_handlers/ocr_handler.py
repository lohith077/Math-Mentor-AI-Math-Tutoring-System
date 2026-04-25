"""Vision LLM OCR + optional Tesseract fallback for math image extraction."""
import re, base64
from PIL import Image
from groq import Groq
import io

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
from config.settings import GROQ_API_KEY, MATH_MODEL, MATH_MODEL_BACKUPS

client = Groq(api_key=GROQ_API_KEY)

# ── Vision LLM approach (primary for math) ──

VISION_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

def extract_text_with_vision(image_bytes: bytes) -> dict:
    """Use a vision-capable LLM to extract math from an image."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = """You are a math OCR specialist. Your job is to extract ALL text and mathematical content visible in this image — faithfully and completely.

### RULES:
1. Extract EVERY word, number, and mathematical symbol exactly as shown in the image.
2. For mathematical notation, use LaTeX inline format:
   - Square roots: $\\sqrt{...}$
   - Fractions: $\\frac{numerator}{denominator}$
   - Integrals: $\\int_{lower}^{upper}$
   - Greek letters: $\\theta$, $\\pi$, $\\alpha$, etc.
   - Exponents: $x^{n}$
   - Subscripts: $x_{n}$
3. Keep all surrounding text exactly as written (e.g., "Question 13:", "equals", "find the value of", "In a box of 10 bulbs...").
4. Do NOT skip any part of the image. Extract everything — headings, options, labels.
5. Do NOT explain or solve the problem. Only extract.
6. Output ONLY the extracted content — nothing else."""

    last_error = None
    for model in VISION_MODELS:
        try:
            print(f"[DEBUG] 👁️ Attempting Vision OCR with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=500
            )
            text = response.choices[0].message.content.strip()
            print(f"[DEBUG] 👁️ Vision OCR result: {text[:100]}...")
            return {"text": text, "confidence": 95.0, "method": "vision_llm", "model": model}
        except Exception as e:
            print(f"[WARNING] ⚠️ Vision OCR failed with {model}: {e}")
            last_error = e
            continue
    
    # If all vision models fail, return None to trigger Tesseract fallback
    print(f"[WARNING] ⚠️ All vision models failed. Falling back to Tesseract.")
    return None


# ── Tesseract approach (fallback) ──

def math_ocr_cleanup(text: str) -> str:
    """Heuristic cleanup for Tesseract math artifacts."""
    # NOTE: NO re.IGNORECASE — we handle V and v explicitly
    replacements = [
        (r"V¥", r"sqrt theta"),
        (r"cos\s*[d¥]\b", r"cos theta"),
        (r"Cos\s*[d¥]\b", r"cos theta"),
        (r"sin\s*[d¥]\b", r"sin theta"),
        (r"Sin\s*[d¥]\b", r"sin theta"),
        (r"tan\s*[d¥]\b", r"tan theta"),
        (r"Tan\s*[d¥]\b", r"tan theta"),
        (r"(?<![a-zA-Z])V(?=\()", r"sqrt"),
        (r"(?<=\d)V\b", r"*sqrt"),
        (r"¥", r"theta"),
        (r"fe\s+", r"integral "),
        (r"d0\b", r"dtheta"),
        (r"(?<=[^\w])d\b", r"dtheta"),
        (r"__+", r"/"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def extract_text_from_image(image_bytes: bytes) -> dict:
    """Extract text from image: Vision LLM first, Tesseract fallback."""
    
    # 1. Try Vision LLM (primary — much better for math)
    vision_result = extract_text_with_vision(image_bytes)
    if vision_result:
        return vision_result
    
    # 2. Fallback to Tesseract (only if installed)
    if TESSERACT_AVAILABLE:
        print("[DEBUG] 📷 Using Tesseract OCR fallback...")
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("L")
        
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image)
        cleaned_text = math_ocr_cleanup(text)
        
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {"text": cleaned_text.strip(), "confidence": round(mean_confidence, 1), "method": "tesseract", "raw_data": data}
    
    # 3. Both failed
    return {"text": "[Error: Could not extract text from image. Please try again or type the problem manually.]", "confidence": 0, "method": "error"}
