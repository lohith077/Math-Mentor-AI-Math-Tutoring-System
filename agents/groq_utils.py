import groq
import datetime
from langsmith import traceable

@traceable(run_type="llm", name="Groq Fallback Wrapper")
def groq_completion_with_fallback(client, model, messages, backups=None, **kwargs):
    """
    Executes a Groq completion. If a RateLimitError (429), 
    BadRequestError (400 - decommissioned/invalid model), or
    NotFoundError (404) occurs, it rotates through backup models.
    """
    models_to_try = [model] + (backups or [])
    last_error = None

    for attempt_model in models_to_try:
        try:
            print(f"[DEBUG] 🚀 Attempting completion with model: {attempt_model}")
            response = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                **kwargs
            )
            return response, attempt_model
        except groq.RateLimitError as e:
            print(f"[WARNING] ⚠️ Rate limit hit for {attempt_model}. Trying next backup...")
            last_error = e
            continue
        except groq.BadRequestError as e:
            print(f"[WARNING] ⚠️ Bad request / decommissioned model {attempt_model}: {e}. Trying next backup...")
            last_error = e
            continue
        except groq.NotFoundError as e:
            print(f"[WARNING] ⚠️ Model not found {attempt_model}: {e}. Trying next backup...")
            last_error = e
            continue
        except Exception as e:
            print(f"[ERROR] ❌ Unexpected error with {attempt_model}: {e}")
            raise e

    print("[ERROR] 💀 All models exhausted (including backups).")
    raise last_error
