"""THROWAWAY diagnostic -- delete when done."""
from dotenv import load_dotenv

load_dotenv()

from app.llm_backend import get_backend

backend = get_backend()

raw_text = backend.complete_text('Return ONLY a JSON object: {"test": "hello"}')
print("TEXT CALL RAW RESPONSE:")
print(repr(raw_text))
print()

raw_vision = backend.complete_vision(
    "Return ONLY a JSON object with this key: brand_name (the brand name visible on this label)",
    "sample_labels/old_tom_bourbon_clean.jpg",
)
print("VISION CALL RAW RESPONSE:")
print(repr(raw_vision))