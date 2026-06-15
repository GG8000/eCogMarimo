"""Ask Google's Gemma model about the analysed scene.

:func: describe_scene turns the classification result into a short text summary
(how many segments and how much area each class covers). :func: ask sends that
summary plus the user's question to the model and returns the answer.

If the package or API key is missing, :func: ask returns a readable message
instead of raising an error, so the notebook keeps working offline. Set the API key in
the environment variable GEMINI_API_KEY.
"""

import os

import numpy as np

MODEL_NAME = "gemma-4-26b-a4b-it"

def describe_scene(segments, classification) -> str:
    """Summarise the classification as plain text for the model to read."""
    names = classification.predictions
    total_pixels = segments.labels.size
    # How many pixels each segment has, so we can add up area per class.
    pixels_per_segment = np.bincount(segments.labels.ravel(), minlength=segments.count)

    seg_counts: dict[str, int] = {}
    area: dict[str, int] = {}
    for seg_id, name in enumerate(names):
        seg_counts[name] = seg_counts.get(name, 0) + 1
        area[name] = area.get(name, 0) + int(pixels_per_segment[seg_id])

    lines = ["Classification of one aerial harbour tile:"]
    for name in sorted(seg_counts):
        share = area[name] / total_pixels * 100
        lines.append(f"- {name}: {seg_counts[name]} segments, {share:.1f}% of the area")
    return "\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    """Combine the scene description and the user's question into one prompt."""
    parts = ["You analyse an aerial harbour scene."]
    if context:
        parts.append(context)
    parts.append(f"Question: {question}")
    parts.append("Answer concisely, using only the information above, and explain your reasoning.")
    return "\n\n".join(parts)


def ask(question: str, context: str = "") -> str:
    """Answer question about the scene using Gemma, with context as background.

    Falls back to a short readable message (instead of raising) when the package
    or API key is missing, or when the API call fails for any reason.
    """
    prompt = build_prompt(question, context)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "LLM offline: set GEMINI_API_KEY to enable answers.\n\nPrompt:\n\n" + prompt

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        reply = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return reply.text
    except ImportError:
        return "LLM offline: install 'google-genai' to enable answers."
    except Exception as error:
        return f"LLM unavailable: {error}"
