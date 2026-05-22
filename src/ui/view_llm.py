import marimo as mo

def get_llm_form():
    """Creates only the llm form widget"""
    return mo.ui.text_area(
        label="System Prompt",
        placeholder="Ask the LLM...",
        rows = 4,
        full_width=True
    ).form(submit_button_label="Generate")


def render_llm_view(llm_form, response_text):
    """Create the UI for the Tab"""
    output = mo.md(response_text) if response_text else mo.md("*Answer will appear here...*")
    return mo.vstack([
        mo.md("### 🤖 LLM Assistent"),
        llm_form,
        output
    ])