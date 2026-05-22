import marimo

__generated_with = "0.23.8"
app = marimo.App(layout_file="layouts/app.grid.json")


@app.cell
def viewer_widget():
    import marimo as mo
    from src.ui.view_viewer import render_viewer_view

    rgb_ui = mo.md("Platzhalter RGB")
    segment_ui = mo.md("Platzhalter Segments")
    class_ui = mo.md("Platzhalter Class")

    viewer_tab_ui = render_viewer_view(
        rgb_ui=rgb_ui,
        segment_ui=segment_ui,
        class_ui=class_ui
    )

    viewer_tab_ui
    return (mo,)


@app.cell
def _():
    from src.ui.view_llm import get_llm_form
    llm_form = get_llm_form()
    return (llm_form,)


@app.cell
def llm_widget(llm_form, mo):
    from src.core.llm_client import generate_response
    from src.ui.view_llm import render_llm_view

    response_text = ""

    if llm_form.value:
        with mo.status.spinner("LLM calculate probabilities..."):
            response_text = generate_response(llm_form.value)

    llm_tab_ui = render_llm_view(
        llm_form=llm_form, 
        response_text=response_text
    )

    llm_tab_ui
    return


if __name__ == "__main__":
    app.run()
