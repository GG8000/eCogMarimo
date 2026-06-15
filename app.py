import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports and one-time setup. Everything else builds on these names."""
    import os

    import marimo as mo

    from ecog import io, segment, classify, llm, widget

    # How wide (in pixels) the tile images are shown. Kept small so the picture
    # and the table of samples sit side by side comfortably.
    IMAGE_WIDTH = 520

    # Pick up the API key from the original project's .env, if there is one, so
    # the LLM works without setting environment variables by hand.
    _env = io.DATA_ROOT.parent / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _key, _value = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _value.strip())
    return IMAGE_WIDTH, classify, io, llm, mo, segment, widget


@app.cell
def _(mo):
    mo.md("""
    # eCogMarimo · Interactive Sampler

    **Load a tile → set the SLIC sliders and click *Segment* → label segments → click *Classify* → ask the assistant.**
    In the sampler, pick a class and click segments to label them. Click a segment again to remove it.
    Scroll to zoom into the map and drag to pan.
    """)
    return


@app.cell
def _(classify, io, mo, segment):
    """The top controls. Their `.value` drives the whole notebook reactively."""
    tile_choice = mo.ui.dropdown(options=io.TILES, label="Tile")
    method_choice = mo.ui.dropdown(options=segment.METHODS, value="SLIC", label="Segmentation")
    classifier_choice = mo.ui.dropdown(options=classify.METHODS, value="Random Forest", label="Classifier")

    mo.hstack([tile_choice, method_choice, classifier_choice], justify="start", gap=1)
    return classifier_choice, method_choice, tile_choice


@app.cell
def _(mo):
    """SLIC parameters and the button that actually (re)runs the segmentation.

    Nothing is segmented until `segment_button` is clicked, so dragging the
    sliders is cheap.
    """
    n_segments_slider = mo.ui.slider(
        100, 20000, step=100, value=10000, label="Number of segments", show_value=True
    )
    compactness_slider = mo.ui.slider(
        1, 50, step=1, value=10, label="Compactness", show_value=True
    )
    segment_button = mo.ui.run_button(label="Segment")
    return compactness_slider, n_segments_slider, segment_button


@app.cell
def _(mo):
    """The button that runs the classification, shown inside the Classification tab."""
    classify_button = mo.ui.run_button(label="Classify")
    return (classify_button,)


@app.cell
def _(
    IMAGE_WIDTH,
    class_view,
    compactness_slider,
    method_choice,
    mo,
    n_segments_slider,
    sampler,
    segment_button,
    tile,
    tile_choice,
    widget,
):
    """The viewer: the three steps side by side (RGB / Segmentation / Classification).

    This cell depends only on the committed scene, never on the labels, so
    clicking a segment in the sampler does not rebuild the columns (or the widget).
    """
    if tile is None:
        rgb_view = mo.md("Select a tile to begin.")
    else:
        rgb_view = mo.image(widget.image_bytes(tile.image), rounded=True, width=IMAGE_WIDTH)

    if method_choice.value == "SLIC":
        _params = mo.vstack([
            mo.md("**SLIC parameters** — applied when you click *Segment* (SLIC method only)."),
            n_segments_slider,
            compactness_slider,
            segment_button,
        ])
    else: _params = None

    if sampler is None:
        sentence = 'Set the sliders and click **Segment** to create segments.' if _params else method_choice.value + " is not implemented yet."
        _picture = mo.md(sentence)
    else:
        _picture = sampler   # the sampler shows the picture and its samples table
    # Picture first, then the sliders underneath — so the segmentation image
    # lines up with the RGB and Classification images for side-by-side comparison.
    if tile_choice.value == None:
        segmentation_view = mo.md("Please select a Tile first...")
    else:
        segmentation_view = mo.vstack([_picture, _params])

    # The three steps next to each other: RGB left, segmentation middle,
    # classification right.
    mo.hstack(
        [
            mo.vstack([mo.md("### RGB"), rgb_view]),
            mo.vstack([mo.md("### Segmentation"), segmentation_view]),
            mo.vstack([mo.md("### Classification"), class_view]),
        ],
        widths="equal",
        gap=2,
        align="start",
    )
    return


@app.cell
def _(mo):
    """The question box for the assistant."""
    llm_form = mo.ui.text_area(
        placeholder="Ask about the scene...", rows=3, full_width=True
    ).form(submit_button_label="Ask")
    return (llm_form,)


@app.cell
def _(get_classification, llm, llm_form, mo, segments):
    """Send the question (with the real classification summary) to Gemma."""
    _classification = get_classification()
    context = ""
    if segments is not None and _classification is not None:
        context = llm.describe_scene(segments, _classification)

    answer = ""
    if llm_form.value:
        # Show the loader above the textarea (not appended below it) while the
        # answer is generated.
        mo.output.replace(mo.vstack([
            mo.md("### Assistant"),
            mo.status.spinner(title="Asking the assistant..."),
            llm_form,
        ]))
        answer = llm.ask(llm_form.value, context=context)

    mo.vstack([
        mo.md("### Assistant"),
        llm_form,
        mo.md(answer) if answer else mo.md("*The answer will appear here.*"),
    ])
    return


@app.cell
def _(io, tile_choice):
    """Load the chosen tile (fast — reads the small resampled image)."""
    tile = io.load_tile(tile_choice.value) if tile_choice.value else None
    return (tile,)


@app.cell
def _(mo):
    """The committed scene (tile + its segments) and the classification result.

    These live in reactive state so the heavy work only happens on a button
    press: the rest of the notebook reads the committed values, not the live
    sliders.
    """
    get_scene, set_scene = mo.state(None)
    get_classification, set_classification = mo.state(None)
    return get_classification, get_scene, set_classification, set_scene


@app.cell
def _(
    compactness_slider,
    method_choice,
    mo,
    n_segments_slider,
    segment,
    segment_button,
    set_scene,
    tile,
):
    """Segment the tile — but only when the *Segment* button is clicked."""
    if segment_button.value and tile is not None and method_choice.value:
        with mo.status.spinner("Segmenting..."):
            _segments = segment.segment_image(
                tile.image,
                method_choice.value,
                n_segments=n_segments_slider.value,
                compactness=compactness_slider.value,
            )
        set_scene({"tile": tile, "segments": _segments})
    return


@app.cell
def _(get_scene):
    """Unpack the committed scene for everything downstream to use."""
    _scene = get_scene()
    scene_tile = _scene["tile"] if _scene else None
    segments = _scene["segments"] if _scene else None
    return scene_tile, segments


@app.cell
def _(scene_tile, segment, segments):
    """The tile with segment borders drawn on — the base for the clickable view."""
    overlay = None
    if scene_tile is not None and segments is not None:
        overlay = segment.draw_boundaries(scene_tile.image, segments)
    return (overlay,)


@app.cell
def _(segments, set_classification):
    """Drop any old classification whenever a new scene is segmented.

    The labels reset on their own: a new scene rebuilds the sampler widget,
    which starts empty.
    """
    segments  # depend on `segments` so this runs when it changes
    set_classification(None)
    return


@app.cell
def _(classify, overlay, segments, widget):
    """Build the interactive sampler — once per scene, not on every click.

    The widget handles labelling, colour tinting and zooming itself, in the
    browser, and syncs only the collected labels back to Python. Because this
    cell does not depend on the labels, clicking a segment never rebuilds it, so
    the picture does not reload and the zoom level stays put.
    """
    sampler = None
    if overlay is not None and segments is not None:
        # "100%" makes the widget fill (and shrink to) its column, so it ends up
        # the same size as the RGB and Classification images instead of forcing
        # a fixed width that overflows onto the neighbouring columns.
        sampler = widget.segment_sampler(
            overlay, segments, classify.CLASSES, width="100%"
        )
    return (sampler,)


@app.cell
def _(sampler):
    """The labels collected in the sampler, as ``{segment_id: class_name}``.

    The widget stores ids as strings (JSON keys), so we turn them back into ints
    for the classifier. This re-runs on each click, but it is cheap and does not
    touch the widget.
    """
    current_labels = {}
    if sampler is not None and sampler.value:
        current_labels = {int(_sid): _cls for _sid, _cls in sampler.value.get("labels", {}).items()}
    return (current_labels,)


@app.cell
def _(
    classifier_choice,
    classify,
    classify_button,
    current_labels,
    mo,
    scene_tile,
    segments,
    set_classification,
):
    """Train and classify the whole tile — but only when *Classify* is clicked."""
    if classify_button.value:
        if scene_tile is None or segments is None or len(set(current_labels.values())) < 2:
            set_classification(None)
        else:
            with mo.status.spinner("Classifying..."):
                set_classification(classify.classify_segments(
                    scene_tile.image, segments, current_labels, classifier_choice.value
                ))
    return


@app.cell
def _(
    IMAGE_WIDTH,
    classify,
    classify_button,
    get_classification,
    mo,
    sampler,
    segments,
    tile_choice,
    widget,
):
    """The Classification tab: the *Classify* button plus the latest result.

    Kept independent of the labels so that labelling does not re-render it.
    """
    _classification = get_classification()
    if _classification is not None and segments is not None:
        _coloured = classify.paint_classes(segments, _classification)
        _legend = " &nbsp;&nbsp; ".join(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:rgb({_r},{_g},{_b})'></span> {_name}"
            for _name, (_r, _g, _b) in classify.CLASSES.items()
        )
        _result = mo.vstack([
            mo.image(widget.image_bytes(_coloured), rounded=True, width=IMAGE_WIDTH),
            mo.md(_legend),
        ])
    elif segments is None:
        _result = mo.md("Segment a tile first, then label segments and classify.")
    
    else:
        _result = mo.md(
            "Label at least two different classes in the sampler, then click **Classify**."
        )

    if tile_choice.value == None:
        class_view = mo.md("Please select a tile first...")
    elif not sampler:
        class_view = mo.md("Please segment your image first...")
    else:
        class_view = mo.vstack([_result, classify_button])
    return (class_view,)


if __name__ == "__main__":
    app.run()
