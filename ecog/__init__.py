"""eCogMarimo (simple) — the analysis steps as plain Python functions.

The package mirrors the workflow of the app, one module per step:

    io        load an aerial tile               -> Tile
    segment   split the tile into segments       -> Segments
    features  describe each segment with numbers -> feature rows
    classify  train a model and label segments   -> Classification
    llm       summarise the result and ask Gemma -> text
    widget    a tiny clickable-image widget (the only front-end code)

None of these modules import Marimo. The notebook ``app.py`` is the only place
that wires them together into an interactive tool.
"""
