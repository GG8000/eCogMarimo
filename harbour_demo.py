import marimo

app = marimo.App(width="medium", app_title="Segment Labeling")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import json
    import math

    return go, math, mo, pd


@app.cell
def _(math, pd):
    # Dummy-Segmente – ersetzt später durch echten SAM-Output
    def make_segments():
        segs = []

        # 2 Schiffe
        segs.append({"id": 1, "area_m2": 412.0, "height_m": 14.5,
            "poly_x": [45,275,288,282,40,33],
            "poly_y": [22,25,55,72,70,48]})
        segs.append({"id": 2, "area_m2": 398.0, "height_m": 13.1,
            "poly_x": [390,590,602,596,384,378],
            "poly_y": [20,23,52,68,65,43]})

        # 12 Container (einzeln + gestapelt)
        for i, (x, y, w, h) in enumerate([
            (65,115,50,22),(119,115,50,22),(173,115,50,22),
            (65,140,50,22),(119,140,50,22),(173,140,50,22),
            (65,175,104,22),(173,175,104,22),
            (310,115,50,22),(364,115,50,22),
            (310,140,50,22),(364,140,50,22),
        ]):
            segs.append({"id": 3+i, "area_m2": 32.0+i*0.2, "height_m": 2.6 if h==22 and w<90 else 5.1,
                "poly_x": [x,x+w,x+w,x],
                "poly_y": [y,y,y+h,y+h]})

        # 3 Gebäude
        for i, (x,y,w,h) in enumerate([(440,100,65,35),(440,142,65,35),(515,100,60,77)]):
            segs.append({"id": 15+i, "area_m2": 310.0+i*40, "height_m": 8.0+i*0.5,
                "poly_x": [x,x+w,x+w,x],
                "poly_y": [y,y,y+h,y+h]})

        # 6 Fahrzeuge
        for i, (x,y) in enumerate([(65,218),(98,218),(131,218),(164,218),(248,218),(281,218)]):
            segs.append({"id": 18+i, "area_m2": 12.2, "height_m": 1.5,
                "poly_x": [x,x+22,x+22,x],
                "poly_y": [y,y,y+14,y+14]})

        # 4 Vegetation
        for i, (cx,cy,rx,ry) in enumerate([(345,305,14,11),(400,308,16,12),(455,305,13,10),(505,308,15,11)]):
            xs = [cx + rx*math.cos(t*math.pi/8) for t in range(16)]
            ys = [cy + ry*math.sin(t*math.pi/8) for t in range(16)]
            segs.append({"id": 24+i, "area_m2": 28.0+i*2, "height_m": 4.2+i*0.3,
                "poly_x": xs, "poly_y": ys})

        return pd.DataFrame(segs)

    df_segs = make_segments()
    return (df_segs,)


@app.cell
def _():
    # Klassen-Definitionen mit Farben
    CLASSES = {
        "Unlabeled":  "#aaaaaa",
        "Container":  "#185FA5",
        "Ship":       "#E24B4A",
        "Building":   "#BA7517",
        "Vehicle":    "#7F77DD",
        "Vegetation": "#3B6D11",
        "Water":      "#1a9988",
        "Road":       "#888888",
        "Other":      "#cc44aa",
    }
    return (CLASSES,)


@app.cell
def _(mo):
    mo.md("""
    # Segment Labeling
    """)
    return


@app.cell
def _(CLASSES, mo):
    # Aktive Klasse wählen
    class_picker = mo.ui.dropdown(
        options=list(CLASSES.keys())[1:],  # ohne Unlabeled
        value="Container",
        label="Active class  →  click segments to label",
    )
    class_picker
    return (class_picker,)


@app.cell
def _(df_segs, mo):
    # Label-State: dict {segment_id: class_name}
    get_labels, set_labels = mo.state(
        {int(row["id"]): "Unlabeled" for _, row in df_segs.iterrows()}
    )
    get_selected, set_selected = mo.state(None)  # NEU
    return get_labels, get_selected, set_labels, set_selected


@app.cell
def _(class_picker, df_segs, get_labels, set_labels, set_selected):
    def handle_click(plot_val):
        if not plot_val:
            return
        curve_number = plot_val[0].get("curveNumber")
        if curve_number is None:
            return
        seg_id = int(df_segs.iloc[curve_number]["id"])
        current = dict(get_labels())
        if current.get(seg_id) == class_picker.value:
            current[seg_id] = "Unlabeled"
        else:
            current[seg_id] = class_picker.value
        set_selected(seg_id)
        set_labels(current)

    return (handle_click,)


@app.cell
def _(CLASSES, df_segs, get_labels, get_selected, go, handle_click, mo):
    # Klick-Event verarbeiten: segment_id aus plot.value lesen
    # und mit aktueller Klasse labeln

    _labels_now = get_labels()
    _plot_trigger = mo.ui.plotly(go.Figure())  # dummy, wird unten ersetzt


    # Figure aufbauen: Farbe je nach aktuellem Label
    _fig = go.Figure()

    # Hintergrund
    _fig.add_shape(type="rect", x0=0, y0=0,   x1=700, y1=90,
                   fillcolor="#3a5a7a", line_width=0, layer="below")
    _fig.add_shape(type="rect", x0=0, y0=90,  x1=700, y1=295,
                   fillcolor="#7a7268", line_width=0, layer="below")
    _fig.add_shape(type="rect", x0=0, y0=295, x1=700, y1=340,
                   fillcolor="#4a6238", line_width=0, layer="below")

    _labels_now = get_labels()

    for _, _row in df_segs.iterrows():
        _sid   = int(_row["id"])
        _cls   = _labels_now.get(_sid, "Unlabeled")
        _color = CLASSES.get(_cls, "#aaaaaa")
        _xs    = list(_row["poly_x"]) + [_row["poly_x"][0]]
        _ys    = list(_row["poly_y"]) + [_row["poly_y"][0]]

        _is_selected = (_sid == get_selected())  # NEU

        _fig.add_trace(go.Scatter(
            x=_xs, y=_ys,
            fill="toself",
            fillcolor=_color,
            opacity=0.75 if _is_selected else (0.55 if _cls != "Unlabeled" else 0.25),
            line=dict(
                color="white" if _is_selected else _color,
                width=3.5 if _is_selected else (2.5 if _cls != "Unlabeled" else 1),
            ),
            mode="lines",
            customdata=[[
                _sid,
                _cls,
                _row['height_m'],
                _row['area_m2']
            ]] * len(_xs),
            text=(
                "ID: %{customdata[0]}<br>"
                "Label: %{customdata[1]}<br>"
                "Height: %{customdata[2]:.1f} m<br>"
                "Area: %{customdata[3]:.1f} m²"
                "<extra></extra>"
            ),
            hoveron="fills",
            showlegend=False,
        ))

    _fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 700], constrain="domain"),
        yaxis=dict(visible=False, range=[340, 0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="#7a7268",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode=False,
        clickmode="event",
    )

    plot = mo.ui.plotly(_fig, on_change=handle_click)
    return (plot,)


@app.cell
def _(plot):
    plot
    return


@app.cell
def _(CLASSES, get_labels, mo):
    # Legende + Live-Zähler
    _labels_now = get_labels()
    _counts = {}
    for _cls in CLASSES:
        _counts[_cls] = sum(1 for v in _labels_now.values() if v == _cls)

    _labeled   = sum(1 for v in _labels_now.values() if v != "Unlabeled")
    _total     = int(len(_labels_now)*0.5)
    _progress  = int(_labeled / _total * 100)
    _bar       = "█" * (_progress // 5) + "░" * (20 - _progress // 5)

    _rows = " · ".join(
        f"**{cls}** {cnt}"
        for cls, cnt in _counts.items()
        if cnt > 0 and cls != "Unlabeled"
    )
    _unlabeled = _counts.get("Unlabeled", 0)

    mo.md(
        f"Progress `{_bar}` {_labeled}/{_total} labeled"
        + (f"  ·  *{_unlabeled} unlabeled*" if _unlabeled else "  ·  ✓ all labeled")
        + (f"\n\n{_rows}" if _rows else "")
    )
    return


@app.cell
def _(mo):
    # Speichern
    save_btn = mo.ui.run_button(label="Save labels to parquet")
    save_btn
    return (save_btn,)


@app.cell
def _(df_segs, get_labels, mo, save_btn):
    if save_btn.value:
        _labels_now = get_labels()
        _out = df_segs.copy()
        _out["label"] = _out["id"].apply(lambda i: _labels_now.get(int(i), "Unlabeled"))
        _out.to_parquet("labels.parquet", index=False)
        _n = (_out["label"] != "Unlabeled").sum()
        result_msg = mo.callout(
            mo.md(f"Saved **{_n} labels** → `labels.parquet`"),
            kind="success",
        )
    else:
        result_msg = mo.md("")
    result_msg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
