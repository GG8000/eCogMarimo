# eCogMarimo · Interactive Sampler

A small, readable [Marimo](https://marimo.io) notebook that reproduces one eCognition workflow on aerial harbour imagery:

> **Load a tile → segment it → label segments → classify → ask an LLM assistant.**

One notebook (`app.py`) plus pure-Python modules under `ecog/`. There is
no custom front-end except a "report where I clicked" widget, all the
image drawing happens in Python. 

---

## Run it

The image tiles and lidar live in a local `data/` folder, kept out of git because
the files are large. Feel free to use any data, but it will need adaptations in `ecog/io.py`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
marimo edit app.py
```

The LLM panel reads `GEMINI_API_KEY`. If it isn't set, the panel just shows the
prompt it *would* have sent, so the app still runs offline. The notebook also picks
up `eCogMarimo/.env` automatically if it exists.

---

## How to use the app

1. Pick a **Tile** — the RGB photo appears in the *RGB* tab.
2. Pick a **Segmentation** method — the *Segmentation* tab shows the tile cut
   into segments, ready to click.
3. Choose a class in **Label as**, then **click segments** to label them
   (click again to remove). A small table tracks your samples.
4. Once you have **two or more classes** labelled, the *Classification* tab fills
   in automatically: every segment is coloured with its predicted class.
5. Ask the **Assistant** about the scene — it is given the real per-class counts
   and areas from the classification.

---

## How the code is organised

```
eCogMarimoSimple
├── app.py            # the Marimo notebook — wires the steps into a tool
├── requirements.txt    # package dependencies
├── .env # Gemini API Key
├── data/               # images and lidar data
└── ecog/               # plain Python, no Marimo, one module per step
    ├── io.py             # load a tile + its nDSM height    -> Tile
    ├── segment.py        # split into segments              -> Segments
    ├── features.py       # describe each segment with numbers
    ├── classify.py       # train a model + colour the result -> Classification
    ├── llm.py            # summarise the scene + ask Gemma
    ├── widget.py         # the tiny clickable-image widget
    └── widget.js         # its browser-side half
```

**The one rule:** `ecog/` never imports Marimo. Those modules are 
functions you can test or reuse on their own, and they document cleanly with
tools like `pdoc` (`pdoc ecog`). `app.py` is just the glue.

### How `app.py` works

Each `@app.cell` is a reactive block. When a value it reads changes, a dropdown,
or the clickable image, Marimo automatically re-runs the cells that depend on
it. You declare those dependencies through function arguments and return values. So picking a tile re-runs *load → segment → draw*, and clicking a
segment re-runs *update labels → re-tint the image → re-classify → update the
assistant's context*. The clicked labels are kept in `mo.state` so they survive
between clicks.

---

## Where to add things

| You want to… | Edit |
|---|---|
| add data | put image in `data/` + add subfolder in `ecog/io.py` in `TILES` |
| add a segmentation algorithm | `ecog/segment.py` (`segment_image`) + add its name to `METHODS` |
| add a classifier | `ecog/classify.py` (`make_model`) + add it to `METHODS` |
| add or change the per-segment features | `ecog/features.py` (`all_features`) |
| change the class list / colours | `ecog/classify.py` (`CLASSES`) |
| change what the LLM is told | `ecog/llm.py` (`describe_scene`) |
