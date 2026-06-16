"""The interactive sampler: an image widget you label by clicking segments.

This is the only front-end code in the project. The widget is deliberately
self-contained: Python hands it the picture, a hidden map of which segment each
pixel belongs to, and the list of classes. From then on it does everything in
the browser — picking a class, tinting clicked segments, zooming and panning —
and only the collected labels (segment id -> class name) are synced back to
Python. Because labelling never touches Python, the widget is never rebuilt
mid-session: clicks are instant and the zoom level stays put.
"""

import base64
from io import BytesIO

import anywidget
import traitlets
import numpy as np
import marimo as mo
from PIL import Image


def image_bytes(image: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 image as PNG bytes (for ``mo.image``)."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(image: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 image as a base64 JPEG data URL (for the widget)."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def id_map_url(labels: np.ndarray) -> str:
    """Encode the segment-id image as a lossless PNG data URL.

    Each segment id is packed into a pixel's red and green channels
    (``id = red + green * 256``), so the browser can look up which segment any
    pixel belongs to. This supports up to 65 535 segments, comfortably more than
    the sliders allow. PNG (not JPEG) is essential here — the ids must survive
    exactly.
    """
    ids = labels.astype(np.uint32)
    rgb = np.zeros(ids.shape + (3,), dtype=np.uint8)
    rgb[:, :, 0] = ids & 0xFF
    rgb[:, :, 1] = (ids >> 8) & 0xFF
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class SegmentSampler(anywidget.AnyWidget):
    """A clickable, zoomable image for labelling segments.

    Read ``.value["labels"]`` from another cell to get ``{segment_id: class}``
    (ids are strings, as JSON requires). Everything else is input the widget
    needs and never changes after it is built.
    """

    image_src = traitlets.Unicode("").tag(sync=True)   # the picture, as a data URL
    seg_src = traitlets.Unicode("").tag(sync=True)      # the packed segment-id map
    classes = traitlets.Dict().tag(sync=True)           # class name -> [r, g, b]
    width = traitlets.Unicode("100%").tag(sync=True)    # CSS max-width of the widget
    labels = traitlets.Dict().tag(sync=True)            # segment id (str) -> class name

    _esm = """
    function render({ model, el }) {
        const classes = model.get("classes");           // {name: [r, g, b]}
        const names = Object.keys(classes);
        let selected = names[0];
        let labels = { ...model.get("labels") };         // {segId(str): className}
        const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

        // ---- layout -----------------------------------------------------
        const root = document.createElement("div");
        root.style.width = model.get("width");
        root.style.maxWidth = "100%";
        root.style.fontFamily = "sans-serif";

        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "flex", flexWrap: "wrap", gap: "6px",
            alignItems: "center", marginBottom: "6px",
        });

        const buttons = {};
        for (const name of names) {
            const b = document.createElement("button");
            b.textContent = name;
            Object.assign(b.style, {
                cursor: "pointer", borderRadius: "6px", padding: "3px 8px",
                color: "#fff", background: rgbCss(classes[name]),
            });
            b.addEventListener("click", () => { selected = name; refreshButtons(); });
            bar.appendChild(b);
            buttons[name] = b;
        }
        
        function refreshButtons() {
            for (const name of names) {
                buttons[name].style.border =
                    name === selected ? "3px solid #111" : "3px solid transparent";
            }
        }

        const resetBtn = document.createElement("button");
        resetBtn.textContent = "Reset view";
        Object.assign(resetBtn.style, {
            cursor: "pointer", marginLeft: "auto", padding: "3px 8px",
        });
        resetBtn.addEventListener("click", () => {
            scale = 1; tx = 0; ty = 0; applyTransform();
        });
        bar.appendChild(resetBtn);

        // The viewport clips the picture; the view inside it is what we zoom/pan.
        const viewport = document.createElement("div");
        Object.assign(viewport.style, {
            position: "relative", overflow: "hidden", width: "100%",
            border: "1px solid #ccc", borderRadius: "6px",
            touchAction: "none", cursor: "crosshair",
        });
        const view = document.createElement("div");
        Object.assign(view.style, { transformOrigin: "0 0", position: "relative" });

        const img = document.createElement("img");
        img.src = model.get("image_src");
        Object.assign(img.style, { display: "block", width: "100%" });
        img.draggable = false;

        const canvas = document.createElement("canvas");   // the colour tint
        Object.assign(canvas.style, {
            position: "absolute", top: "0", left: "0",
            width: "100%", height: "100%", pointerEvents: "none",
        });

        view.appendChild(img);
        view.appendChild(canvas);
        viewport.appendChild(view);
        root.appendChild(viewport);
        
        const hint = document.createElement("div");
        hint.textContent = "Click a segment to label it (again to remove). " +
            "Scroll to zoom, drag to pan.";
        Object.assign(hint.style, { fontSize: "12px", color: "#666", marginBottom: "4px" });
        root.appendChild(hint);
        
        root.appendChild(bar);   // class buttons sit below the image

        const summary = document.createElement("div");
        Object.assign(summary.style, { margin: "8px 0 4px", fontWeight: "bold" });
        root.appendChild(summary);
        const list = document.createElement("div");
        Object.assign(list.style, { maxHeight: "180px", overflow: "auto", fontSize: "13px" });
        root.appendChild(list);

        el.appendChild(root);

        // ---- load the segment-id map ------------------------------------
        let segIds = null, natW = 0, natH = 0, tintArr = null;
        const ctx = canvas.getContext("2d");

        const segImg = new Image();
        segImg.onload = () => {
            natW = segImg.naturalWidth;
            natH = segImg.naturalHeight;
            canvas.width = natW;
            canvas.height = natH;
            const off = document.createElement("canvas");
            off.width = natW; off.height = natH;
            const octx = off.getContext("2d", { willReadFrequently: true });
            octx.drawImage(segImg, 0, 0);
            const data = octx.getImageData(0, 0, natW, natH).data;
            segIds = new Int32Array(natW * natH);
            for (let i = 0; i < segIds.length; i++) {
                segIds[i] = data[i * 4] + data[i * 4 + 1] * 256;
            }
            tintArr = new Uint8ClampedArray(natW * natH * 4);
            redraw();
        };
        segImg.src = model.get("seg_src");

        // ---- drawing and the samples list -------------------------------
        function redraw() {
            if (!segIds) return;
            tintArr.fill(0);
            for (let i = 0; i < segIds.length; i++) {
                const name = labels[segIds[i]];
                if (name === undefined) continue;
                const c = classes[name];
                const j = i * 4;
                tintArr[j] = c[0]; tintArr[j + 1] = c[1];
                tintArr[j + 2] = c[2]; tintArr[j + 3] = 140;
            }
            ctx.putImageData(new ImageData(tintArr, natW, natH), 0, 0);
            refreshSummary();
        }

        function refreshSummary() {
            const entries = Object.entries(labels);
            const counts = {};
            for (const [, name] of entries) counts[name] = (counts[name] || 0) + 1;
            const parts = Object.keys(counts).sort().map((n) => `${n}: ${counts[n]}`);
            summary.textContent = entries.length
                ? `Training samples (${entries.length}) — ${parts.join(", ")}`
                : "Training samples: none yet — pick a class and click segments.";
            const rows = entries
                .map(([id, name]) => [parseInt(id, 10), name])
                .sort((a, b) => a[0] - b[0])
                .map(([id, name]) =>
                    `<tr><td style="padding:1px 12px 1px 0">${id}</td><td>${name}</td></tr>`)
                .join("");
            list.innerHTML = entries.length
                ? `<table><thead><tr><th style="text-align:left;padding-right:12px">Segment</th>`
                  + `<th style="text-align:left">Class</th></tr></thead><tbody>${rows}</tbody></table>`
                : "";
        }

        function toggleAt(clientX, clientY) {
            if (!segIds) return;
            const rect = img.getBoundingClientRect();   // already includes the zoom
            const col = Math.floor((clientX - rect.left) / rect.width * natW);
            const row = Math.floor((clientY - rect.top) / rect.height * natH);
            if (col < 0 || col >= natW || row < 0 || row >= natH) return;
            const id = segIds[row * natW + col];
            if (labels[id] === selected) {
                delete labels[id];          // same class again removes the label
            } else {
                labels[id] = selected;      // label it (or switch its class)
            }
            redraw();
            model.set("labels", { ...labels });
            model.save_changes();
        }

        // ---- zoom and pan ----------------------------------------------
        let scale = 1, tx = 0, ty = 0;
        function applyTransform() {
            view.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
        }
        applyTransform();

        viewport.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = viewport.getBoundingClientRect();
            const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
            let next = Math.max(1, Math.min(8, scale * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
            const ratio = next / scale;
            tx = cx - (cx - tx) * ratio;     // keep the point under the cursor fixed
            ty = cy - (cy - ty) * ratio;
            scale = next;
            if (scale === 1) { tx = 0; ty = 0; }
            applyTransform();
        }, { passive: false });

        let dragging = false, moved = 0, lastX = 0, lastY = 0, downX = 0, downY = 0;
        viewport.addEventListener("pointerdown", (e) => {
            dragging = true; moved = 0;
            lastX = downX = e.clientX; lastY = downY = e.clientY;
            viewport.setPointerCapture(e.pointerId);
        });
        viewport.addEventListener("pointermove", (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            moved += Math.abs(dx) + Math.abs(dy);
            if (scale > 1) { tx += dx; ty += dy; applyTransform(); }   // pan when zoomed
            lastX = e.clientX; lastY = e.clientY;
        });
        viewport.addEventListener("pointerup", (e) => {
            dragging = false;
            try { viewport.releasePointerCapture(e.pointerId); } catch (_) {}
            if (moved < 5) toggleAt(downX, downY);   // a tap, not a drag, labels
        });

        refreshButtons();
    }
    export default { render };
    """


def segment_sampler(image: np.ndarray, segments, classes: dict, width: str = "100%"):
    """Build the interactive sampler for one segmented tile.

    ``image`` is the picture to show (with segment borders already drawn);
    ``segments`` carries the per-pixel segment ids; ``classes`` maps each class
    name to its RGB colour. Read the collected labels from ``.value["labels"]``.
    """
    return mo.ui.anywidget(SegmentSampler(
        image_src=data_url(image),
        seg_src=id_map_url(segments.labels),
        classes={name: list(colour) for name, colour in classes.items()},
        width=width,
    ))
