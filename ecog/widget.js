// widget.js — the browser side of the SegmentSampler anywidget.

// anywidget calls render() once, passing `model` (the synced Python state) and
// `el` (the DOM node to fill). Everything below runs entirely in the browser:
// the picture, the segment-id map and the class list come in via `model`, and
// the only thing sent back to Python is the `labels` dict. Because labelling
// never round-trips to Python, clicks are instant and the zoom level survives.
function render({ model, el }) {
    // ---- state pulled from Python -----------------------------------------
    const classes = model.get("classes");           // {name: [r, g, b]}
    const names = Object.keys(classes);
    let selected = names[0];                         // currently active class (first by default)
    let labels = { ...model.get("labels") };         // {segId(str): className}; local copy we mutate
    const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;  // [r,g,b] -> CSS colour string

    // ---- layout: outer container ------------------------------------------
    const root = document.createElement("div");
    root.style.width = model.get("width");
    root.style.maxWidth = "100%";
    root.style.fontFamily = "sans-serif";

    // ---- layout: the class-picker toolbar ---------------------------------
    const bar = document.createElement("div");
    Object.assign(bar.style, {
        display: "flex", flexWrap: "wrap", gap: "6px",
        alignItems: "center", marginBottom: "6px",
    });

    // One button per class, tinted with that class's colour. Clicking a button
    // just changes which class the next segment-click will assign.
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

    // Highlight the active class with a dark border; the rest get a transparent
    // border of the same width so the buttons don't shift when selection changes.
    function refreshButtons() {
        for (const name of names) {
            buttons[name].style.border =
                name === selected ? "3px solid #111" : "3px solid transparent";
        }
    }

    // "Reset view" snaps zoom/pan back to the default (declared further down).
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset view";
    Object.assign(resetBtn.style, {
        cursor: "pointer", marginLeft: "auto", padding: "3px 8px",  // marginLeft:auto pushes it to the right edge
    });
    resetBtn.addEventListener("click", () => {
        scale = 1; tx = 0; ty = 0; applyTransform();
    });
    bar.appendChild(resetBtn);

    // ---- layout: the image stack ------------------------------------------
    // viewport = fixed window that clips its contents.
    // view     = the thing we actually move/scale (CSS transform applies here).
    // Zooming/panning transforms `view`; `viewport` stays put and hides overflow.
    const viewport = document.createElement("div");
    Object.assign(viewport.style, {
        position: "relative", overflow: "hidden", width: "100%",
        border: "1px solid #ccc", borderRadius: "6px",
        touchAction: "none", cursor: "crosshair",  // touchAction:none lets us handle pointer gestures ourselves
    });
    const view = document.createElement("div");
    Object.assign(view.style, { transformOrigin: "0 0", position: "relative" });  // scale/translate anchored at top-left

    // Three stacked layers, bottom to top:
    //   img          — the photo (a JPEG data URL from Python)
    //   canvas       — the semi-transparent colour tint of labelled segments
    //   borderCanvas — the 1px segment outlines
    const img = document.createElement("img");
    img.src = model.get("image_src");
    Object.assign(img.style, { display: "block", width: "100%" });
    img.draggable = false;   // stop the browser's native image-drag from hijacking pans

    const canvas = document.createElement("canvas");   // the colour tint
    Object.assign(canvas.style, {
        position: "absolute", top: "0", left: "0",
        width: "100%", height: "100%", pointerEvents: "none",  // clicks pass through to the viewport
    });

    // Segment borders are drawn here in the browser from the id map, rather than
    // baked into the photo. That keeps the transmitted picture small (no thousands
    // of hard lines to compress) and keeps borders a crisp one pixel wide at every
    // zoom level, so fine segments stay visible.
    const borderCanvas = document.createElement("canvas");
    Object.assign(borderCanvas.style, {
        position: "absolute", top: "0", left: "0",
        width: "100%", height: "100%", pointerEvents: "none",
    });

    // Stacking order is set by append order: img first (bottom), borders last (top).
    view.appendChild(img);
    view.appendChild(canvas);
    view.appendChild(borderCanvas);   // borders sit on top of the tint
    viewport.appendChild(view);
    root.appendChild(viewport);

    // Usage hint under the image.
    const hint = document.createElement("div");
    hint.textContent = "Click a segment to label it (again to remove). " +
        "Scroll to zoom, drag to pan.";
    Object.assign(hint.style, { fontSize: "12px", color: "#666", marginBottom: "4px" });
    root.appendChild(hint);

    root.appendChild(bar);   // class buttons sit below the image

    // Running tally of how many segments are labelled per class.
    const summary = document.createElement("div");
    Object.assign(summary.style, { margin: "8px 0 4px", fontWeight: "bold" });
    root.appendChild(summary);

    el.appendChild(root);

    // ---- decode the segment-id map ----------------------------------------
    // segIds[row*natW + col] = which segment that pixel belongs to.
    // natW/natH = the id map's native pixel size (the resolution clicks resolve at).
    let segIds = null, natW = 0, natH = 0, tintArr = null;
    const ctx = canvas.getContext("2d");

    const segImg = new Image();
    segImg.onload = () => {
        natW = segImg.naturalWidth;
        natH = segImg.naturalHeight;
        canvas.width = natW;
        canvas.height = natH;

        // Draw the id map to an offscreen canvas so we can read raw pixel bytes.
        // willReadFrequently hints the browser to keep it CPU-side for fast reads.
        const off = document.createElement("canvas");
        off.width = natW; off.height = natH;
        const octx = off.getContext("2d", { willReadFrequently: true });
        octx.drawImage(segImg, 0, 0);
        const data = octx.getImageData(0, 0, natW, natH).data;  // flat RGBA: [r,g,b,a, r,g,b,a, ...]

        // Reverse of the Python packing `id = red + green*256`:
        //   data is flat sequence [r,g,b,a,r,g,b,a,....] therefore i * 4
        //   data[i*4]     = red   (low byte)
        //   data[i*4 + 1] = green (high byte)
        // This MUST mirror id_map_url() exactly — swap the channels or drop the
        // *256 and every click silently resolves to the wrong segment.
        segIds = new Int32Array(natW * natH);
        for (let i = 0; i < segIds.length; i++) {
            segIds[i] = data[i * 4] + data[i * 4 + 1] * 256;
        }

        tintArr = new Uint8ClampedArray(natW * natH * 4);  // reusable RGBA buffer for the tint layer
        drawBorders();
        redraw();
    };
    segImg.src = model.get("seg_src");

    // ---- drawing ----------------------------------------------------------
    // Borders: a pixel is an edge if the segment to its right OR below differs.
    // Computed once, when the id map loads (segment shapes never change).
    function drawBorders() {
        if (!segIds) return;
        borderCanvas.width = natW;
        borderCanvas.height = natH;
        const bd = new Uint8ClampedArray(natW * natH * 4);
        for (let row = 0; row < natH; row++) {
            for (let col = 0; col < natW; col++) {
                const i = row * natW + col;
                const id = segIds[i];
                const edge =
                    (col + 1 < natW && segIds[i + 1] !== id) ||      // neighbour to the right differs
                    (row + 1 < natH && segIds[i + natW] !== id);     // neighbour below differs
                if (edge) {
                    const j = i * 4;
                    bd[j] = 0; bd[j + 1] = 0; bd[j + 2] = 205; bd[j + 3] = 255;  // opaque blue
                }
            }
        }
        borderCanvas.getContext("2d").putImageData(new ImageData(bd, natW, natH), 0, 0);
    }

    // Tint: repaint the colour overlay for every currently-labelled segment.
    // Called after each label change. Cheap enough to redo from scratch each time.
    function redraw() {
        if (!segIds) return;
        tintArr.fill(0);   // clear to fully transparent
        for (let i = 0; i < segIds.length; i++) {
            const name = labels[segIds[i]];   // note: segIds[i] is a number, JSON keys are strings — JS coerces on lookup
            if (name === undefined) continue; // unlabelled pixel stays transparent
            const c = classes[name];
            const j = i * 4;
            tintArr[j] = c[0]; tintArr[j + 1] = c[1];
            tintArr[j + 2] = c[2]; tintArr[j + 3] = 140;   // alpha 140/255 ≈ semi-transparent
        }
        ctx.putImageData(new ImageData(tintArr, natW, natH), 0, 0);
        refreshSummary();
    }

    // Recompute the "N samples — classA: x, classB: y" line.
    function refreshSummary() {
        const entries = Object.entries(labels);
        const counts = {};
        for (const [, name] of entries) counts[name] = (counts[name] || 0) + 1;
        const parts = Object.keys(counts).sort().map((n) => `${n}: ${counts[n]}`);
        summary.textContent = entries.length
            ? `Training samples (${entries.length}) — ${parts.join(", ")}`
            : "Training samples: none yet — pick a class and click segments.";
    }

    // ---- labelling: turn a screen click into a segment toggle -------------
    function toggleAt(clientX, clientY) {
        if (!segIds) return;
        // getBoundingClientRect() gives the image's on-screen box *including* the
        // current zoom/pan, so dividing into it maps screen coords -> 0..1 -> pixel
        // coords without us having to undo the transform manually.
        const rect = img.getBoundingClientRect();
        const col = Math.floor((clientX - rect.left) / rect.width * natW);
        const row = Math.floor((clientY - rect.top) / rect.height * natH);
        if (col < 0 || col >= natW || row < 0 || row >= natH) return;  // clicked outside the image

        const id = segIds[row * natW + col];
        if (labels[id] === selected) {
            delete labels[id];          // clicking the same class again removes the label
        } else {
            labels[id] = selected;      // label it, or switch it to the current class
        }
        redraw();

        // The one place we sync back to Python. save_changes() pushes the new
        // labels dict across the bridge so other Marimo cells can read .value.
        model.set("labels", { ...labels });
        model.save_changes();
    }

    // ---- zoom and pan -----------------------------------------------------
    // The whole view is positioned by a single CSS transform: translate then scale.
    let scale = 1, tx = 0, ty = 0;
    function applyTransform() {
        view.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    }
    applyTransform();

    // Wheel zoom, clamped to 1x..8x. The tx/ty math keeps the point under the
    // cursor fixed while zooming (so you zoom "into" wherever you're pointing).
    viewport.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const cx = e.clientX - rect.left, cy = e.clientY - rect.top;   // cursor position within the viewport
        let next = Math.max(1, Math.min(8, scale * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));  // scroll up = zoom in
        const ratio = next / scale;
        tx = cx - (cx - tx) * ratio;     // keep the point under the cursor fixed
        ty = cy - (cy - ty) * ratio;
        scale = next;
        if (scale === 1) { tx = 0; ty = 0; }   // fully zoomed out: re-centre exactly
        applyTransform();
    }, { passive: false });   // passive:false because we call preventDefault()

    // Pointer drag = pan (only meaningful when zoomed in). We also track total
    // movement so we can tell a pan apart from a click.
    let dragging = false, moved = 0, lastX = 0, lastY = 0, downX = 0, downY = 0;
    viewport.addEventListener("pointerdown", (e) => {
        dragging = true; moved = 0;
        lastX = downX = e.clientX; lastY = downY = e.clientY;
        viewport.setPointerCapture(e.pointerId);   // keep receiving moves even if the cursor leaves the element
    });
    viewport.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);                      // accumulate travel distance
        if (scale > 1) { tx += dx; ty += dy; applyTransform(); }   // pan only when zoomed in
        lastX = e.clientX; lastY = e.clientY;
    });
    viewport.addEventListener("pointerup", (e) => {
        dragging = false;
        try { viewport.releasePointerCapture(e.pointerId); } catch (_) {}
        // If the pointer barely moved (<5px total) it was a tap, so treat it as a
        // label click. Anything more was a drag/pan and must NOT label.
        if (moved < 5) toggleAt(downX, downY);
    });

    refreshButtons();   // draw the initial selected-class highlight
}

export default { render };

