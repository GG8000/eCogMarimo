import marimo as mo

def render_viewer_view(rgb_ui, segment_ui, class_ui):
    
    inner_tabs = mo.ui.tabs({
        "RGB" : rgb_ui,
        "Segmentation" : segment_ui,
        "Classification" : class_ui
    })
    
    return mo.vstack([
        mo.md("## Viewer"),
        inner_tabs
    ])

