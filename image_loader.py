import marimo

__generated_with = "0.23.7"
app = marimo.App(
    width="medium",
    app_title="Image Selection",
    layout_file="layouts/image_loader.grid.json",
)


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import tifffile as tiff
    import os
    import matplotlib as plt
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    return Path, Resampling, mo, np, os, rasterio, tiff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preview
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image Viewer
    """)
    return


@app.cell
def _(Path):
    IMAGE_PATH_ROOT = Path("./data/")
    RESAMPLING_SCALE = 0.25
    return IMAGE_PATH_ROOT, RESAMPLING_SCALE


@app.cell
def _(Path, Resampling, rasterio):
    ## To show the images, resample to reduce loading times
    def resample_tif(file_path : Path, scale : int):
        new_file_path = file_path.parent / f"{file_path.stem}_resampled.tif"
        with rasterio.open(file_path) as src:
            out_shape = (src.count, 
                         int(src.height * scale), 
                         int(src.width * scale)
                        )
        
            resampled_img_array = src.read(
                out_shape = out_shape,
                resampling=Resampling.bilinear
            )
        
            new_transform = src.transform * src.transform.scale(
                (src.width / resampled_img_array.shape[-1]),
                (src.height / resampled_img_array.shape[-2])
            )

            profile = src.profile.copy()
            profile.update({
                "transform" : new_transform,
                "height" : resampled_img_array.shape[1],
                "width" : resampled_img_array.shape[2]
            })

            with rasterio.open(new_file_path, "w", **profile) as dst:
                dst.write(resampled_img_array)
            print("Successfully resampled the tile, so it loads faster. For analysis the original is used")

    return (resample_tif,)


@app.cell
def _(mo):
    tile_dropdown = mo.ui.dropdown(
        options=["30", "35", "40"],
        value="30", label="Select a Tile: "
    )
    tile_dropdown
    return (tile_dropdown,)


@app.cell
def _(
    IMAGE_PATH_ROOT,
    Path,
    RESAMPLING_SCALE,
    np,
    os,
    resample_tif,
    tiff,
    tile_dropdown,
):
    # Load tif into numpy array
    tile_number = tile_dropdown.selected_key

    dir_name = f"3151{tile_number}_56865"

    # File for processing
    file_name = f"BE_ORTHO_27032011_3151{tile_number}_56865_UTM31N.tif"
    tile_path = os.path.join(IMAGE_PATH_ROOT, dir_name, file_name)

    # File resampled just for viewing
    file_name_resampled = f"BE_ORTHO_27032011_3151{tile_number}_56865_UTM31N_resampled.tif"
    tile_path_resampled = os.path.join(IMAGE_PATH_ROOT, dir_name, file_name_resampled)

    if not os.path.exists(tile_path_resampled):
        resample_tif(Path(tile_path), RESAMPLING_SCALE)

    resampled = tiff.imread(tile_path_resampled)

    if resampled.shape[0] == 3:
        display_array = np.transpose(resampled, (1,2,0))
    else:
        display_array = resampled

    val_min = np.min(resampled)
    val_max = np.max(resampled)
    norm_array = ((resampled - val_min)/(val_max-val_min) * 255).astype(np.uint8)

    return (norm_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample Viewer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Switch Logic
    """)
    return


@app.cell
def _(mo):
    switch = mo.ui.switch()
    return (switch,)


@app.cell
def _(mo, switch):

    switch_label = "RGB" if switch.value else "Samples"

    mo.hstack([switch, mo.md(f"**{switch_label}**")])
    return


@app.cell
def _(mo, norm_array, switch):
    sample_preview = switch.value
    if sample_preview:
        mein_view = mo.md("### 🖼️ SAMPLE View")
    else:
        # Ansicht B: Switch ist AUS (z. B. einfache, große Ansicht)
        mein_view = mo.image(norm_array, height="100%")

    # Den gewählten View anzeigen
    mein_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Segmentation
    """)
    return


@app.cell
def _(mo):
    segmentation_dropdown = mo.ui.dropdown(
        options=["SAM", "Felszenswab", "..."],
        value="SAM",
        label="Select Algorithm",
    )
    segmentation_dropdown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification
    """)
    return


@app.cell
def _(mo):
    classification_dropdown = mo.ui.dropdown(
        options=["RF", "SVM", "..."],
        value="RF",
        label="Select Classifier",
    )
    classification_dropdown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Large Language Model
    """)
    return


@app.cell
def _(mo):
    def simple_echo_model(messages, config):
        return f"You said: {messages[-1].content}"

    mo.ui.chat(
        simple_echo_model,
        prompts=["Hello", "How are you?"],
        show_configuration_controls=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis Assistent
    """)
    return


if __name__ == "__main__":
    app.run()
