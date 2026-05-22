# eCogMarimo

# File Structure
```
📦eCogMarimo
 ┣ 📂data # img data
 ┃ ┣ 📂315130_56865
 ┃ ┣ 📂315135_56865
 ┃ ┗ 📂315140_56865
 ┣ 📂layouts # Layout for the grid in marimo
 ┃ ┗ 📜image_loader.grid.json
 ┣ 📂src
 ┃ ┣ 📂core # Core Logic for the application (pure python)
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜classification.py
 ┃ ┃ ┣ 📜llm_client.py
 ┃ ┃ ┣ 📜sampler.py
 ┃ ┃ ┣ 📜segmentation.py
 ┃ ┃ ┗ 📜viewer.py
 ┃ ┣ 📂ui # ui parts for the interactive part of the app
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜common.py
 ┃ ┃ ┣ 📜view_class.py
 ┃ ┃ ┣ 📜view_llm.py
 ┃ ┃ ┣ 📜view_segment.py
 ┃ ┃ ┗ 📜view_viewer.py
 ┃ ┗ 📜__init__.py
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜app.py # Main application
 ┗ 📜requirements.txt
```


