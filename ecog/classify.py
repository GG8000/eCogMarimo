"""Train a classifier on the labelled segments and predict all the others.

:data:`CLASSES` lists the land-cover classes and the colour each is drawn in.
:func:`classify_segments` trains the chosen model on the user's labelled
segments and returns a :class:`Classification`: the predicted class name for
every segment. :func:`paint_classes` turns that into a colour image for the
Classification tab.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from ecog.features import training_table, all_features

# Class name -> the RGB colour it is drawn in. This single list is read by the
# labelling dropdown, the colour legend and the result image alike.
CLASSES = {
    "Container": (255, 165, 0),
    "Vegetation": (154, 205, 50),
    "Water": (30, 144, 255),
    "Impervious": (220, 20, 60),
    "Car": (210, 180, 140),
    "Boat" : (159, 43, 104),
}

# Name shown in the classifier dropdown -> the model key used below.
METHODS = {
    "Random Forest": "rf",
    #"Support Vector Machine": "svm", # <- Not implemented yet
    #"k-Nearest Neighbour": "knn", # <- Not implemented yet
}


@dataclass
class Classification:
    """The predicted class name for every segment, indexed by segment id."""

    predictions: list   # predictions[seg_id] -> class name


def make_model(method: str, n_samples: int = 100):
    """Create a fresh scikit-learn model for the chosen ``method`` key."""
    if method == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=0)
    if method == "svm":
        # TODO return SVM Model
        pass
    if method == "knn":
        # Never ask for more neighbours than we have training samples.
        # TODO return knn Model 
        pass
    # Add Any Model you want
    raise ValueError(f"Unknown method {method!r}. Options: {list(METHODS.values())}")



def classify_segments(tile, segments, labels: dict, method: str = "rf") -> Classification:
    """Train on the labelled segments, then predict the class of every segment."""
    X_train, y_train = training_table(tile, segments, labels)

    model = make_model(method, n_samples=len(y_train))
    model.fit(X_train, y_train)

    predictions = model.predict(all_features(tile, segments))
    return Classification(predictions=list(predictions))


def paint_classes(segments, classification: Classification) -> np.ndarray:
    """Return an image where every segment is filled with its class colour."""
    # One colour per segment id, then look it up for every pixel of the label image.
    colour_per_segment = np.array(
        [CLASSES.get(name, (0, 0, 0)) for name in classification.predictions],
        dtype=np.uint8,
    )
    return colour_per_segment[segments.labels]
