"""Streamlit entry point for SIGNOVA.

The live recognizer in streamlit_app.py runs in the browser and does not need
an SVM to operate.  This wrapper keeps sklearn's normal SVC behaviour when two
or more classes are available, but gracefully handles the sparse one-class
case produced when MediaPipe can only extract one usable dataset label.
"""

from pathlib import Path
import runpy

import numpy as np
import sklearn.svm


_OriginalSVC = sklearn.svm.SVC


class SafeSVC:
    """Use real SVC for 2+ classes; use a constant model for one class."""

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        class_weight=None,
        random_state=None,
        **kwargs,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.class_weight = class_weight
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self.classes_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes

        if len(classes) < 2:
            self._constant_class = classes[0] if len(classes) else None
            return self

        self._model = _OriginalSVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        if self._model is not None:
            return self._model.predict(X)
        return np.asarray([self._constant_class] * len(X))

    def predict_proba(self, X):
        if self._model is not None:
            return self._model.predict_proba(X)
        return np.ones((len(X), 1), dtype=float)


sklearn.svm.SVC = SafeSVC

# Streamlit reruns app.py repeatedly. runpy executes the real app on every
# rerun instead of relying on Python's import cache.
runpy.run_path(
    str(Path(__file__).with_name("streamlit_app.py")),
    run_name="__main__",
)
