# Notebooks

`ML_MINI_PROJECT_original.ipynb` is the original Colab exploration. It trains the four baseline models.

The production version of the same work lives in `src/` as importable, tested modules. Running `python -m src.train` reproduces the notebook's baseline numbers exactly (same split, same seed) and adds scaling pipelines, grid search, repeated cross-validation and all report figures.
