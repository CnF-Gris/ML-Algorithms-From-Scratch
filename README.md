# How to run this code
- Take any editor, or IDE that supports jupyter notebook files (I used VS Code)
- Use Python 3.10.11
- Activate your `venv` (or `conda`) and run `pip install -r requirement.txt` from the root directory of the project.

## What's in main.py?
Nothing, just some tests, as jupyter notebooks do not reload custom 
libraries in an easy way.

## Important Things
Almost every model implements `sklearn` `BaseEstimator`, so you can use a `GridSearchCV` to get the best parameters