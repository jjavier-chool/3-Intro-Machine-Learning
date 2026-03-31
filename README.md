# 3-Intro-Machine-Learning
The third assignment for CS429. To download and decompress the dataset, execute:
```
tar -zxf aclImdb_v1.tar.gz
```
The file is given at https://ai.stanford.edu/~amaas/data/sentiment/.

## TODO:
- Task 1
- Task 2
- Task 3
- Task 4
- Task 5
- Report on Overleaf

## How to Run
For Tasks 3 and 4, run:
```
python task3.py
```
```
python task4.py
```
It is probably necessary to downgrade numpy or test in a separate environment with numpy 1.0 due to some jank with numpy 2.0.

Environment method I used:
```console
python3 -m venv assign3-env
source assign3-env/bin/activate
pip install "numpy<2" pandas matplotlib scikit-learn nltk joblib torch
```
Deactive with:
```console
deactivate
```

Reactivate with:
```console
source assign3-env/bin/activate
```
