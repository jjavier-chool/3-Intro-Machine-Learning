# 3-Intro-Machine-Learning
The third assignment for CS429.

## TODO:
- Task 1
- Task 2
- Task 3
- Task 4
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
pip install "numpy<2" pandas matplotlib idx2numpy scikit-learn
```
Deactive with:
```console
deactivate
```

Reactivate with:
```console
source assign3-env/bin/activate
```
