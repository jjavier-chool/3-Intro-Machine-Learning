# 3-Intro-Machine-Learning
The third assignment for CS429. To download and decompress the dataset, execute:
```
tar -zxf aclImdb_v1.tar.gz
```
The file is given at https://ai.stanford.edu/~amaas/data/sentiment/. Alternatively, run task1.py.

## How to Run
You can run tasks manually using each of:
```console
python task1.py
#python task2.py # Not executable
python task3.py
python task4.py
python task5.py
```

Or you can use the utility script:
```console
python main.py [1,3-5|all]
```

These will run the task-specific code, but additional utilities are provided:
```console
python main.py all # Runs all tasks in sequence
python main.py 3,5 # Runs task 3 then task 5, used to quickly evaluate tasks 5.1 and 5.2
```

It may be necessary to downgrade numpy or test in a separate environment with numpy 1.0 due to some jank with numpy 2.0.

Environment method I used:
```console
python3 -m venv assign3-env
source assign3-env/bin/activate
pip install -r requirements.txt
```
Deactive with:
```console
deactivate
```

Reactivate with:
```console
source assign3-env/bin/activate
```
