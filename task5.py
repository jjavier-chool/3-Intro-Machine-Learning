'''
Note: It might be worth just incorporating all the tasks into one big file
'''

from tasks2_4 import baseline_training, FNN

DROPOUT = [0.6, 0.3, 0.3]

if __name__ == "__main__":
  baseline_training(model = FNN(DROPOUT))