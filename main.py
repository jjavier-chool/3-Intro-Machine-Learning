#!/usr/bin/env python3

'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Utility script for quickly calling different tasks.
'''

def invoke(which: int, *rest):
  print(f"### Task {which} ###")
  match which:
    case 1:
      from task1 import main
      main(*rest)
    
    case 2:
      print(f"Error: task {which} is not executable")
    
    case 3:
      from task3 import main
      main(*rest)
    
    case 4:
      from task4 import main
      main(*rest)
    
    case 5:
      from task5 import main
      main(*rest)

def main(which: str = 'all', *rest):
  if which == 'all':
    which = '1,3,4,5'
  seen = set()

  for wh in which.split(','):
    if wh in seen:
      print(f"Ignoring {wh}, already seen")
    invoke(int(wh), *rest)

if __name__ == "__main__":
  import sys
  main(*sys.argv[1:])