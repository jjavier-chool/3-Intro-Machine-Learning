def main(which: str = 'all', *rest):
  match which:
    case '1':
      from task1 import main
      main(*rest)
    
    case '2':
      print(f"Error: task {which} is not executable")
    
    case '3':
      from task3 import main
      main(*rest)
    
    case '4':
      from task4 import main
      main(*rest)
    
    case '5':
      from task5 import main
      main(*rest)
    
    case 'all':
      if rest:
        print("Warning: Cannot pass extra arguments to all tasks")
      
      import task1
      import task3
      import task4
      import task5

      print("### Task 1 ###")
      task1.main()
      print("### Task 3 ###")
      task3.main()
      print("### Task 4 ###")
      task4.main()
      print("### Task 5 ###")
      task5.main()

if __name__ == "__main__":
  import sys
  main(*sys.argv[1:])