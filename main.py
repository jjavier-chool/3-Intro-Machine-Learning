def main(which: str = 'all'):
  match which:
    case '1'|'2':
      print(f"Error: task {which} is not executable")
    
    case '3':
      from task3 import main
      main()
    
    case '4':
      from task4 import main
      main()
    
    case '5':
      from task5 import main
      main()
    
    case 'all':
      import task3
      import task4
      import task5

      task3.main()
      task4.main()
      task5.main()

if __name__ == "__main__":
  import sys
  main(*sys.argv[1:])