from time import perf_counter

class perf_timer:
    '''Context manager for timing a block of code.'''
    
    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, *exc):
        self.end = perf_counter()
        self.total = self.end - self.start