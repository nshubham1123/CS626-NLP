import sys, os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def dont_print(func):
    def dont_print_wrapper(*args, **kwargs):
        blockPrint()
        val = func(*args, **kwargs)
        enablePrint()
        return val
    
    return dont_print_wrapper