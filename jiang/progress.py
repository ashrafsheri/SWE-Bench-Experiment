"""Keep long runs alive when a terminal or tool session disconnects."""
import os
import sys


def progress(message):
    try:
        print(message, flush=True)
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
