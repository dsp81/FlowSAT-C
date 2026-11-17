"""sys.path set-up so that both `metacf` and the reused `flowsat_C.*` modules import."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../flowsat_C
PARENT = os.path.dirname(REPO)                                           # .../DJ_Sat


def ensure():
    for p in (REPO, PARENT):
        if p not in sys.path:
            sys.path.insert(0, p)
