"""Package paths. Every script imports this, so the package runs from wherever it is unpacked."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")


def results(stage):
    p = os.path.join(RESULTS, stage); os.makedirs(p, exist_ok=True); return p


def fig(name):
    os.makedirs(FIGS, exist_ok=True); return os.path.join(FIGS, name)


def paper(name):
    os.makedirs(PAPER, exist_ok=True); return os.path.join(PAPER, name)
