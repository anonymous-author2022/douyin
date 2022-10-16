from ananke.graphs import ADMG
from ananke.identification import OneLineID
from ananke.estimation import CausalEffect
from ananke.datasets import load_afixable_data
from ananke.estimation import AutomatedIF
import numpy as np
import pandas as pd


def construct_causal_graph(vertices, di_edges, bi_edges=[]):
    graph = ADMG(vertices, di_edges)
    return graph


def get_ace(G, data, causal, effect, algo="aipw"):
    ace_obj = CausalEffect(graph=G, treatment=causal, outcome=effect)
    ace_ipw = ace_obj.compute_effect(data, algo)
    return ace_ipw
