# -*- coding: utf-8 -*-

"""Network Optimization and Causal Analysis of Petrurb-seq."""

import re

from .api import *  # noqa


def daggity_to_dot(daggity_string):
    """
    Modified from dowhy: https://www.pywhy.org/dowhy/v0.11.1/_modules/dowhy/utils/graph_operations.html#daggity_to_dot
    Converts the input daggity_string to valid DOT graph format.

    :param daggity_string: Output graph from Daggity site
    :returns: DOT string
    """

    graph = re.sub(r"\n", "; ", daggity_string)
    graph = re.sub(r"^dag ", "digraph ", graph)
    graph = re.sub("{;", "{", graph)
    graph = re.sub("};", "}", graph)
    graph = re.sub("outcome,*,", "", graph)
    graph = re.sub("adjusted,*", "", graph)
    graph = re.sub("exposure,*", "", graph)
    graph = re.sub("latent,*", 'observed="no",', graph)
    graph = re.sub(",]", "]", graph)
    graph = re.sub(
        r'bb="[\d.,]+";?', "", graph
    )  # Remove bb line with four numbers and optional trailing semicolon
    return graph
