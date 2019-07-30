import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 4
p = 5
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
g.add_weighted_edges_from(elist)

qaoa_methods.q_adapt_vqe_min(n, g, layer=p, pool=operator_pools.qaoa())
