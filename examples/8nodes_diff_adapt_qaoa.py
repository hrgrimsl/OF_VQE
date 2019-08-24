import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 8
p = 16
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist=[(0, 1, 1.0), (0, 2, 1.2), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0), (0, 6, 1.0), (0, 7, 1.0), (1, 2, 1.3), (2, 3, 1.5), (3, 4, 1.6), (4, 5, 1.8), (5, 6, 1.9), (6, 7, 2.0)]
g.add_weighted_edges_from(elist)

qaoa_methods.adapt_qaoa(n, g, layer=p, pool=operator_pools.qaoa())
#qaoa_methods.qaoa(n, g, layer=p, pool=operator_pools.qaoa())
