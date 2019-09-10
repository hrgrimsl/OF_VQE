import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 7
p = 2
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1.4), (1, 2, 1.1), (2, 3, 1.0), (3, 4, 2.0), (1, 3, 1.5), (1, 4, 0.6), (2, 4, 1.5), (3, 5, 1.8), (4, 5, 0.75), (3, 6, 1.3)]
g.add_weighted_edges_from(elist)

qaoa_methods.adapt_qaoa(n, g, layer=p, pool=operator_pools.qaoa())
#qaoa_methods.qaoa(n, g, layer=p, pool=operator_pools.qaoa())
