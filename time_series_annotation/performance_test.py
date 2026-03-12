import pstats
from pstats import SortKey

# load the performance test data
p = pstats.Stats('restats')

# p.strip_dirs().sort_stats(-1).print_stats()
# p.sort_stats(SortKey.CUMULATIVE).print_stats(50)

p.sort_stats(SortKey.CUMULATIVE).print_stats('__init__')
