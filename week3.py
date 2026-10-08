from itertools import permutations

def tsp(g):
    
    if len(g) == 1:
        print("Path: [0, 0]")
        print("Cost: 0")
        print("SUCCESS")
        return

    cost = float('inf')
    best = []

    for p in permutations(range(1, len(g))):
        path = [0] + list(p) + [0]
        c = sum(g[path[i]][path[i + 1]]
                for i in range(len(g)))
        if c < cost:
            cost = c
            best = path

    print("Path:", best)
    print("Cost:", cost)
    print("SUCCESS")


def gc(g, m):
    n = len(g)
    color = [0] * n

    def solve(v):
        if v == n:
            return True
        for c in range(1, m + 1):
            if all(g[v][i] == 0 or color[i] != c
                   for i in range(n)):
                color[v] = c
                if solve(v + 1):
                    return True
                color[v] = 0
        return False

    if solve(0):
        print("Colors:", color)
        print("SUCCESS")
    else:
        print("FAILURE")

print("T1:Small Complete Graph (TSP)")
tsp([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

print("\nT2:Triangle Graph (Graph Coloring)")
gc([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], 3)

print("\nT3:Bipartite Graph (Graph Coloring)")
gc([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], 2)

print("\nT4:One City Graph (TSP) - Boundary Case")
tsp([[0]])

print("\nT5:Complete Graph K4 with 3 Colors (Graph Coloring)")
gc([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
], 3)
