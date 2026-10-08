
# DFS Traversal Function

def dfs(graph, node, visited):

    visited.append(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited


# Test Function

def test_dfs(graph, start):

    if len(graph) == 1:
        return "FAIL : Single Node Graph"

    visited = dfs(graph, start, [])

    if len(visited) != len(graph):
        return "FAIL : Disconnected Graph"

    return "PASS", visited



# Test Case 1 (Connected Graph)

graph1 = {
    'A':['B','C'],
    'B':['D'],
    'C':['E'],
    'D':[],
    'E':[]
}

print("Test Case 1:", test_dfs(graph1,'A'))


# Test Case 2 (Multiple Branches)

graph2 = {
    'A':['B','C','D'],
    'B':['E'],
    'C':['F'],
    'D':['G'],
    'E':[],
    'F':[],
    'G':[]
}

print("Test Case 2:", test_dfs(graph2,'A'))



# Test Case 3 (Cyclic Graph)

graph3 = {
    'A':['B','C'],
    'B':['A','D'],
    'C':['A'],
    'D':[]
}

print("Test Case 3:", test_dfs(graph3,'A'))



# Test Case 4 (Single Node)

graph4 = {
    'A':[]
}

print("Test Case 4:", test_dfs(graph4,'A'))


# Test Case 5 (Disconnected Graph)

graph5 = {
    'A':['B'],
    'B':[],
    'C':['D'],
    'D':[]
}

print("Test Case 5:", test_dfs(graph5,'A'))
