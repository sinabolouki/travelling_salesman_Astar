import math


class Graph:

    def __init__(self, vertices, max_num):
        self.V = vertices
        self.max_num = max_num
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def Kruksal_MST(self):

        result = []

        i = 0
        e = 0

        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.max_num):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        total_weight = 0
        for u, v, weight in result:
            total_weight += weight
        return total_weight


def calc_min_distance(mat, param, closed_set):
    n = len(mat)
    min = max(mat[param])
    for j in range(n):
        if mat[j][param] < min and j not in closed_set and param != j:
            min = mat[j][param]
    return min


def kruskal(mat, param, closed_set):
    n = len(mat)
    if param in closed_set:
        graph = Graph(n - len(closed_set), n)
    else:
        graph = Graph(n - len(closed_set) - 1, n)
    for i in range(n):
        if i in closed_set or i == param:
            continue
        for j in range(i):
            if j in closed_set or j == param:
                continue
            else:
                graph.addEdge(i, j, mat[i][j])
    return graph.Kruksal_MST()


def calc_heuristic(mat, param, closed_set):
    d1 = calc_min_distance(mat, 0, closed_set)
    d2 = calc_min_distance(mat, param, closed_set)
    d3 = kruskal(mat, param, closed_set)
    return d1 + d2 + d3


def calc_dis(parents, mat):
    dist = 0
    parents.append(0)
    for i in range(1, len(parents)):
        dist += mat[parents[i]][parents[i - 1]]
        parents[i - 1] += 1
    print(dist)
    parents = parents[:-1]
    print(*parents, sep=" ")


def f_calc(f_score, set):
    for i in f_score:
        if i[0:-1] == set:
            return i[-1]
        return math.inf


def a_star(mat):
    n = len(mat)
    open_set = [0]
    came_from = [0] * n
    g_score = [math.inf] * n
    g_score[0] = 0
    parent = [[0]] * n
    f_score = [math.inf] * n
    f_score[0] = calc_heuristic(mat, 0, parent[0])
    while open_set:
        open_set.sort(key=lambda x: [f_score[x], x])
        print(open_set)
        current = open_set.pop(0)
        if len(parent[current]) == n:
            return calc_dis(current, parent[current], mat)
        for i in range(n):
            if i in parent[current]:
                continue
            t_gscore = g_score[current] + mat[current][i]
            new_h = calc_heuristic(mat, i, parent[current])
            new_f = t_gscore + new_h
            if g_score[i] > t_gscore:
                new_parent = parent[current].copy()
                new_parent.append(i)
                parent[i] = new_parent

                came_from[i] = current
                f_score[i] = new_f
                g_score[i] = t_gscore
                if i not in open_set:
                    open_set.append(i)


n = int(input())
line = []
for i in range(n):
    a = input().split()
    a_int = list(map(int, a))
    line.append(a_int)
print(line)
a_star(line)
