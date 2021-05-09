import numpy as np

def dijkstra(pathmap):
    """
    <time>2018.11.25</time>
    <summary>Dijkstra路由算法，计算一个网络中各点到其他点的最优路线的距离和路径</summary>

    <param name="pathmap">a 2-dimension array of network points' distance</param>
    :return:
    optimalpath_distance: a 2-dimension array of the optimal paths
    optimalpath_route: a 2-dimension list of list representing the optimal route

    <example>
        input:
        pathmap = np.array([[ 0,  2, 10, 12, 12, 16],
                            [ 2,  0,  2, 11, 13, 11],
                            [10,  2,  0,  3,  6, 10],
                            [12, 11,  3,  0,  4, 15],
                            [12, 13,  6,  4,  0,  5],
                            [16, 11, 10, 15,  5,  0]])

        output:
        optimalpathmap =   [[ 0,  2,  4,  7, 10, 13],
                            [ 2,  0,  2,  5,  8, 11],
                            [ 4,  2,  0,  3,  6, 10],
                            [ 7,  5,  3,  0,  4,  9],
                            [10,  8,  6,  4,  0,  5],
                            [13, 11, 10,  9,  5,  0]])

        optimalpathroute = [[[],[],[1],[1,2],[1,2],[1]],
                            [[],[],[],[2],[2],[]],
                            [[1],[],[],[],[],[]],
                            [[2,1],[2],[],[],[],[4]],
                            [[2,1],[2],[],[],[],[]],
                            [[1],[],[],[4],[],[]]]
    </example>
    """
    try:
        m, n = pathmap.shape
    except ValueError:
        raise ValueError("the input map should be a 2-dimension array")
    if m != n and m:
        raise ValueError("the input map is not a square matrix")
    for path in pathmap.flat:
        if path < 0:
            raise ValueError("the input map contains minus")

    # 最优路径距离、路线
    optimalpath_distance = np.zeros([m, n], dtype=np.float)
    optimalpath_route = []

    # 计算每个点到其他各点的最优路径，i为结点序号
    for i in range(m):
        # 未访问的结点集合S，由于set中不能含有0，
        # 所以所有结点存储在set中时比实际结点序号多1
        S = set(range(m + 1))
        # 初始化这两个最优路线变量
        optimalpath_route.append([])
        for j in range(n):
            optimalpath_route[i].append([])
        optimalpath_distance[i] = pathmap[i]
        # 当前到达结点
        currentpoint = i
        # 到当前point_min点的最短距离
        D = 0
        while S:
            # point_min = -1
            S.remove(currentpoint + 1)
            # 更新一轮最优路线
            for point in S:
                realpoint = point - 1  # 真实的结点序号比从S中取出的点序号小1
                if (
                    D + pathmap[currentpoint, realpoint]
                    < optimalpath_distance[i, realpoint]
                ):
                    optimalpath_distance[i, realpoint] = (
                        D + pathmap[currentpoint, realpoint]
                    )  # 更新最优路线距离
                    optimalpath_route[i][realpoint] = []  # 清空路线路径
                    # 重新添加最优路线路径（即经过的结点）
                    for pre_point in optimalpath_route[i][currentpoint]:
                        optimalpath_route[i][realpoint].append(pre_point)
                    optimalpath_route[i][realpoint].append(currentpoint)
            # 将距离最小的点踢出未访问结点的集合
            path_min = np.inf
            for point in S:
                realpoint = point - 1  # 真实的结点序号比从S中取出的点序号小1
                if optimalpath_distance[i, realpoint] < path_min:
                    path_min = optimalpath_distance[i, realpoint]
                    D = path_min
                    currentpoint = realpoint
    return optimalpath_distance, optimalpath_route
