import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def conf_free(q: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> bool:
    """
    Check if a configuration is in the free space.

    This function checks if the configuration q lies outside of all the obstacles in the connfiguration space.

    @param q: An np.ndarray of shape (2,) representing a robot configuration.
    @param obstacles: A list of obstacles. Each obstacle is a tuple of the form (center, radius) representing a circle.
    @return: True if the configuration is in the free space, i.e. it lies outside of all the circles in `obstacles`.
             Otherwise return False.
    """
    for obstacle in obstacles:
        ob_pos = obstacle[0]
        radius = obstacle[1]
        if np.linalg.norm(q-ob_pos) <= radius:
            return False
    return True

def edge_free(edge: Tuple[np.ndarray, np.ndarray], obstacles: List[Tuple[np.ndarray, float]]) -> bool:
    """
    Check if a graph edge is in the free space.

    This function checks if a graph edge, i.e. a line segment specified as two end points, lies entirely outside of
    every obstacle in the configuration space.

    @param edge: A tuple containing the two segment endpoints.
    @param obstacles: A list of obstacles as described in `config_free`.
    @return: True if the edge is in the free space, i.e. it lies entirely outside of all the circles in `obstacles`.
             Otherwise return False.
    """
    point1, point2 = edge
    x1, y1 = point1
    x2, y2 = point2
    for obstacle in obstacles:
        x0, y0 = obstacle[0]
        radius = obstacle[1]
        dist = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / ((y2-y1)**2+(x2-x1)**2)**0.5
        if dist <= radius:
            return False
    return True



def random_conf(width: float, height: float) -> np.ndarray:
    """
    Sample a random configuration from the configuration space.

    This function draws a uniformly random configuration from the configuration space rectangle. The configuration
    does not necessarily have to reside in the free space.

    @param width: The configuration space width.
    @param height: The configuration space height.
    @return: A random configuration uniformily distributed across the configuration space.
    """
    result = np.random.rand(2,)
    result[0] *= width
    result[1] *= height
    return result


def nearest_vertex(conf: np.ndarray, vertices: np.ndarray) -> int:
    """
    Finds the nearest vertex to conf in the set of vertices.

    This function searches through the set of vertices and finds the one that is closest to
    conf using the L2 norm (Euclidean distance).

    @param conf: The configuration we are trying to find the closest vertex to.
    @param vertices: The set of vertices represented as an np.array with shape (n, 2). Each row represents
                     a vertex.
    @return: The index (i.e. row of `vertices`) of the vertex that is closest to `conf`.
    """
    result = 0
    champ = np.linalg.norm(conf-vertices[0])
    for i, vertex in enumerate(vertices):
        dist = np.linalg.norm(conf-vertex)
        if dist < champ:
            result = i
            champ = dist
    return result

def extend(origin: np.ndarray, target: np.ndarray, step_size: float=0.2) -> np.ndarray:
    """
    Extends the RRT at most a fixed distance toward the target configuration.

    Given a configuration in the RRT graph `origin`, this function returns a new configuration that takes a
    step of at most `step_size` towards the `target` configuration. That is, if the L2 distance between `origin`
    and `target` is less than `step_size`, return `target`. Otherwise, return the configuration on the line
    segment between `origin` and `target` that is `step_size` distance away from `origin`.

    @param origin: A vertex in the RRT graph to be extended.
    @param target: The vertex that is being extended towards.
    @param step_size: The maximum allowed distance the returned vertex can be from `origin`.

    @return: A new configuration that is as close to `target` as possible without being more than
            `step_size` away from `origin`.
    """
    diff = target - origin
    dist = np.linalg.norm(diff)
    if dist < step_size: return target
    return origin + step_size * diff / dist


def rrt(origin: np.ndarray, width: float, height: float, obstacles: List[Tuple[np.ndarray, float]],
        trials: int=1000, step_size: float=0.2) -> (np.ndarray, np.ndarray):
    """
    Explore a workspace using the RRT algorithm.

    This function builds an RRT using `trials` samples from the free space.

    @param origin: The starting configuration of the robot.
    @param width: The width of the configuration space.
    @param height: The height of the configuration space.
    @param obstacles: A list of circular obstacles.
    @param trials: The number of configurations to sample from the free space.
    @param step_size: The step_size to pass to `extend`.

    @return: A tuple (`vertices`, `parents`), where `vertices` is an (n, 2) `np.ndarray` where each row is a configuration vertex
             and `parents` is an array identifying the parent, i.e. `parents[i]` is the parent of the vertex in
             the `i`th row of `vertices.
    """
    num_verts = 1

    vertices = np.zeros((trials + 1, len(origin)))
    vertices[0, :] = origin

    parents = np.zeros(trials + 1, dtype=int)
    parents[0] = -1

    for trial in range(trials):
        #TODO: Fill this loop out for your assignment.
        q_rand = random_conf(width, height)
        nearest_index = nearest_vertex(q_rand, vertices[:num_verts,:])
        q_near = vertices[nearest_index]
        q_s = extend(q_near, q_rand, step_size=step_size)
        if not conf_free(q_s, obstacles) or not edge_free((q_near, q_s), obstacles):
            continue
        else:
            vertices[num_verts] = q_s
            parents[num_verts] = nearest_index
            num_verts += 1

    return vertices[:num_verts, :], parents[:num_verts]

def backtrack(index: int, parents: np.ndarray) -> List[int]:
    """
    Find the sequence of nodes from the origin of the graph to an index.

    This function returns a List of vertex indices going from the origin vertex to the vertex `index`.

    @param index: The vertex to find the path through the tree to.
    @param parents: The array of vertex parents as specified in the `rrt` function.

    @return: The list of vertex indicies such that specifies a path through the graph to `index`.
    """
    path = [index]
    current = index
    while current != 0:
        parent = parents[current]
        path.append(parent)
        current = parent
    path.reverse()
    return path


# testing!

width = 3
height = 4
padding = 0.25

def far_enough(new_ob, obstacles, padding) -> bool:
    for obstacle in obstacles:
        if np.linalg.norm(new_ob[0] - obstacle[0]) <= 2 * padding: return False
    return True

obstacles = []
for i in range(5):
    new_ob = (np.array(random_conf(width-0.5, height-1.5)+padding+np.array([0,0.5])), padding)
    while not far_enough(new_ob, obstacles, padding):
        new_ob = (np.array(random_conf(width-0.5, height-1.5)+padding+np.array([0,0.5])), padding)
    obstacles.append(new_ob)
# obstacles = [(np.array([1, 1]), 0.25), (np.array([2, 2]), 0.25), (np.array([1, 3]), 0.3)]
goal = (np.array([1.5, 3.75]), 0.25)

origin = (1.5, 0.1)

vertices, parents = rrt(origin, width, height, obstacles, step_size=1)

index = nearest_vertex(goal[0], vertices)

if np.linalg.norm(vertices[index, :] - goal[0]) < 0.25:
    print('Path found!')
    path_verts = backtrack(index, parents)
else:
    print('No path found!')
    path_verts = []


# plot

fig, ax = plt.subplots()

ax.set_xlim([0, width])
ax.set_ylim([0, height])
ax.set_aspect('equal')

for i in range(len(parents)):
    if parents[i] < 0:
        continue
    plt.plot([vertices[i, 0], vertices[parents[i], 0]],
             [vertices[i, 1], vertices[parents[i], 1]], c='k')

for i in path_verts:
    if parents[i] < 0:
        continue
    plt.plot([vertices[i, 0], vertices[parents[i], 0]],
             [vertices[i, 1], vertices[parents[i], 1]], c='r')

for o in obstacles:
    ax.add_artist(plt.Circle(tuple(o[0]), o[1]))

ax.add_artist(plt.Circle(tuple(goal[0]), goal[1], ec=(0.004, 0.596, 0.105), fc=(1, 1, 1)))

plt.scatter([2.5], [3.5], zorder=3, c=np.array([[0.004, 0.596, 0.105]]), s=3)
plt.scatter(vertices[path_verts, 0], vertices[path_verts, 1], c=np.array([[1, 0, 0]]), s=3, zorder=2)
plt.scatter(vertices[1:, 0], vertices[1:, 1], c=np.array([[0, 0, 0]]), s=3)


# plot it
plt.show()