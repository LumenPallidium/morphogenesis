import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection
from tqdm import tqdm

def scale_free_decay(x, t, step_size = 0.01, exponent = -3):
    x *= (1 + step_size/(t+1))**(exponent)
    return x


class Domain:
    def __init__(self,
                 within_func,
                 generate_func,):
        self.within_func = within_func
        self.generate_func = generate_func

    def within(self, pos):
        return self.within_func(pos)
    
    def generate(self):
        return self.generate_func()

class Edge:
    def __init__(self, source, target, age = 0, color = "#000000", width = 1):
        self.source = source
        self.target = target
        self.age = age
        self.color = color
        self.width = width
    
    def __repr__(self):
        return f"Edge({self.source}, {self.target} ({self.age}, {self.color}))"

class Network:
    def __init__(self,
                 domain,
                 root_node = None,
                 new_vein_distance = 0.05,
                 birth_distance_v = 0.05,
                 birth_distance_a = 0.05,
                 kill_distance = 0.025,
                 darts_per_step = 100,
                 start_size = 4,
                 start_width = 2.7,
                 colors = None,
                 decay_type = "scale_free",
                 width_decay = 0.98,
                 distance_decay = 0.99,
                 kill_decay = 0.995,
                 max_auxin_sources = 300,
                 source_drift = 0.04,
                 dim = 2):
        self.domain = domain
        self.darts_per_step = darts_per_step
        self.start_size = start_size
        self.dim = dim

        self.new_vein_distance = new_vein_distance
        self.birth_distance_v = birth_distance_v
        self.birth_distance_a = birth_distance_a
        self.kill_distance = kill_distance

        self.decay_type = decay_type
        self.distance_decay = distance_decay
        self.width_decay = width_decay
        self.kill_decay = kill_decay
        self.source_drift = source_drift

        if root_node is None:
            root_node = np.zeros(dim)
        self.root_node = root_node
        self.max_auxin_sources = max_auxin_sources
        self.widths = [start_width] * (start_size + 1)

        if colors is None:
            colors = ["#000000"] * (start_size + 1)
        self.colors = colors

        self.vein_nodes, self.auxin_sources = self.initialize_network()
        # enables fast querying
        self.vein_tree = KDTree(self.vein_nodes)
        self.auxin_tree = KDTree(self.auxin_sources)

        self.current_step = 0
        # initialize edges from root node to vein nodes
        self.edges = [Edge(0,
                           i,
                           age = 0,
                           color = colors[i-1],
                           width = start_width) for i in range(1, len(self.vein_nodes) + 1)]
        self.edge_count = len(self.edges)

    def initialize_network(self):
        vein_nodes = []
        # doing as list cause of the domain check
        for i in range(self.start_size):
            vein_nodes.append(self.domain.generate())

        vein_nodes = np.array([self.root_node] + vein_nodes) * self.new_vein_distance

        auxin_sources = []
        # note some darts "miss"
        for _ in range(self.darts_per_step):
            pos = self.domain.generate()
            # sources must be at least birth_distance away from vein nodes
            if cdist(pos[None], vein_nodes).min() > self.birth_distance_a:
                auxin_sources.append(pos)
        auxin_sources = np.array(auxin_sources)
        return vein_nodes, auxin_sources
    
    def _generate_new_veins(self):
        # get the nearest vein node to each auxin source
        vein_dist, vein_indices = self.vein_tree.query(self.auxin_sources)
        diffs = self.auxin_sources - self.vein_nodes[vein_indices]
        diffs /= vein_dist[:, None]

        unique_vein_indices, new_indices = np.unique(vein_indices, return_inverse = True)
        auxin_pull = np.zeros((len(unique_vein_indices), self.dim))
        np.add.at(auxin_pull, new_indices, diffs)

        # normalize auxin pull
        auxin_pull /= np.linalg.norm(auxin_pull, axis=1)[:, None]

        # new nodes are new_vein_distance away from indexed vein nodes
        new_vein_nodes = self.vein_nodes[unique_vein_indices] + auxin_pull * self.new_vein_distance
        # check if new nodes are within domain
        within = np.array([self.domain.within(node) for node in new_vein_nodes])
        if within.sum() > 0:
            new_vein_nodes = new_vein_nodes[within]

            # remove vein nodes that could not be generated
            unique_vein_indices = unique_vein_indices[within]

            new_colors = [self.colors[idx] for idx in unique_vein_indices]
            new_widths = [self.widths[idx] * self.width_decay for idx in unique_vein_indices]

            new_edges = [Edge(idx,
                            self.edge_count + i,
                            age = self.current_step,
                            color = new_colors[i],
                            width = new_widths[i],
                            ) for i, idx in enumerate(unique_vein_indices)]
            self.colors += new_colors
            self.widths += new_widths
            self.edges += new_edges
            self.edge_count += len(new_edges)
            self.vein_nodes = np.concatenate([self.vein_nodes, new_vein_nodes])
        self.vein_tree = KDTree(self.vein_nodes)

    def _generate_auxin_sources(self):
        if len(self.auxin_sources) < self.max_auxin_sources:
            # add new auxin sources
            new_points = []
            for _ in range(self.darts_per_step):
                pos = self.domain.generate()
                # sources must be at least birth_distance away from vein nodes
                vein_ball = self.vein_tree.query_ball_point(pos,
                                                            self.birth_distance_v)
                auxin_ball = self.auxin_tree.query_ball_point(pos,
                                                            self.birth_distance_a)
                if len(vein_ball) == 0 and len(auxin_ball) == 0:
                    new_points.append(pos)
            if new_points:
                new_points = np.array(new_points)
                self.auxin_sources = np.concatenate([self.auxin_sources, new_points])
        if self.source_drift > 0:
            new_auxin_sources = np.random.randn(*self.auxin_sources.shape) * self.source_drift + self.auxin_sources.copy()
            # check domain
            within = np.array([self.domain.within(node) for node in new_auxin_sources])
            if within.sum() > 0:
                self.auxin_sources[within] = new_auxin_sources[within]

        # update this outside of loop cause removals
        self.auxin_tree = KDTree(self.auxin_sources)

    def _update_params(self):
        if self.decay_type == "exponential":
            self.new_vein_distance *= self.distance_decay
            self.birth_distance_v *= self.distance_decay
            self.birth_distance_a *= self.distance_decay
            self.source_drift *= self.distance_decay
        elif self.decay_type == "scale_free":
            self.new_vein_distance = scale_free_decay(self.new_vein_distance, self.current_step)
            self.birth_distance_v = scale_free_decay(self.birth_distance_v, self.current_step)
            self.birth_distance_a = scale_free_decay(self.birth_distance_a, self.current_step)

            self.source_drift = scale_free_decay(self.source_drift, self.current_step)

        # kill distance always decays exponentially
        self.kill_distance *= self.kill_decay


    def step(self):
        self.current_step += 1
        self._generate_new_veins()

        # kill auxin sources that are too close to vein nodes
        kill_indices = self.vein_tree.query_ball_tree(self.auxin_tree, self.kill_distance)
        kill_indices = {i for indices in kill_indices for i in indices}
        kill_indices = list(kill_indices)
        # TODO : should i regenerate the tree?
        self.auxin_sources = np.delete(self.auxin_sources, kill_indices, axis=0)

        self._generate_auxin_sources()
        self._update_params()


def plot_edges(edges,
               points,
               ax=None,
               background_color = "#061530",
               out_size = (30, 30),
               alpha_scale = 0.997,
               **kwargs):

    if ax is None:
        ax = plt.gca()
    
    points = np.asarray(points)
    
    edge_indices = np.array([(edge.source, edge.target) for edge in edges])
    lines = points[edge_indices]
    
    ages = np.array([edge.age for edge in edges])
    alphas = alpha_scale ** ages
    colors = [edge.color for edge in edges]
    #widths = 0.4 * (np.log(ages.max() - ages + 1) + 1)
    widths = [edge.width for edge in edges]

    lc = LineCollection(lines,
                        linewidths = widths,
                        colors = colors,
                        alpha = alphas,
                        joinstyle = "miter",
                        capstyle = "round",
                        **kwargs)
    ax.add_collection(lc)

    ax.set_xlim(points[:, 0].min() - 0.25, points[:, 0].max() + 0.25)
    ax.set_ylim(points[:, 1].min() - 0.25, points[:, 1].max() + 0.25)
            
    ax.set_facecolor(background_color)
    ax.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(out_size)
    fig.patch.set_facecolor(background_color)
    fig.savefig("images/hyphae.png", dpi=300)
       
    return ax


def create_mp4_from_plot(edges, points, max_age=None, fps=30, filename='edge_animation.mp4', **plot_kwargs):
    """
    Create an MP4 video from the plot_edges function output, filtering edges by age.
    
    Parameters:
    edges (list): List of Edge objects.
    points (array-like): Array of shape (N, 2) or (N, 3) where N is the number of points.
    max_age (int, optional): Maximum age to consider. If None, use the maximum age in the edges.
    fps (int): Frames per second for the output video.
    filename (str): Name of the output MP4 file.
    **plot_kwargs: Additional keyword arguments to pass to plot_edges().
    """
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.close()  # Prevent the empty figure from being displayed
    # set background color
    fig.patch.set_facecolor("#061530")
    
    if max_age is None:
        max_age = max(edge.age for edge in edges)
    
    num_frames = max_age + 1  # +1 to include age 0
    
    def animate(frame):
        ax.clear()
        # Filter edges based on age
        edges_to_plot = [edge for edge in edges if edge.age <= frame]
        
        plot_edges(edges_to_plot, points, ax=ax, **plot_kwargs)
        ax.set_title(f"Edges Plot (Age <= {frame}, Frame {frame + 1}/{num_frames})")

        return ax,

    anim = FuncAnimation(fig, animate,
                         frames=num_frames,
                         #blit=True,
                         )
    
    # Set up the writer
    writer = writers['ffmpeg'](fps=fps)
    
    # Save the animation
    anim.save(filename, writer=writer)
    
    print(f"Animation saved as {filename}")


if __name__ == "__main__":
    import os
    n_steps = 1000

    os.makedirs("images", exist_ok=True)
    
    # within domain is a function that returns True if the point is in the domain
    within_domain = lambda x: (x**2).sum() < 25
    # use polar coordinates to generate points
    def generate_in_domain():
        r = np.random.rand() * 5
        theta = np.random.rand() * 2 * np.pi
        return np.array([r * np.cos(theta), r * np.sin(theta)])
    
    domain = Domain(within_domain, generate_in_domain)

    colors = ["#AF47D2", "#FF8F00", "#FFDB00", "#3AA6B9", "#FF9EAA"]

    net = Network(domain,
                  colors = colors
                  )
    
    pbar = tqdm(range(n_steps))

    for _ in range(n_steps):
        net.step()
        auxin_sources = net.auxin_sources.shape[0]
        nodes = net.vein_nodes.shape[0]
        pbar.set_description(f"A: {auxin_sources} N: {nodes}")
        pbar.update()

    ax = plot_edges(net.edges, net.vein_nodes)
    # create_mp4_from_plot(net.edges,
    #                      net.vein_nodes,
    #                      filename='images/hyphae.mp4')

