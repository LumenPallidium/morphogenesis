import numpy as np
import trimesh

class Domain:
    def __init__(self,
                 within_func,
                 generate_func,
                 volume = 1):
        self.within_func = within_func
        self.generate_func = generate_func
        self.volume = volume

    def within(self, pos):
        return self.within_func(pos)
    
    def generate(self, **kwargs):
        return self.generate_func(**kwargs)
    
    def __add__(self, other):
        total_volume = self.volume + other.volume
        def within_func(x):
            return self.within(x) or other.within(x)
        
        def generate_func():
            if np.random.rand() < (self.volume / total_volume):
                return self.generate()
            else:
                return other.generate()
        return Domain(within_func,
                      generate_func,
                      volume = total_volume)
    
class TrimeshDomain(Domain):
    def __init__(self, mesh_path):
        trimesh_obj = trimesh.load(mesh_path)

        self.mesh_path = mesh_path
        self.trimesh_obj = trimesh_obj

        def within_func(x):
            return self.trimesh_obj.contains([x])[0]
        
        def generate_func():
            sample, _ = trimesh.sample.sample_surface(self.trimesh_obj, 1)
            return sample[0]
        super().__init__(within_func, generate_func)

    def get_natural_scales(self, scale_factor = 0.01):
        """
        Get a "natural scale" for the domain which allows good growth
        speed.
        
        """
        
        extent = self.trimesh_obj.bounds.ptp(axis=0).max()
        random_vertex = self.trimesh_obj.vertices[np.random.randint(0, len(self.trimesh_obj.vertices))]

        params = {"new_vein_distance" : extent * scale_factor,
                 "birth_distance_v" : extent * scale_factor,
                 "birth_distance_a" : extent * scale_factor,
                 "kill_distance" : extent * scale_factor * 4,
                 "root_node" : random_vertex}
        
        return params

def circle2d(radius, center = None):
    """The humble (filled) circle."""
    #TODO : can i make a generic function for the n-sphere?
    if center is None:
        center = np.array([0, 0])
    def within_domain(x):
        return np.linalg.norm(x - center) < radius
    def generate_in_domain(center = center):
        r = np.random.rand() * radius
        theta = np.random.rand() * 2 * np.pi
        return np.array([r * np.cos(theta) + center[0],
                         r * np.sin(theta) + center[1]])
    return Domain(within_domain, generate_in_domain)

def donut2d(outer_radius, inner_radius, center = None, force_void = False):
    """
    A 2D Donut domain.

    By default, only generate points in the annulus between the inner and outer radii.
    However, networs can still grow in the void/donut hole. force_void will force
    the network to avoid the void when growing.
    """
    if center is None:
        center = np.array([0, 0])
    def within_domain(x):
        in_outer = np.linalg.norm(x - center) < outer_radius
        if force_void:
            in_outer = in_outer and (np.linalg.norm(x - center) > inner_radius)
        return in_outer
    def generate_in_domain(center = center):
        r = np.random.rand() * (outer_radius - inner_radius) + inner_radius
        theta = np.random.rand() * 2 * np.pi
        return np.array([r * np.cos(theta) + center[0],
                         r * np.sin(theta) + center[1]])
    return Domain(within_domain, generate_in_domain)


def thick_cylinder_cup(height, outer_radius, thickness, base_thickness, center=None):
    """
    (from Claude)
    Create a thick cylinder (cup-like shape) domain with volume.
    
    Parameters:
    height (float): The height of the cup
    outer_radius (float): The outer radius of the cup at its widest point (the top)
    thickness (float): The thickness of the cup walls
    base_thickness (float): The thickness of the cup's base
    center (np.array): The center point of the base of the cup
    """
    if center is None:
        center = np.array([0, 0, 0])
    
    inner_radius = outer_radius - thickness

    def within_domain(x):
        x_rel = x - center
        r = np.sqrt(x_rel[0]**2 + x_rel[1]**2)
        z = x_rel[2]
        
        # Check if point is within the height range
        if z < 0 or z > height:
            return False
        
        # Check if point is within the base
        if z <= base_thickness:
            return r <= outer_radius
        
        # Check if point is between the outer and inner radii
        return (r <= outer_radius) and (r >= inner_radius or z <= base_thickness)

    def generate_in_domain():
        # Decide whether to generate in the base or the walls
        if np.random.random() < base_thickness / height:
            # Generate in the base
            z = np.random.uniform(0, base_thickness)
            r = outer_radius * np.sqrt(np.random.random())
        else:
            # Generate in the walls
            z = np.random.uniform(base_thickness, height)
            max_r = outer_radius
            min_r = inner_radius
            r = np.random.uniform(min_r, max_r)
        
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y, z]) + center

    return Domain(within_domain, generate_in_domain)


def thick_triangular_base(height, top_inner_width, bottom_inner_width, wall_thickness, center=None):
    if center is None:
        center = np.array([0, 0, 0])

    top_outer_width = top_inner_width + 2 * wall_thickness
    bottom_outer_width = bottom_inner_width + 2 * wall_thickness
    slope = (top_outer_width - bottom_outer_width) / height

    def within_domain(x):
        x_rel = x - center
        z = x_rel[2]

        # Check if point is within the height range
        if z < 0 or z > height:
            return False

        # Calculate the outer and inner widths at this height
        outer_width = bottom_outer_width + slope * z
        inner_width = bottom_inner_width + slope * z

        rot1 = np.array([[np.cos(2 * np.pi/3), np.sin(2 * np.pi/3)],
                        [-np.sin(2 * np.pi/3), np.cos(2 * np.pi/3)]])
        rot2 = np.array([[np.cos(-2 * np.pi/3), np.sin(-2 * np.pi/3)],
                        [-np.sin(-2 * np.pi/3), np.cos(-2 * np.pi/3)]])
        x_rot1, y_rot1 = np.dot(rot1, x_rel[:2])
        x_rot2, y_rot2 = np.dot(rot2, x_rel[:2])

        for x_i in [x_rel[0], x_rot1, x_rot2]:
            if (x_i >= inner_width / 2) and (x_i <= outer_width / 2):
                for y_i in [x_rel[1], y_rot1, y_rot2]:
                    if (y_i >= -np.sqrt(3)*x_i) and (y_i <= np.sqrt(3)*x_i):
                        return True

        return False

    def generate_in_domain(z = None):
        if z is None:
            z = np.random.uniform(0, height)
        outer_width = bottom_outer_width + slope * z
        inner_width = bottom_inner_width + slope * z

        x = np.random.uniform(inner_width / 2, outer_width / 2)
        y = np.random.uniform(-np.sqrt(3)*x, np.sqrt(3)*x)

        roll = np.random.rand()
        if roll < (1/3):
            rot = np.array([[np.cos(2 * np.pi/3), np.sin(2 * np.pi/3)],
                            [-np.sin(2 * np.pi/3), np.cos(2 * np.pi/3)]])
            x, y = np.dot(rot, np.array([x, y]))
        elif roll > (2/3):
            rot = np.array([[np.cos(-2 * np.pi/3), np.sin(-2 * np.pi/3)],
                            [-np.sin(-2 * np.pi/3), np.cos(-2 * np.pi/3)]])
            x, y = np.dot(rot, np.array([x, y]))
        
        return np.array([x, y, z]) + center

    return Domain(within_domain, generate_in_domain)

def thick_cut_ellipsoid(height, base_radius, max_radius, thickness, center=None):

    if center is None:
        center = np.array([0, 0, 0])

    delta_radius = max_radius - base_radius

    def within_domain(x):
        x_rel = x - center
        r = np.sqrt(x_rel[0]**2 + x_rel[1]**2)
        z = x_rel[2]
        
        # Check if point is within the height range
        if z < 0 or z > height:
            return False
        
        arc_position = 4 * (z * (height - z)) / (height**2)
        
        outer_radius = base_radius + arc_position * delta_radius
        inner_radius = outer_radius - thickness
        
        # Check if point is between the outer and inner radii
        return (r <= outer_radius) and (r >= inner_radius)

    def generate_in_domain():
        z = np.random.uniform(0, height)
        arc_position = 4 * (z * (height - z)) / (height**2)
        r = base_radius + arc_position * delta_radius
        
        dr = np.random.uniform(0, thickness)
        r -= dr
        
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y, z]) + center

    return Domain(within_domain, generate_in_domain)

domain_lookup_2d = {
    "circle2d" : {"function" : circle2d,
                  "parameters" : {"radius" : {"type" : "float",
                                              "default" : 4}}},
    "donut2d" : {"function" : donut2d,
                 "parameters" : {"outer_radius" : {"type" : "float",
                                                    "default" : 4},
                                 "inner_radius" : {"type" : "float",
                                                    "default" : 2},
                                 "force_void" : {"type" : "bool",
                                                 "default" : True},}}
}

domain_lookup_3d = {
    "thick_cylinder_cup" : {"function" : thick_cylinder_cup,
                            "parameters" : {"height" : {"type" : "float",
                                                        "default" : 4},
                                            "outer_radius" : {"type" : "float",
                                                              "default" : 4},
                                            "thickness" : {"type" : "float",
                                                           "default" : 0.5},
                                            "base_thickness" : {"type" : "float",
                                                                "default" : 1}}},
    "thick_triangular_base" : {"function" : thick_triangular_base,
                               "parameters" : {"height" : {"type" : "float",
                                                           "default" : 4},
                                               "top_inner_width" : {"type" : "float",
                                                                    "default" : 4},
                                               "bottom_inner_width" : {"type" : "float",
                                                                       "default" : 2},
                                               "wall_thickness" : {"type" : "float",
                                                                   "default" : 0.5}}},
    "thick_cut_ellipsoid" : {"function" : thick_cut_ellipsoid,
                             "parameters" : {"height" : {"type" : "float",
                                                         "default" : 4},
                                             "base_radius" : {"type" : "float",
                                                              "default" : 1.5},
                                             "max_radius" : {"type" : "float",
                                                             "default" : 2.5},
                                             "thickness" : {"type" : "float",
                                                            "default" : 0.3}}},
}