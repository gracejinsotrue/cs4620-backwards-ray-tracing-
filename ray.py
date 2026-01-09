import concurrent.futures

import numpy as np

from utils import vec, normalize

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class BVHNode:
    def __init__(
        self, bbox_min, bbox_max, objs, axis=None, left=None, right=None
    ):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.objs = objs
        self.axis = axis
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


class Ray:
    def __init__(self, origin, direction, start=0.0, end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:
    def __init__(
        self,
        k_d,
        k_s=0.0,
        p=20.0,
        k_m=0.0,
        k_a=None,
        texture=None,
        texture_repeat=(1.0, 1.0),
        k_t=0.0, ior=1.5
    ):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient (used when no texture or as multiplier)
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
          texture : (H,W,3) ndarray -- optional texture image (float32 or uint8)
          texture_repeat : (2,) -- number of times to repeat texture in (u,v)
          k_t : transmission coefficient from 0-1 
          ior: index of refraction for materials such as glasses
        """
        self.k_d = np.array(k_d, dtype=np.float64)
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else self.k_d
        self.texture = texture
        self.texture_repeat = tuple(texture_repeat)
        self.k_t = k_t 
        self.ior = ior 

    def sample(self, uv):
        """Sample the diffuse color at UV (u,v in [0,1]). If no texture return k_d."""
        if self.texture is None or uv is None:
            return self.k_d

        tex = self.texture.astype(np.float32)
        if tex.max() > 2.0:
            tex = tex / 255.0

        h, w = tex.shape[0], tex.shape[1]
        u = (uv[0] * self.texture_repeat[0]) % 1.0
        v = (uv[1] * self.texture_repeat[1]) % 1.0

        v = 1.0 - v
        x = u * (w - 1)
        y = v * (h - 1)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        sx = x - x0
        sy = y - y0

        c00 = tex[y0, x0]
        c10 = tex[y0, x1]
        c01 = tex[y1, x0]
        c11 = tex[y1, x1]

        c0 = c00 * (1 - sx) + c10 * sx
        c1 = c01 * (1 - sx) + c11 * sx
        c = c0 * (1 - sy) + c1 * sy
        return c


class Hit:
    def __init__(
        self,
        t,
        point=None,
        normal=None,
        material=None,
        uv=None,
    ):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
          uv : (2,) -- optional texture coordinates
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.uv = uv


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:
    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

        self.bbox_min = self.center - np.array([self.radius] * 3)
        self.bbox_max = self.center + np.array([self.radius] * 3)
        self.centroid = self.center

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # we make a vector from the ray origin to the spehre chenter
        oc = ray.origin - self.center
        # for a ray-sphere intersection:
        # # ||p + tv - c||^2 = r^2
        # (d·d)t^2 + 2(d·oc)t + (oc·oc - r^2) = 0
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(ray.direction, oc)
        c = np.dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        # ifddiscinrimant is negative, return no intersection
        if discriminant < 0:
            return no_hit
        # find the two intersection pointso therwise
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # find first valid intersecton
        # this is the smallest t within the [start,end] range
        t = None
        if ray.start <= t1 <= ray.end:
            t = t1
        elif ray.start <= t2 <= ray.end:
            t = t2
        else:
            return no_hit
        # calculate hit point and noraml
        point = ray.origin + t * ray.direction
        normal = normalize(point - self.center)

        nx, ny, nz = normal
        u = 0.5 + np.arctan2(nz, nx) / (2.0 * np.pi)
        v = 0.5 - np.arcsin(ny) / np.pi
        uv = np.array([u, v], dtype=np.float64)

        return Hit(t, point, normal, self.material, uv)


class Triangle:
    def __init__(self, vs, material, uvs=None):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
          uvs (3,2) -- optional per-vertex UV coordinates (u,v)
        """
        self.vs = vs
        self.material = material
        self.uvs = None if uvs is None else np.array(uvs, dtype=np.float64)

        v0, v1, v2 = vs
        self.bbox_min = np.min(np.vstack([v0, v1, v2]), axis=0)
        self.bbox_max = np.max(np.vstack([v0, v1, v2]), axis=0)
        self.centroid = (v0 + v1 + v2) / 3

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # apply the Möller–Trumbore intersection algorithm
        # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        a, b, c = self.vs
        edge1 = b - a
        edge2 = c - a
        pvec = np.cross(ray.direction, edge2)
        det = np.dot(edge1, pvec)
        if abs(det) < 1e-8:
            return no_hit  # ray is parallel to triangle

        inv_det = 1.0 / det
        s = ray.origin - a
        u = np.dot(s, pvec) * inv_det
        if u < 0 or u > 1:
            return no_hit

        s_x_edge1 = np.cross(s, edge1)
        v = inv_det * np.dot(ray.direction, s_x_edge1)
        if v < 0 or u + v > 1:
            return no_hit

        t = np.dot(edge2, s_x_edge1) * inv_det
        if t < ray.start or t > ray.end:
            return no_hit

        point = ray.origin + t * ray.direction
        normal = np.cross(edge1, edge2)
        normal = normal.astype(float)
        normal /= np.linalg.norm(normal)

        uv = None
        if self.uvs is not None:
            w0 = 1.0 - u - v
            uv = w0 * self.uvs[0] + u * self.uvs[1] + v * self.uvs[2]

        return Hit(t, point, normal, self.material, uv)


class Camera:
    def __init__(
        self,
        eye=vec([0, 0, 0]),
        target=vec([0, 0, -1]),
        up=vec([0, 1, 0]),
        vfov=90.0,
        aspect=1.0,
    ):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect

        # build the camera coordinate system
        # w points from target to eye
        # this is the opposite of view direciton
        w = normalize(eye - target)
        # u points to right
        u = normalize(np.cross(up, w))

        # v points up
        v = np.cross(w, u)

        # create camera-to world transformation matrix
        # ts transformation matrix transforms points from caerma space to world space
        self.M = np.eye(4)
        self.M[0:3, 0] = u
        self.M[0:3, 1] = v
        self.M[0:3, 2] = w
        self.M[0:3, 3] = eye

        # distance to image plane (focal length)

        # for any FOV angle the image plane is at distance d where
        # the height of the image plane is 2*d*tan(vfov/2)
        # therefore set image plane to height to 2
        # aka ranging -1 to 1 as specified in assignmetn docs
        # so d = 1 / tan(vfov/2)
        self.f = 1.0 / np.tan(vfov / 2.0 * np.pi / 180.0)

        # Store uvw basis vectors dimensions
        self.u = u
        self.v = v
        self.w = w

        # self.f = None; # you should set this to the distance from your center of projection to the image plane
        # self.M = np.eye(4);  # set this to the matrix that transforms your camera's coordinate system to world coordinates
  

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """

        # from [0,1] x [0,1], we need to map to image plane coordinates
        # for the image plane,
        #  x ranges from -aspect to aspect, y ranges from -1 to 1
        # flip y because 0,0 top left
        u_coord = -self.aspect + 2.0 * self.aspect * img_point[0]
        v_coord = 1.0 - 2.0 * img_point[1]

        # we need to point image plane in camera coord
        #  image plane is at distance f along -w (the view direction)
        # this means the point in camera space is: (u_coord, v_coord, -f)
        # -> to world coords using the camera basis vectors
        ray_origin = self.eye
        ray_direction = u_coord * self.u + v_coord * self.v - self.f * self.w

        return Ray(ray_origin, ray_direction)


class PointLight:
    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # part 3 diffuse shading with point lights
        # vector from hit point to light
        light_dir = self.position - hit.point
        light_distance = np.linalg.norm(light_dir)
        light_dir = light_dir / light_distance  # normalize

        shadow_origin = hit.point + hit.normal * 1e-4
        shad_ray = Ray(shadow_origin, light_dir, 0.0, float(light_distance))

        shadow_hit = scene.intersect(shad_ray)  # check if smth blocks it
        if shadow_hit.t < no_hit.t:
            return vec([0.0, 0.0, 0.0])

        # lambertian formula is:
        # k_d * I * max(0, n · l) / r^2
        n_dot_l = np.maximum(0, np.dot(hit.normal, light_dir))

        # this is the falloff for light intensity
        intensity = self.intensity / (light_distance**2)

        # definition of diffuse reflection
        diffuse_color = (
            hit.material.sample(hit.uv)
            if hasattr(hit.material, "sample")
            else hit.material.k_d
        )
        diffuse = diffuse_color * intensity * n_dot_l

        # part 4 add specular shading as well

        # sppecular formula: k_s * I * max(0, v · r)^p
        # v is view direction whcih is viewing towards camera
        view_dir = normalize(-ray.direction)
        # r is reflection of light direction about normal
        #  r = 2(n·l)n - l
        reflect_dir = 2 * np.dot(hit.normal, light_dir) * hit.normal - light_dir
        v_dot_r = np.maximum(0, np.dot(view_dir, reflect_dir))
        specular = hit.material.k_s * intensity * (v_dot_r**hit.material.p)

        return diffuse + specular


class AmbientLight:
    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        del ray, scene
        mat = hit.material
        ambient_color = (
            mat.sample(hit.uv) if hasattr(mat, "sample") else mat.k_a
        )
        # Use k_a as multiplier if texture present: (texture * k_a)
        if mat.texture is not None:
            return ambient_color * (mat.k_a if mat.k_a is not None else 1.0)
        return mat.k_a * self.intensity


class Scene:
    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

        if surfs:
            self.kdtree = self.build_bvh(surfs)
        else:
            self.kdtree = None

    def build_bvh(self, objs, depth=0, max_leaf=6):
        bbox_min = np.min([o.bbox_min for o in objs], axis=0)
        bbox_max = np.max([o.bbox_max for o in objs], axis=0)

        if len(objs) <= max_leaf or depth >= 32:
            return BVHNode(bbox_min, bbox_max, objs)

        box = bbox_max - bbox_min
        axis = int(np.argmax(box))

        objs_sorted = sorted(objs, key=lambda o: float(o.centroid[axis]))
        mid = len(objs_sorted) // 2

        left_objs = objs_sorted[:mid]
        right_objs = objs_sorted[mid:]

        if not (left_objs := objs_sorted[:mid]) or not (
            right_objs := objs_sorted[mid:]
        ):
            return BVHNode(bbox_min, bbox_max, objs)

        left = self.build_bvh(left_objs, depth + 1, max_leaf)
        right = self.build_bvh(right_objs, depth + 1, max_leaf)

        return BVHNode(bbox_min, bbox_max, None, axis, left, right)

    def intersect_bvh(self, ray: Ray, node: BVHNode):
        tmin, _ = self.intersect_aabb(ray, node.bbox_min, node.bbox_max)
        if not tmin:
            return no_hit

        if node.is_leaf():
            closest = no_hit
            for tri in node.objs:
                h = tri.intersect(ray)
                if h.t < closest.t:
                    closest = h
            return closest

        left, right = (node.left, node.right)
        assert left is not None and right is not None
        tL = self.intersect_aabb(ray, left.bbox_min, left.bbox_max)
        tR = self.intersect_aabb(ray, right.bbox_min, right.bbox_max)

        hitL = hitR = no_hit
        if tL[0] is not None and tR[0] is not None:
            if tL[0] < tR[0]:
                hitL = self.intersect_bvh(ray, left)
                if hitL.t < tR[0]:
                    return hitL
                hitR = self.intersect_bvh(ray, right)
            else:
                hitR = self.intersect_bvh(ray, right)
                if hitR.t < tL[0]:
                    return hitR
                hitL = self.intersect_bvh(ray, left)
        else:
            if tL[0] is not None:
                hitL = self.intersect_bvh(ray, left)
            if tR[0] is not None:
                hitR = self.intersect_bvh(ray, right)

        return hitL if hitL.t < hitR.t else hitR

    def intersect_aabb(self, ray, bbox_min, bbox_max):
        invD = 1.0 / ray.direction
        t0 = (bbox_min - ray.origin) * invD
        t1 = (bbox_max - ray.origin) * invD
        tmin = np.maximum.reduce(np.minimum(t0, t1))
        tmax = np.minimum.reduce(np.maximum(t0, t1))
        if tmax >= max(tmin, 0.0):
            return tmin, tmax
        return None, None

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # part 2:
        # find closest hit

        if self.kdtree is None:
            closest_hit = no_hit

            # check all surfaces and find the closest intersection
            for surf in self.surfs:
                hit = surf.intersect(ray)
                if hit.t < closest_hit.t:
                    closest_hit = hit

            return closest_hit
        return self.intersect_bvh(ray, self.kdtree)


MAX_DEPTH = 5 # more than 1 because glass...


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    if hit == no_hit:
        return scene.bg_color
    
    color = vec([0, 0, 0])

    for light in lights:
        color += light.illuminate(ray, hit, scene)

    if depth < MAX_DEPTH:
        if hit.material.k_m > 0:
            reflected_dir = (
                ray.direction - 2 * np.dot(ray.direction, hit.normal) * hit.normal
            )
            reflected_ray = Ray(hit.point + 1e-4 * hit.normal, reflected_dir)
            reflected_hit = scene.intersect(reflected_ray)

            reflected_color = shade(
                reflected_ray, reflected_hit, scene, lights, depth + 1
            )
            color += hit.material.k_m * reflected_color

        # refraction for transparent materials (aka just glass for our bum ahh)
        if hit.material.k_t > 0:
            # determine if ray is entering or exiting the material
            n = hit.normal
            eta = 1.0 / hit.material.ior  # ratio of indices
            # for example, air/glass
            cos_i = -np.dot(normalize(ray.direction), n)
            
            if cos_i < 0:  # ray is inside the material, it is exiting
                n = -n
                eta = hit.material.ior  # ratio of indices: glass/air
                cos_i = -cos_i
            
            # this is snells law to compute refracted direction
            sin2_t = eta * eta * (1.0 - cos_i * cos_i)
            # check for total internal reflection
            if sin2_t <= 1.0:  
                cos_t = np.sqrt(1.0 - sin2_t)
                refracted_dir = eta * normalize(ray.direction) + (eta * cos_i - cos_t) * n
                # ran into a problem initially here
                # so we offset the origin just by a little,
                #  inside the surface to avoid self-intersection
                refracted_ray = Ray(hit.point - 1e-4 * n, refracted_dir)
                refracted_hit = scene.intersect(refracted_ray)
                
                refracted_color = shade(
                    refracted_ray, refracted_hit, scene, lights, depth + 1
                )
                color += hit.material.k_t * refracted_color

    return color


def render_subregion(pos, camera, scene, lights, nx, ny):
    x0, y0, x1, y1 = pos
    region = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    for i in range(y1 - y0):
        for j in range(x1 - x0):
            x = x0 + j
            y = y0 + i
            # normalize pixel coords [0,1] x [0,1]
            u = (x + 0.5) / nx
            v = (y + 0.5) / ny

            ray = camera.generate_ray(vec([u, v]))

            # do the intersection
            hit = scene.intersect(ray)

            # updated w/ diffuse shading from part 3
            if hit.t < np.inf:
                # take in all light sources which is the change
                # with part 3
                color = vec([0.0, 0.0, 0.0])
                color += shade(ray, hit, scene, lights)
                region[i, j] = color
            else:
                region[i, j] = scene.bg_color
    return pos, region


def subregion_worker(args):
    return render_subregion(*args)


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    #"""
    TILE_SIZE = 64
    tiles = []
    for y in range(0, ny, TILE_SIZE):
        for x in range(0, nx, TILE_SIZE):
            tiles.append((x, y, min(x + TILE_SIZE, nx), min(y + TILE_SIZE, ny)))

    output_image = np.zeros((ny, nx, 3), np.float32)
    # runs multiple python interpreters on multiple cores

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (x0, y0, x1, y1), tile in executor.map(
            subregion_worker,
            [(t, camera, scene, lights, nx, ny) for t in tiles],
        ):
            output_image[y0:y1, x0:x1] = tile
    # for t in tiles: # trying single thread becuase uh 
    #     (x0, y0, x1, y1), tile = render_subregion(t, camera, scene, lights, nx, ny)
    #     output_image[y0:y1, x0:x1] = tile

    return output_image


def evenly_tile_triangles(
    triangles, axis_u=0, axis_v=2, repeat=(1.0, 1.0), clip=True
):
    """
    map a texture across a group of triangles
    """
    pts = np.vstack([tri.vs for tri in triangles])
    mx = pts.max(axis=0)
    mn = pts.min(axis=0)
    span = mx - mn
    span[span == 0] = 1.0

    for tri in triangles:
        vs = tri.vs
        u = (vs[:, axis_u] - mn[axis_u]) / span[axis_u]
        v = (vs[:, axis_v] - mn[axis_v]) / span[axis_v]
        uv = np.vstack([u, v]).T
        uv *= np.array(repeat, dtype=np.float64)
        if clip:
            uv = np.clip(uv, 0.0, 1.0)
        tri.uvs = uv
