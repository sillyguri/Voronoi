import copy
from typing import List

import numpy as np

from .algorithms import DAGElement, bowyer_watson
from .geometry_2d import Point2D, Point3D, VoronoiRegion, VoronoiVertex
def stereographic_project(points: List[Point3D], point: Point3D)->List[Point2D]:
    points_copy = copy.deepcopy(points)
    points_copy = rotate_points(points_copy, rotation_matrix(point))
    projection = []
    for p in points_copy:
        if abs(p.z - 1) < 1e-9:
            pass
        else:
            projection.append(Point2D(p.x/(1-p.z), p.y/(1-p.z)))
    return projection
def rotation_matrix(point: Point3D) -> np.ndarray:
    # Returns a rotation matrix to rotate point to (0, 0, 1)
    if np.isclose(point.z, -1):
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    else:
        return np.array([
            [1-point.x**2/(1+point.z), -point.x*point.y/(1+point.z), -point.x],
            [-point.x*point.y/(1+point.z), 1-point.y**2/(1+point.z), -point.y],
            [point.x, point.y, point.z]
        ])
def random_point_on_sphere()->Point3D:
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2*np.pi)
    costheta = rng.uniform(-1, 1)
    theta = np.acos(costheta)
    return Point3D(phi, theta)

def circumcenter(v1: Point3D, v2: Point3D, v3: Point3D) -> Point3D:
    """Return the point on the sphere that is equidistant from v1, v2, and v3 on the opposite side of the sphere from p."""
    A = np.array([
        [v1.x, v1.y, v1.z],
        [v2.x, v2.y, v2.z],
        [v3.x, v3.y, v3.z]
    ])
    B = np.array([
        [v1.x**2 + v1.y**2 + v1.z**2],
        [v2.x**2 + v2.y**2 + v2.z**2],
        [v3.x**2 + v3.y**2 + v3.z**2]
    ])
    # Solve for the circumcenter using Cramer's rule
    D = np.linalg.det(A)
    Dx = np.linalg.det(np.hstack((B, A[:, 1:])))
    Dy = np.linalg.det(np.hstack((A[:, :1], B, A[:, 2:])))
    Dz = np.linalg.det(np.hstack((A[:, :2], B)))
    
    if D == 0:
        raise ValueError("The points are collinear")
    
    cx, cy, cz = Dx/D, Dy/D, Dz/D
    norm = np.sqrt(cx**2 + cy**2 + cz**2)
    cx, cy, cz = cx/norm, cy/norm, cz/norm
    return Point3D(np.atan2(cy, cx), np.acos(max(-1.0, min(1.0, cz))))

def rotate_points(points: List[Point3D], rot_matrix: np.ndarray)->List[Point3D]:
    # Rotate points via rotation matrix
    for p in points:
        p.rotate(rot_matrix)
    return points

def haversine(v1: Point3D,v2: Point3D)->float:
    return 2*np.asin(((1-np.cos(v2.theta-v1.theta)+np.sin(v1.theta)*np.sin(v2.theta)*(1-np.cos(v2.phi-v1.phi)))/2)**0.5)

def sphericalAngle(v1: Point3D,v2: Point3D,v3: Point3D)->float:
    a=haversine(v2,v3)
    b=haversine(v1,v3)
    c=haversine(v1,v2)
    return np.acos((np.cos(a)-np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)))

def calculate_spherical_area(vertices:List[Point3D])->float:
    totAngle=0
    for i in range(len(vertices)):
        totAngle+=sphericalAngle(vertices[(i-1)%len(vertices)],vertices[i],vertices[(i+1)%len(vertices)])
    return totAngle-np.pi*(len(vertices)-2)

def angle_key(v_idx, vertices, site_vec, ref1, ref2):
    v = vertices[v_idx].site
    v_vec = np.array([v.x, v.y, v.z])
    v_tan = v_vec - np.dot(v_vec, site_vec) * site_vec
    return np.atan2(np.dot(v_tan, ref2), np.dot(v_tan, ref1))

def bowyer_watson_3D(siteList: List[Point3D]):
    centroid = np.mean([[p.x, p.y, p.z] for p in siteList], axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        point1 = random_point_on_sphere()
    else:
        c = centroid / norm
        point1 = Point3D(np.atan2(c[1], c[0]), np.acos(float(np.clip(c[2], -1.0, 1.0))))
    point2 = point1.neg()

    project1 = stereographic_project(siteList, point1)
    triangles1, _, _, leaf_elements1 = bowyer_watson(project1)
    points = []
    convex_hull = []
    # Check which edges of the triangulation only have one triangle connected to them.
    for tri in leaf_elements1:
        for i, j in tri.neighbors:
            if i == -1:
                points.append(tri.triangle[j])
                points.append(tri.triangle[(j+1)%3])
    # Remove duplicates from points
    points = list(set(points))
    for p in points:
        convex_hull.append(siteList[p])
    # Perform Delaunay on remaining points, and add the convex hull triangles back in
    project2 = stereographic_project(siteList, point2)
    triangles2 = bowyer_watson(project2)[0]
    seen = set()
    finalTriangles = []
    for t in triangles1 + triangles2:
        key = frozenset(t)
        if key not in seen:
            seen.add(key)
            finalTriangles.append(t)

    return finalTriangles

def voronoi_diagram_3D(siteList: List[Point3D]):
    num_original_sites = len(siteList)
    triangles = bowyer_watson_3D(siteList)

    # 1. We first start computing the Voronoi vertices by finding the circumcenter of each triangle.
    vertices = []
    vertexDict = {i: [] for i in range(num_original_sites)}

    # We find the centroid of all sites to determine the "inside" and "outside" regions of the delaunay hull, letting us find the circumcenters for the Voronoi diagram.
    centroid = np.array([[s.x, s.y, s.z] for s in siteList[:num_original_sites]]).mean(axis=0)

    for tri in triangles:
        v1, v2, v3 = siteList[tri[0]], siteList[tri[1]], siteList[tri[2]]
        # 2. Pick the circumcenter on the opposite side of the triangle's plane from the centroid.
        cc = circumcenter(v1, v2, v3)
        cc_vec = np.array([cc.x, cc.y, cc.z])
        p1 = np.array([v1.x, v1.y, v1.z])
        # Find normal vector for plane of triangle
        normal = np.cross(np.array([v2.x - v1.x, v2.y - v1.y, v2.z - v1.z]),
                          np.array([v3.x - v1.x, v3.y - v1.y, v3.z - v1.z]))
        # check if circumcenter is on the same side of the plane as the centroid, and if so, flip it to the other side
        centroid_side = np.dot(centroid - p1, normal)
        cc_side = np.dot(cc_vec - p1, normal)
        if np.sign(cc_side) == np.sign(centroid_side):
            cc = cc.neg()
        
        # Add the vertex everywhere it needs to be
        v_idx = len(vertices)
        vertices.append(VoronoiVertex(v_idx, cc))
        for s_idx in tri:
            vertexDict[s_idx].append(v_idx)

    # 3. Build Voronoi regions
    # We sort the Voronoi vertices around each site by considering the tangent plane at the site, and sorting by angles formed in that plane
    voronoi_regions = []
    for i in range(num_original_sites):
        site = siteList[i]
        site_vec = np.array([site.x, site.y, site.z])

        # Build two orthogonal basis vectors in the tangent plane at site
        arbitrary = np.array([0.0, 0.0, 1.0]) if abs(site_vec[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ref1 = np.cross(site_vec, arbitrary)
        ref1 /= np.linalg.norm(ref1)
        ref2 = np.cross(site_vec, ref1)

        vertexDict[i].sort(key=lambda v_idx: angle_key(v_idx, vertices, site_vec, ref1, ref2))
        voronoi_regions.append(VoronoiRegion(i, vertexDict[i]))

    return [voronoi_regions, vertices, siteList]