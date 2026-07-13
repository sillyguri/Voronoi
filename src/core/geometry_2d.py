from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass(frozen=True)
class Point2D:
    """Standard Cartesian coordinate."""
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Point2D':
        return self.__mul__(scalar)
@dataclass
class Point3D:
    """Spherical coordinates in radians."""
    phi: float    # Azimuthal angle [0, 2π]
    theta: float  # Polar angle [0, π]

    @property
    def x(self):
        return np.sin(self.theta) * np.cos(self.phi)
    
    def neg(self) -> 'Point3D':
        return Point3D(self.phi + np.pi, np.pi - self.theta)

    @property
    def y(self):
        return np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self):
        return np.cos(self.theta)
    
    def rotate(self, rotation_matrix: np.ndarray) -> 'Point3D':
        # Rotate point via rotation matrix
        vec = np.array([self.x, self.y, self.z])
        rotated_vec = rotation_matrix @ vec
        self.phi = np.arctan2(rotated_vec[1], rotated_vec[0])
        self.theta = np.arccos(np.clip(rotated_vec[2] / np.linalg.norm(rotated_vec), -1.0, 1.0))


# move dag element class from algorithms.py to here


@dataclass
class VoronoiEdge:
    endpoint1: Point2D
    endpoint2: Point2D
    is_infinite: bool
    direction: Point2D = None # Direction vector for infinite edges, represents direction from endpoint1 to endpoint2 

    @property
    def point1(self) -> Point2D:
        return self.endpoint1

    @property
    def point2(self) -> Point2D:
        return self.endpoint2

@dataclass
class VoronoiCell:
    site: Point2D
    vertices: List[Point2D]
    edges: List[VoronoiEdge]
@dataclass
class VoronoiVertex:
    ID: int
    site: Point2D
@dataclass
class Query:
    site: Point2D
    ID: str
@dataclass
class VoronoiRegion:
    site: Point2D
    vertices: List[int]


def circumcenter(triangle: List[Point2D])->Point2D:
    a=triangle[0]
    b=triangle[1]
    c=triangle[2]
    d = 2*(a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y))
    ux = ((a.x**2+a.y**2)*(b.y-c.y)+(b.x**2+b.y**2)*(c.y-a.y)+(c.x**2+c.y**2)*(a.y-b.y))/d
    uy = ((a.x**2+a.y**2)*(c.x-b.x)+(b.x**2+b.y**2)*(a.x-c.x)+(c.x**2+c.y**2)*(b.x-a.x))/d
    return Point2D(ux, uy)
def orient2d(a, b, c):
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

def inside_circumcircle(p: Point2D, triangle: List[Point2D]) -> bool:
    a, b, c = triangle

    mat = np.array([
        [a.x-p.x, a.y-p.y, (a.x-p.x)**2 + (a.y-p.y)**2],
        [b.x-p.x, b.y-p.y, (b.x-p.x)**2 + (b.y-p.y)**2],
        [c.x-p.x, c.y-p.y, (c.x-p.x)**2 + (c.y-p.y)**2]
    ])

    det = np.linalg.det(mat)

    orientation = orient2d(a, b, c)

    if orientation > 0:
        return det > 1e-9
    else:
        return det < -1e-9
EPS = 1e-12

def inside_triangle(p, triangle):
    a, b, c = triangle

    def sign(p1, p2, p3):
        return ((p1.x - p3.x)*(p2.y - p3.y)
              - (p2.x - p3.x)*(p1.y - p3.y))

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
    has_pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)

    return not (has_neg and has_pos)
def cross(p1: Point2D, p2: Point2D)->float:
    return p1.x * p2.y - p1.y * p2.x
def area2D(vertices: List[Point2D]) -> float:
    """Signed shoelace sum for a simple polygon (no 1/2 factor).

    For three vertices ``[a, b, c]`` this equals ``cross(b - a, c - a)``, so
    existing orientation tests (e.g. ``point_left_of_edge``) stay unchanged.
    Vertices should be ordered consistently (counterclockwise gives a positive
    area in a right-handed x-right / y-up frame).
    """
    return abs(sum([cross(vertices[i],vertices[(i+1)%len(vertices)]) for i in range(len(vertices))]))
def angleBetween(otherPoint: Point2D,point: Point2D)->float:
    return np.arctan2(otherPoint.y-point.y,otherPoint.x-point.x)
def distanceBetween(otherPoint: Point2D,point: Point2D)->float:
    return np.hypot(otherPoint.x-point.x, otherPoint.y-point.y)
def find_super_triangle(points: List[Point2D])->List[Point2D]:
    """Find a super triangle that encompasses all the points."""
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    return [
        Point2D(mid_x + 3 * delta_max, mid_y - delta_max),
        Point2D(mid_x, mid_y + 3 * delta_max),
        Point2D(mid_x - 3 * delta_max, mid_y - delta_max)
    ]