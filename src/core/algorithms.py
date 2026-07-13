import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from .geometry_2d import (
    Point2D,
    Query,
    VoronoiRegion,
    VoronoiVertex,
    angleBetween,
    circumcenter,
    distanceBetween,
    find_super_triangle,
    inside_circumcircle,
    inside_triangle,
)

class DAGElement:
    def __init__(self, triangle, neighbors=None):
        """
        triangle: A list/tuple of three points (the vertices themselves)
        neighbors: A list of 3 tuples [(neighbor_obj, edge_index), ...] 
                   where -1 represents no neighbor.
        """
        self.triangle = triangle
        # Initialize neighbors as empty if not provided
        self.neighbors = neighbors if neighbors else [(-1, -1), (-1, -1), (-1, -1)]
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0
def bowyer_watson(siteList: list):
    
    # 1. Setup Super-Triangle
    # Helper takes actual points
    boundingPoints = find_super_triangle(siteList)
    
    # Mutate the input list by appending the bounding points
    siteList.extend(boundingPoints)
    
    # The original sites are at indices 0 to n_sites - 1
    # The bounding points are the last 3 indices
    n_sites = len(siteList) - 3
    insertion_order = random.sample(range(n_sites), n_sites)
    st_p1, st_p2, st_p3 = n_sites, n_sites + 1, n_sites + 2
    
    # DAGElement now stores INDICES in its triangle tuple
    root = DAGElement((st_p1, st_p2, st_p3))
    all_elements = [root]

    # Iterate strictly over the original points using their list position
    for p_idx in insertion_order:
        point_coord = siteList[p_idx]
    
        # 2. Search for the leaf triangle
        # We pass siteList so dag_search can resolve indices to coordinates
        target = dag_search(root, point_coord, siteList)
        
        # 3. Split the found triangle
        v0_idx, v1_idx, v2_idx = target.triangle
        
        t0 = DAGElement((v0_idx, v1_idx, p_idx))
        t1 = DAGElement((v1_idx, v2_idx, p_idx))
        t2 = DAGElement((v2_idx, v0_idx, p_idx))
        
        new_tris = [t0, t1, t2]
        all_elements.extend(new_tris)
        
        # Update internal neighbors (using your corrected mapping)
        t0.neighbors = [target.neighbors[0], (t1, 2), (t2, 1)]
        t1.neighbors = [target.neighbors[1], (t2, 2), (t0, 1)]
        t2.neighbors = [target.neighbors[2], (t0, 2), (t1, 1)]

        # Update external neighbors
        for i, (ext_nb, ext_edge) in enumerate(target.neighbors):
            if ext_nb != -1:
                new_tris[i].neighbors[0] = (ext_nb, ext_edge)
                ext_nb.neighbors[ext_edge] = (new_tris[i], 0)

        target.children = new_tris
        
        # 4. Legalize Edges
        queue = [t0, t1, t2]
        while queue:
            curr = queue.pop()
            
            # Find point by index
            queryIndex = curr.triangle.index(p_idx)
            
            opp_nb, opp_edge = curr.neighbors[(queryIndex + 1) % 3]
            
            if opp_nb == -1:
                continue
                
            otherVertex_idx = opp_nb.triangle[(opp_edge - 1) % 3]
            
            # RESOLVE TO COORDINATES for the helper function
            otherVertex_coord = siteList[otherVertex_idx]
            curr_triangle_coords = [siteList[idx] for idx in curr.triangle]
            
            if inside_circumcircle(otherVertex_coord, curr_triangle_coords):
                
                # Flipped triangles (storing indices)
                n1_tri = (p_idx, curr.triangle[(queryIndex + 1) % 3], otherVertex_idx)
                n2_tri = (p_idx, otherVertex_idx, curr.triangle[(queryIndex - 1) % 3])
                
                n1 = DAGElement(n1_tri)
                n2 = DAGElement(n2_tri)
                all_elements.extend([n1, n2])
                
                n1.neighbors = [
                    curr.neighbors[queryIndex],
                    opp_nb.neighbors[(opp_edge + 1) % 3],
                    (n2, 0)
                ]
                n2.neighbors = [
                    (n1, 2),
                    opp_nb.neighbors[(opp_edge - 1) % 3],
                    curr.neighbors[(queryIndex - 1) % 3]
                ]
                
                curr.children.extend([n1, n2])
                opp_nb.children.extend([n1, n2])
                
                updates = [
                    (curr.neighbors[queryIndex], n1, 0),
                    (curr.neighbors[(queryIndex - 1) % 3], n2, 2),
                    (opp_nb.neighbors[(opp_edge - 1) % 3], n2, 1),
                    (opp_nb.neighbors[(opp_edge + 1) % 3], n1, 1) 
                ]
                
                for (ext_nb, ext_edge), new_obj, new_edge in updates:
                    if ext_nb != -1:
                        ext_nb.neighbors[ext_edge] = (new_obj, new_edge)
                        
                queue.extend([n1, n2])

    # 5. Extract Final Triangles 
    finalTriangles = []
    allTriangles = []
    finalLeafElements = []
    # Set of bounding point indices
    st_set = {st_p1, st_p2, st_p3}

    for elem in all_elements:
        if elem.is_leaf():
            allTriangles.append(elem.triangle)
            if not any(v in st_set for v in elem.triangle):
                finalTriangles.append(elem.triangle)
                finalLeafElements.append(elem)

    return (finalTriangles, allTriangles, root, finalLeafElements)


def voronoi_new(siteList: list, boundingbox: list):
    # 1. Capture the original count before mutation
    num_original_sites = len(siteList)
    
    # 2. Compute bounding points (using your logic from image_1c3458)
    lowX = 2.1 * boundingbox[0][0] - 1.1 * boundingbox[1][0]
    lowY = 2.1 * boundingbox[0][1] - 1.1 * boundingbox[1][1]
    uppX = 2.1 * boundingbox[1][0] - 1.1 * boundingbox[0][0]
    uppY = 2.1 * boundingbox[1][1] - 1.1 * boundingbox[0][1]
    siteList.append(Point2D(lowX,lowY))
    siteList.append(Point2D(uppX,uppY))
    siteList.append(Point2D(lowX,uppY))
    siteList.append(Point2D(uppX,lowY))
    # These will be appended inside bowyer_watson or manually here. 
    # Since bowyer_watson mutates the list, we pass the original siteList.
    # Note: Ensure find_super_triangle returns these 4 or use the 3 from Bowyer-Watson.
    (triangles, allTriangles, root, _) = bowyer_watson(siteList)
    
    # 3. Initialize Voronoi Vertex data
    # vertices: list of VoronoiVertex objects (circumcenters)
    # vertexDict: maps site index -> list of triangle indices it belongs to
    vertices = []
    vertexDict = {i: [] for i in range(num_original_sites)}
    
    for i, tri_indices in enumerate(triangles):
        # Resolve indices to coordinates for circumcenter calculation
        tri_coords = [siteList[idx] for idx in tri_indices]
        
        # Create the Voronoi vertex (the circumcenter of the Delaunay triangle)
        v_center = circumcenter(tri_coords)
        vertices.append(VoronoiVertex(i, v_center))
        
        # 4. Map 'real' sites to this Voronoi vertex
        for s_idx in tri_indices:
            if s_idx < num_original_sites:
                vertexDict[s_idx].append(i)
    
    # 5. Build Voronoi Regions
    voronoi_regions = []
    for s_idx in range(num_original_sites):
        site_p = siteList[s_idx]
        
        # Sort the circumcenters around the site to form a convex polygon
        # This uses your exact atan2 logic, but with index-based lookup
        vertexDict[s_idx].sort(key=lambda v_idx: angleBetween(vertices[v_idx].site,site_p))
        
        # Create region for the site at siteList[s_idx]
        voronoi_regions.append(VoronoiRegion(s_idx, vertexDict[s_idx]))
        
    return [voronoi_regions, vertices,siteList,allTriangles,root]


def dag_search(root, query_point, siteList: list):
    current_node = root
    while not current_node.is_leaf():
        for child in current_node.children:
            child_coords = [siteList[idx] for idx in child.triangle]
            if inside_triangle(query_point, child_coords):
                current_node = child
                #found_child = True
                break
    return current_node

class QueryElement:
    def __init__(self, triangle, children, regionID=-1):
        """
        triangle: A list/tuple of three Point2D objects (NOT indices)
        children: list of child QueryElement nodes
        regionID: index into IDList for leaf nodes (-1 for internal/gap nodes)
        """
        self.triangle = triangle
        self.children = children
        self.regionID = regionID

    def is_leaf(self):
        return len(self.children) == 0
def reverseDAGHelper(currNodes: List[QueryElement], boundingTriangle: List[int], siteList: List[Point2D], numRecursion=0) -> QueryElement:
    if len(currNodes) == 1:
        return currNodes[0]
    if len(currNodes) == 2:
        # wrap both in a single root triangle that covers both
        # collect all unique vertices from both triangles
        allVerts = list(dict.fromkeys(currNodes[0].triangle + currNodes[1].triangle))
        # find 3 that form a valid covering triangle via ear clip / just pick the outer 3
        # simplest: use the 3 vertices that form the largest triangle
        best = None
        bestArea = -1
        for i in range(len(allVerts)):
            for j in range(i+1, len(allVerts)):
                for k in range(j+1, len(allVerts)):
                    A = siteList[allVerts[i]]
                    B = siteList[allVerts[j]]
                    C = siteList[allVerts[k]]
                    area = abs((B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x))
                    if area > bestArea:
                        bestArea = area
                        best = [allVerts[i], allVerts[j], allVerts[k]]
        root = QueryElement(best, currNodes)
        return root
    triangleDict = {}
    visited = {}
    independentSet = {}
    for node in currNodes:
        for vertex in node.triangle:
            if vertex not in triangleDict:
                triangleDict[vertex] = []
                visited[vertex] = False
                independentSet[vertex] = False
            triangleDict[vertex].append(node)
    nextNodes = []
    for vertex in triangleDict:
        if vertex in boundingTriangle:
            continue
        if visited[vertex]:
            continue
        if len(triangleDict[vertex]) > 12:
            continue
        visited[vertex] = True
        independentSet[vertex] = True
        newPoints = []
        for node in triangleDict[vertex]:
            for otherVertex in node.triangle:
                if otherVertex != vertex and otherVertex not in newPoints:
                    newPoints.append(otherVertex)
                    visited[otherVertex] = True
        newPoints.sort(key=lambda otherPoint: angleBetween(siteList[otherPoint], siteList[vertex]))
        while len(newPoints) > 3:
            found=False
            for i in range(len(newPoints)):
                a = newPoints[i]
                b = newPoints[(i + 1) % len(newPoints)]
                c = newPoints[(i + 2) % len(newPoints)]
                A = siteList[a]
                B = siteList[b]
                C = siteList[c]
                crossValue = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)
                if 0 > crossValue:
                    continue
                validTriangle = True
                for point in newPoints:
                    if point in (a, b, c):
                        continue
                    if inside_triangle(siteList[point], [A, B, C]):
                        validTriangle = False
                        break
                if validTriangle:
                    newPoints.remove(b)
                    nextNodes.append(QueryElement([a, b, c], triangleDict[vertex]))
                    found=True
                    break
            if not found:
                raise NotImplementedError
        nextNodes.append(QueryElement(
        [newPoints[0], newPoints[1], newPoints[2]], 
        triangleDict[vertex]))
    for node in currNodes:
        stillExists = True
        for vertex in node.triangle:
            if independentSet[vertex]:
                stillExists = False
        if stillExists:
            nextNodes.append(node)
    numRecursion += 1
    return reverseDAGHelper(nextNodes, boundingTriangle, siteList, numRecursion)

def convexHull(points: List[Point2D]) -> List[int]:
    """Returns indices into points list forming the convex hull in CCW order."""
    n = len(points)
    if n < 3:
        return list(range(n))
    pivot_idx = min(range(n), key=lambda i: (points[i].y, points[i].x))
    pivot = points[pivot_idx]
    indices = sorted(range(n), key=lambda i: angleBetween(points[i], pivot))
    hull = []
    for idx in indices:
        while len(hull) >= 2:
            a, b, c = points[hull[-2]], points[hull[-1]], points[idx]
            cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(idx)
    return hull

def triangulateVoronoiRegions(voronoiRegions, voronoiVertices, boundingBox):
    """
    Triangulates each Voronoi region via fan triangulation (valid since convex).
    Fills gap between convex hull of Voronoi vertices and a super-triangle.
    Extends voronoiVertices in-place with super-triangle vertices.
    Returns (leafNodes, boundingTriangleIndices, voronoiVertices).
    All triangles store integer indices into voronoiVertices.
    """
    min_x, min_y = boundingBox[0]
    max_x, max_y = boundingBox[1]
    dx = max_x - min_x
    dy = max_y - min_y
    delta = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # append super triangle vertices to voronoiVertices
    base = len(voronoiVertices)
    st_idx = [base, base + 1, base + 2]
    voronoiVertices.append(VoronoiVertex(base,     Point2D(mid_x,             mid_y + 3 * delta)))
    voronoiVertices.append(VoronoiVertex(base + 1, Point2D(mid_x - 3 * delta, mid_y - delta)))
    voronoiVertices.append(VoronoiVertex(base + 2, Point2D(mid_x + 3 * delta, mid_y - delta)))

    siteList = [v.site for v in voronoiVertices]
    leafNodes = []

    # 1. fan-triangulate each Voronoi region using indices from region.vertices
    for regionID, region in enumerate(voronoiRegions):
        pts = region.vertices  # already indices into voronoiVertices
        if len(pts) < 3:
            continue
        for i in range(1, len(pts) - 1):
            leafNodes.append(QueryElement([pts[0], pts[i], pts[i + 1]], [], regionID=regionID))

    # 2. fill gap: convex hull of original Voronoi vertices -> super triangle
    originalCount = base  # voronoiVertices before super triangle was appended
    hullIndices = convexHull(siteList[:originalCount])

    for i in range(len(hullIndices)):
        a = hullIndices[i]
        b = hullIndices[(i + 1) % len(hullIndices)]
        mid = Point2D((siteList[a].x + siteList[b].x) / 2,
                      (siteList[a].y + siteList[b].y) / 2)
        nearest_st = min(st_idx, key=lambda v: distanceBetween(siteList[v], mid))
        leafNodes.append(QueryElement([a, b, nearest_st], [], regionID=-1))

    # 3. fill corners near each super triangle vertex
    for stv in st_idx:
        sorted_hull = sorted(hullIndices, key=lambda p: distanceBetween(siteList[p], siteList[stv]))
        leafNodes.append(QueryElement([stv, sorted_hull[0], sorted_hull[1]], [], regionID=-1))

    return leafNodes, st_idx, siteList

class VoronoiSetting:
    def __init__(self, IDList, PointList, BoundingBox, QueryList):
        self.IDList = IDList
        self.PointList = PointList.copy()
        self.boundingBox = BoundingBox
        [self.VoronoiRegions, self.VoronoiVertices, self.QueryPointList, triangles, root] = voronoi_new(PointList, BoundingBox)
        # build Kirkpatrick hierarchy on Voronoi region triangulation
        # triangulateVoronoiRegions extends VoronoiVertices with super-triangle vertices
        leafNodes, boundingTriangle, self.KirkpatrickSiteList = triangulateVoronoiRegions(
            self.VoronoiRegions, self.VoronoiVertices, BoundingBox)
        self.queryRoot = reverseDAGHelper(leafNodes, boundingTriangle, self.KirkpatrickSiteList)
        self.queryList = [Query(query, self.IDList[self.searchPoint(query)]) for query in QueryList]

    def searchPoint(self, query: Point2D):
        currNode = dag_search(self.queryRoot, query, self.KirkpatrickSiteList)
        if currNode.regionID == -1 or currNode.regionID >= len(self.IDList):
            # gap triangle — brute force fallback
            return min(range(len(self.PointList)),
                       key=lambda i: distanceBetween(self.PointList[i], query))
        return currNode.regionID

    def drawing(self):
        draw_voronoi_diagram(self.IDList, self.PointList, self.boundingBox, self.VoronoiRegions, self.VoronoiVertices, self.queryList)

def draw_voronoi_diagram(IDList,siteList,boundingBox,voronoiRegions,voronoiVertices,queryList):
    fig, ax = plt.subplots(figsize=(8, 8))
    min_x,min_y=boundingBox[0]
    max_x,max_y=boundingBox[1]
    
    # Navigation: Reset Zoom logic centers the view
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Use a colormap for distinct color requirements
    cmap = plt.cm.get_cmap('Pastel1', len(voronoiRegions))
    for i, region in enumerate(voronoiRegions):
        # Convert Point2D objects to coordinate tuples for Matplotlib
        path_data = [(voronoiVertices[p].site.x, voronoiVertices[p].site.y) for p in region.vertices]
        
        # Draw the polygonal region
        poly = Polygon(path_data, facecolor=cmap(i), edgecolor='black', alpha=0.7)
        poly.set_clip_path(Rectangle((min_x,min_y), max_x - min_x, max_y - min_y)
)
        ax.add_patch(poly)

        # Plot the site point
        ax.plot(siteList[region.site].x, siteList[region.site].y, 'ro', markersize=1)
        
        # Display area (calculated via Shoelace’s formula)[cite: 1]
        # ax.text(siteList[region.site].x,siteList[region.site].y + 0.2, f"{IDList[i]}", 
        #         ha='center', fontsize=6)
    for query in queryList:
        ax.plot(query.site.x, query.site.y, 'go', markersize=1)
        ax.text(query.site.x,query.site.y + 0.05, f"{query.ID}", 
                ha='center', fontsize=3)
    plt.title("Voronoi Analysis Tool: 2D Cartesian Test")
    plt.show()
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'ui', 'uscities.csv'))
    df = df[(df['lat'] > 22) & (df['lat'] < 50) & (df['lng'] > -130) & (df['lng'] < -67)]
    df = df[(df['population'] > 5000)]
    IDList = list(df['city_ascii'])
    pointList = [Point2D(row['lng']*100+random.uniform(-1e-9,1e-9), row['lat']*100+random.uniform(-1e-9,1e-9)) for _, row in df.iterrows()]
    boundingBox=[[-13000,2200],[-6700,5000]]
    queryList=[]
    for _ in range(0):
        queryList.append(Point2D(random.uniform(-13000,-6700),random.uniform(2200,5000)))
    setting=VoronoiSetting(IDList,pointList,boundingBox,queryList)
    setting.drawing()