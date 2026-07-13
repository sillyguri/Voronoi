import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def draw_voronoi_diagram(IDList, siteList, boundingBox, voronoiRegions, voronoiVertices, queryList):
    fig, ax = plt.subplots(figsize=(8, 8))
    min_x, min_y = boundingBox[0]
    max_x, max_y = boundingBox[1]

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    cmap = plt.colormaps["Pastel1"].resampled(len(voronoiRegions))
    for i, region in enumerate(voronoiRegions):
        path_data = [(voronoiVertices[p].site.x, voronoiVertices[p].site.y) for p in region.vertices]
        poly = Polygon(path_data, facecolor=cmap(i), edgecolor='black', alpha=0.7)
        poly.set_clip_path(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y))
        ax.add_patch(poly)
        ax.plot(siteList[region.site].x, siteList[region.site].y, 'ro', markersize=1)

    for query in queryList:
        ax.plot(query.site.x, query.site.y, 'go', markersize=1)
        ax.text(query.site.x, query.site.y + 0.05, f"{query.ID}", ha='center', fontsize=3)

    plt.title("Voronoi Analysis Tool: 2D Cartesian Test")
    plt.show()
