import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from core.geometry_3d import voronoi_diagram_3D


def slerp(p1, p2, num_points):
    p1 = np.array(p1) / np.linalg.norm(p1)
    p2 = np.array(p2) / np.linalg.norm(p2)
    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-9:
        return np.array([p1])
    t = np.linspace(0, 1, num_points)
    return (np.sin((1-t)*omega) / np.sin(omega))[:, None] * p1 + \
           (np.sin(t*omega) / np.sin(omega))[:, None] * p2


class SphericalVoronoiViewer:
    def __init__(self, site_ids, site_list, resolution=400):
        regions, vertices, _ = voronoi_diagram_3D(site_list)

        self.regions = regions
        self.site_ids = site_ids
        self.cmap = plt.colormaps["tab20"].resampled(len(regions))
        self.rotation = np.eye(3)
        self.drag_start = None

        self.site_vecs = np.array([[p.x, p.y, p.z] for p in site_list])
        self.vertex_vecs = np.array([[v.site.x, v.site.y, v.site.z] for v in vertices])

        self.edge_arcs = []
        seen_edges = set()
        for region in regions:
            for k in range(len(region.vertices)):
                a, b = region.vertices[k], region.vertices[(k + 1) % len(region.vertices)]
                key = (min(a, b), max(a, b))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                va, vb = vertices[a].site, vertices[b].site
                self.edge_arcs.append(slerp([va.x, va.y, va.z], [vb.x, vb.y, vb.z], 60))

        self.n_sites = len(site_list)
        self.region_of_site = np.full(self.n_sites, -1, dtype=int)
        for i, r in enumerate(regions):
            self.region_of_site[r.site] = i

        N = resolution
        xs = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(xs, xs)
        Z_sq = 1 - X**2 - Y**2
        self.inside = Z_sq >= 0
        Z = np.where(self.inside, np.sqrt(np.maximum(Z_sq, 0)), 0)
        self.pixels_cam = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        self.N = N

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_drag)

        self.draw()
        plt.show()

    def draw(self):
        self.ax.clear()
        R = self.rotation
        N = self.N

        pts_world = (R.T @ self.pixels_cam.T).T
        dots = pts_world @ self.site_vecs.T
        nearest_site = np.argmax(dots, axis=1)
        nearest_region = self.region_of_site[nearest_site].reshape(N, N)

        img = np.ones((N, N, 4))
        for i in range(len(self.regions)):
            mask = (nearest_region == i) & self.inside
            img[mask] = self.cmap(i)
        img[~self.inside] = [1, 1, 1, 0]

        self.ax.imshow(img, extent=[-1, 1, -1, 1], origin="lower",
                       interpolation="bilinear", zorder=1)

        for arc in self.edge_arcs:
            rot = (R @ arc.T).T
            front = rot[:, 2] >= 0
            if not front.any():
                continue
            segs_x, segs_y = [], []
            for pt, vis in zip(rot, front):
                if vis:
                    segs_x.append(pt[0]); segs_y.append(pt[1])
                else:
                    if segs_x:
                        self.ax.plot(segs_x, segs_y, "k-", linewidth=0.8,
                                     zorder=2, solid_capstyle="round")
                        segs_x, segs_y = [], []
            if segs_x:
                self.ax.plot(segs_x, segs_y, "k-", linewidth=0.8,
                             zorder=2, solid_capstyle="round")

        rot_v = (R @ self.vertex_vecs.T).T
        front_v = rot_v[rot_v[:, 2] >= 0]
        if len(front_v):
            self.ax.plot(front_v[:, 0], front_v[:, 1], "k.", markersize=5, zorder=3)

        rot_s = (R @ self.site_vecs.T).T
        for i, proj in enumerate(rot_s):
            if proj[2] < 0:
                continue
            self.ax.plot(proj[0], proj[1], "ro", markersize=7, zorder=4)
            self.ax.text(proj[0], proj[1] + 0.04, self.site_ids[i],
                         ha="center", va="bottom", fontsize=7,
                         fontweight="bold", zorder=5)

        self.ax.add_patch(Circle((0, 0), 1.0, fill=False, color="black",
                                 linewidth=1.5, zorder=6))
        self.ax.set_xlim(-1.15, 1.15)
        self.ax.set_ylim(-1.15, 1.15)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title("Spherical Voronoi — drag to rotate")
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.drag_start = (event.xdata, event.ydata)

    def _on_release(self, _):
        self.drag_start = None

    def _on_drag(self, event):
        if self.drag_start is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.drag_start[0]
        dy = event.ydata - self.drag_start[1]
        self.drag_start = (event.xdata, event.ydata)
        ax_angle = -dy * np.pi
        ay_angle = dx * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax_angle), -np.sin(ax_angle)],
                       [0, np.sin(ax_angle),  np.cos(ax_angle)]])
        Ry = np.array([[ np.cos(ay_angle), 0, np.sin(ay_angle)],
                       [0,                 1, 0],
                       [-np.sin(ay_angle), 0, np.cos(ay_angle)]])
        self.rotation = Ry @ Rx @ self.rotation
        self.draw()
