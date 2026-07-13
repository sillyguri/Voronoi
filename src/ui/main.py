import random
import pandas as pd
from pathlib import Path

from core.algorithms import VoronoiSetting
from core.geometry_2d import Point2D
from ui.show_2D import draw_voronoi_diagram


def main() -> None:
    df = pd.read_csv(Path(__file__).parent / "uscities.csv")
    df = df[(df['lat'] > 22) & (df['lat'] < 50) & (df['lng'] > -130) & (df['lng'] < -67)]
    df = df[df['population'] > 10000]
    IDList = list(df['city_ascii'])
    pointList = [
        Point2D(row['lng'] * 100 + random.uniform(-1e-9, 1e-9), row['lat'] * 100 + random.uniform(-1e-9, 1e-9))
        for _, row in df.iterrows()
    ]
    boundingBox = [[-13000, 2200], [-6700, 5000]]
    queryList = []
    for _ in range(0):
        queryList.append(Point2D(random.uniform(-13000, -6700), random.uniform(2200, 5000)))
    setting = VoronoiSetting(IDList, pointList, boundingBox, queryList)
    draw_voronoi_diagram(IDList, pointList, boundingBox, setting.VoronoiRegions, setting.VoronoiVertices, queryList)


if __name__ == '__main__':
    main()