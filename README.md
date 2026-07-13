# Voronoi Diagram Analysis Tool

A comprehensive computational geometry tool for generating, visualizing, and analyzing Voronoi diagrams on both 2D Cartesian and 3D spherical surfaces.

## Project Overview

This tool provides an interactive environment for spatial partitioning. It supports user-driven coordinate entry and automated generation of geometric structures with optimized algorithmic performance.

## Key Features

* **Interactive Coordinate Grid**: Users can import coordinates or click points directly on the grid to serve as sites for the Voronoi diagram.
* **Efficient Voronoi Generation**: Diagrams are represented with distinct colors, computed using an expected-time $O(n \\log n)$ algorithm.
* **Data Export**: The tool exports files containing site points, boundary segments, and region sizes (including infinite regions), sorted by region size.
* **Trapezoidal Decomposition**: The system constructs a trapezoidal decomposition of the Voronoi diagram in $O(n \\log n)$ expected time.
* **Point Location**: An additional set of $m$ query points can be displayed, with a mapping to their nearest Voronoi sites generated in $O(m \\log n)$ time.

## Implementation Details

### 2D Cartesian Case

* **Coordinate System**: Uses standard Cartesian coordinates for all entries and calculations.
* **Navigation**: Features translation and a "Reset Zoom" function that centers the view on a bounding box containing all points.
* **Core Algorithm**: Voronoi diagrams are constructed via Delaunay Triangulation using the **Bowyer-Watson algorithm** with random-order insertion to maintain $O(n \\log n)$ efficiency.
* **Area Calculation**: Region areas are calculated using **Shoelace’s formula**.

### 3D Spherical Case

* **Input Formats**: Supports spherical $(\\phi, \\theta)$ or Latitude/Longitude coordinates.
* **Visualization**: The sphere is rendered via an **orthographic projection**, with dynamic zooming based on point distribution.
* **Spherical Triangulation**: Employs **stereographic projection** to map spherical points to a 2D plane for Delaunay Triangulation before returning the results to the sphere.
* **Spherical Geometry**: Region areas are computed using **Girard's Theorem**.

## Installation & Execution
### Installation

Follow these steps to set up the project locally on your machine:

1. Clone the Repository

Open your terminal or command prompt and run:
```bash
git clone https://github.com
cd your-repo-name
```

2. Create a Virtual Environment
* **Windows:**
```cmd
python -m venv Voronoi
"Voronoi\\Scripts\\activate.bat"
```

* **macOS / Linux:**
```bash
python3 -m venv Voronoi
source Voronoi/bin/activate
```

3. Install the Project
```bash
pip install -e .
```

### Running the Application
```bash
python -m ui.main
```
To create 2D or 3D Voronoi diagrams with your own input files:
```bash
python -m ui.voronoi_user_input
```

### Project Structure
* `src/core/` - Contains the algorithmic logic and calculations.
* `src/ui/` - Contains the user interface and main display window.
* `tests/` - Contains sample input files for creating 2D and 3D Voronoi diagrams.
* `pyproject.toml` - Handles configuration and project dependencies.


## Authors

* Ryan Dong (ryd@mit.edu)
* Arnav Goel (arnavmg@mit.edu)
* Aprameya Tripathy (aprameya@mit.edu)