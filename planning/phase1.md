# Phase 1: Synthetic Space Data & Astrodynamics Setup

## 🎯 Phase Objective

In space domain awareness (SDA), acquiring labeled, real-world imagery of satellites with exact ground-truth positions is incredibly difficult. Therefore, the first step is to build a **physics-based synthetic data generator** grounded in real-world public datasets.

You will simulate the relative orbital motion of a target spacecraft using the **Clohessy-Wiltshire (CW) equations**. Instead of generating purely random noise for the background, you will use a **public Star Catalog (Hipparcos/Tycho-2)** to render accurate star fields. For the spacecraft itself, you will leverage assets from public aerospace datasets like **Stanford's SPEED+ (Spacecraft Pose Estimation Dataset)**. This data will serve as the realistic training and validation set for your deep learning detector and Kalman filter in later phases.

---

## 📚 1. Research & Theory

Before writing code, review these concepts to understand the math and data you are implementing.

### A. Relative Orbital Mechanics (Clohessy-Wiltshire / Hill's Frame)

When two spacecraft are close to each other in Low Earth Orbit (LEO), we use a relative coordinate system called the Hill frame (or Local Vertical Local Horizontal - LVLH).

* **Math:** The CW equations are linearized differential equations that model the relative 3D motion $[x, y, z]$ of a "Deputy" satellite relative to a "Chief" satellite.
* **Read:** [Wikipedia: Clohessy-Wiltshire equations](https://en.wikipedia.org/wiki/Clohessy-Wiltshire_equations?utm_source=gemini)

### B. Public Datasets for Space Perception

* **The Hipparcos Star Catalog:** A European Space Agency catalog of 118,218 stars with high-precision positions (Right Ascension and Declination) and visual magnitudes.
* **SPEED+ Dataset:** Stanford University's dataset for spacecraft pose estimation. It contains realistic renders and hardware-in-the-loop images of the Tango spacecraft.
* **Read / Explore:**
* [Skyfield / Astropy Python Libraries](https://rhodesmill.org/skyfield/stars.html?utm_source=gemini) (For loading star catalogs).
* [Stanford SLAB SPEED+ Dataset overview](https://www.google.com/search?q=https://slab.stanford.edu/projects/spacecraft-pose-estimation-dataset&utm_source=gemini).



---

## 🧪 2. Colab Lab Assignment: `00_orbital_dynamics_and_synthetic_data.ipynb`

**Goal:** Create a notebook that visualizes the math, pulls from public datasets, and proves your synthetic generation logic works before moving it to your Python package.

### Step 1: Propagate the Orbit

1. Define the orbital mean motion $n = \sqrt{\frac{\mu}{a^3}}$. For a standard 400km LEO orbit, $n \approx 0.00113 \text{ rad/s}$.
2. Implement the $6 \times 6$ CW State Transition Matrix in NumPy.
3. Initialize a state vector $X_0 = [x, y, z, v_x, v_y, v_z]^T$ (e.g., target is 500m ahead and 100m above).
4. Write a loop to propagate the state over 1000 seconds with a timestep `dt = 1.0` seconds.
5. **Output:** Use `matplotlib` to plot a 3D trajectory of the target relative to the camera.

### Step 2: Generate an Accurate Star Field (Using Astropy/Skyfield)

1. Use the `skyfield` or `astropy` Python libraries to download a subset of the **Hipparcos star catalog** (e.g., stars up to magnitude 6.0).
2. Define a simulated camera pointing direction in Right Ascension (RA) and Declination (Dec).
3. Project the 3D star coordinates into your 2D camera pixel grid based on your defined Field of View (FOV).
4. Scale the pixel intensity of each star based on its actual visual magnitude from the dataset. Apply `cv2.GaussianBlur` to simulate the Point Spread Function (PSF).

### Step 3: Inject Sensor Noise

1. Add Gaussian read noise: `noise = np.random.normal(mean=0, std=5, size=(1024, 1024))`
2. Add "Hot Pixels": Randomly select 50 pixels and set their intensity to 255.

### Step 4: Render the Target Spacecraft (Using SPEED+ Assets)

1. Download a sample image of the Tango spacecraft from the public **SPEED+ dataset** (or grab a public 3D CAD model `.obj` of a CubeSat).
2. Map the 3D position $[x, y, z]$ from Step 1 to a 2D pixel coordinate $[u, v]$ using a simple pinhole projection: $u = f_x \frac{x}{z} + c_x$.
3. Scale/Rotate the spacecraft asset based on distance and overlay it onto the Hipparcos starfield at $(u, v)$.
4. Generate a YOLO-format bounding box `[class_id, center_x, center_y, width, height]` (normalized from 0 to 1) for the target.

---

## 💻 3. Programming Implementation: `src/sim_engine.py`

Once your Colab notebook works, refactor the code into clean, object-oriented Python inside your `src/` directory.

Create a class `SpaceDataSimulator`:

```python
import numpy as np
import cv2
import pandas as pd
from skyfield.api import Star, load
from skyfield.data import hipparcos

class SpaceDataSimulator:
    def __init__(self, output_dir="data/synthetic", img_size=(1024, 1024), fov_deg=45.0):
        self.output_dir = output_dir
        self.img_size = img_size
        self.ts = load.timescale()
        
        # Load Hipparcos dataset for accurate starfield rendering
        with load.open(hipparcos.URL) as f:
            self.star_catalog = hipparcos.load_dataframe(f)
            
        # ... setup camera matrix K based on FOV ...

    def generate_cw_trajectory(self, state_0, time_steps, dt):
        """Propagates CW equations and returns a sequence of 6D states."""
        pass

    def render_starfield(self, ra, dec):
        """Projects Hipparcos stars into the 2D camera FOV with accurate magnitudes."""
        pass

    def render_frame(self, target_state, frame_id, target_image_path):
        """
        1. Generate realistic starfield using self.render_starfield().
        2. Project target_state (3D) to 2D pixels.
        3. Overlay target asset (e.g., from SPEED dataset).
        4. Apply Gaussian blur/hot pixels.
        5. Return YOLO bounding box and ECI state.
        """
        pass

    def generate_dataset(self, num_frames=500):
        """
        Loops through time_steps, calls render_frame, saves images, 
        and writes a ground_truth.csv with [frame_id, x, y, z, vx, vy, vz, bbox].
        """
        pass

```

---
## Resources

Here is a curated list of Medium articles, technical blogs, and documentation guides specifically tailored to **Phase 1: Synthetic Space Data & Astrodynamics**.

These resources contain Python code snippets and mathematical explanations that you can replicate directly in your `00_orbital_dynamics_and_synthetic_data.ipynb` Colab notebook.

---

### 1. Simulating Orbital Dynamics (The CW Equations)

*To implement the Clohessy-Wiltshire State Transition Matrix and propagate your target's 3D position over time.*

* **Article:** [Introduction to Orbital Mechanics with Python (Poliastro)](https://www.google.com/search?q=https://medium.com/%2540lucas.f.a.carvalho/introduction-to-orbital-mechanics-with-python-9b5f5dd83120&utm_source=gemini) by Lucas Carvalho *(Medium)*
* **Why it's relevant:** While you will write your own CW equations using NumPy, this article introduces the fundamentals of orbital mechanics in Python (state vectors, Keplerian elements).
* **What to replicate in Colab:** Understand how to define a 6D state vector (`[x, y, z, vx, vy, vz]`), loop through time steps `dt`, and use `matplotlib.pyplot.plot3D` to visualize the resulting trajectory.


* **Reference Repo:** [poliastro / poliastro](https://github.com/poliastro/poliastro?utm_source=gemini)
* **Why it's relevant:** Poliastro is the standard Python library for interactive astrodynamics. Looking at their `twobody` and relative motion modules will show you how professionals structure orbital propagation code.



### 2. Generating the Star Field (Hipparcos & Skyfield)

*To pull real star data (Right Ascension, Declination, and Magnitude) and project it onto a 2D image.*

* **Article:** [How to Create a Star Map in Python](https://www.google.com/search?q=https://towardsdatascience.com/how-to-create-a-star-map-in-python-850f55cb65ff&utm_source=gemini) by Andrew Drinkwater *(Towards Data Science)*
* **Why it's relevant:** This tutorial uses the `skyfield` package and the Hipparcos catalog to load star coordinates and plot them based on user location and field of view.
* **What to replicate in Colab:** Use the code provided in the article to download the Hipparcos catalog, filter stars by visual magnitude (e.g., $M \le 6.0$), and convert their spherical coordinates (RA/Dec) into a 2D grid. Instead of rendering them as a Matplotlib scatter plot (like the article does), you will map them to a NumPy pixel array.


* **Documentation Code Snippet:** [Skyfield: Drawing a Star Chart](https://rhodesmill.org/skyfield/stars.html?utm_source=gemini)
* **Why it's relevant:** The official Skyfield docs provide copy-pasteable Python code to load the Hipparcos dataframe and calculate stereographic projections.



### 3. Space Camera Geometry & Projection

*To project your 3D target spacecraft into the 2D image plane using a pinhole camera model.*

* **Article:** [Camera Calibration and 3D Reconstruction](https://www.google.com/search?q=https://medium.com/towards-data-science/camera-calibration-and-3d-reconstruction-74df09fb13b1&utm_source=gemini) *(Towards Data Science)*
* **Why it's relevant:** It explains the Intrinsic Camera Matrix ($K$), focal length ($f_x, f_y$), and principal point ($c_x, c_y$).
* **What to replicate in Colab:** Implement the pinhole math in NumPy: $u = f_x \frac{x}{z} + c_x$ and $v = f_y \frac{y}{z} + c_y$. Create a function `project_3d_to_2d(target_x, target_y, target_z, K_matrix)` that outputs the pixel coordinate where your spacecraft should be drawn.



### 4. Simulating Sensor Noise & Artifacts

*To make your synthetic images look like they were taken by a radiation-bombarded space sensor.*

* **Article:** [Image Noise Models in Python](https://www.google.com/search?q=https://towardsdatascience.com/image-noise-models-in-python-86f781df0342&utm_source=gemini) by Sajjad Amjad *(Towards Data Science)*
* **Why it's relevant:** Covers exactly how to inject different types of mathematical noise into image matrices using NumPy and OpenCV.
* **What to replicate in Colab:** Implement **Gaussian Noise** (to simulate electronic read noise from the sensor) and **Salt-and-Pepper Noise** (to simulate cosmic-ray strikes / hot pixels).


* **Article:** [Image Processing with Python: Blurring and Smoothing](https://www.google.com/search?q=https://towardsdatascience.com/image-processing-with-python-blurring-and-smoothing-51b682ce2eb0&utm_source=gemini) *(Towards Data Science)*
* **Why it's relevant:** Explains Gaussian Blurs. You will use `cv2.GaussianBlur` on your star pixels to simulate the optical Point Spread Function (PSF) so stars look like glowing orbs rather than sharp 1-pixel white dots.



### 5. Managing the Dataset (Bounding Boxes for YOLO)

*To save the images with the correct labels so Phase 3 (Deep Learning) can use them.*

* **Article:** [How to create a custom dataset for YOLOv8](https://www.google.com/search?q=https://medium.com/%2540m.ali.faisal.786/how-to-create-a-custom-dataset-for-yolov8-1ce4998782a2&utm_source=gemini) by Muhammad Ali Faisal *(Medium)*
* **Why it's relevant:** Explains the `.txt` format YOLO requires for bounding boxes: `[class_id center_x center_y width height]` (normalized from 0 to 1).
* **What to replicate in Colab:** After you paste your spacecraft asset onto the starfield, calculate the bounding box. Normalize the coordinates by dividing by the image width/height (e.g., $1024 \times 1024$) and save them to a `.txt` file corresponding to each generated image.


---
## ✅ Phase 1 Definition of Done (DoD) Checklist

* [ ] I understand the basics of the Clohessy-Wiltshire relative motion model.
* [ ] Read the documentation for extracting star coordinates via `Skyfield`/`Astropy` and downloaded a sample of the SPEED+ dataset or an open-source CubeSat CAD model.
* [ ] Colab notebook `00_orbital_dynamics_and_synthetic_data.ipynb` is created and successfully plots a 3D orbit.
* [ ] Colab notebook successfully generates a realistic space image combining the Hipparcos catalog, sensor noise, and a target spacecraft overlay.
* [ ] Refactored the code into `src/sim_engine.py`.
* [ ] Ran the simulator to generate a dataset of 500+ sequential images.
* [ ] Generated `ground_truth.csv` mapping every image to its true 3D position, velocity, and 2D bounding box.