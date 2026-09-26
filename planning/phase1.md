# Phase 1: Synthetic Space Data & Astrodynamics Setup

## 🎯 Phase Objective

In space domain awareness (SDA), acquiring labeled, real-world imagery of satellites with exact ground-truth positions is incredibly difficult due to classification and operational constraints. Therefore, the first step of this project is to build a **physics-based synthetic data generator**.

You will simulate the relative orbital motion of a target spacecraft using the **Clohessy-Wiltshire (CW) equations** and generate synthetic camera frames with realistic space sensor noise (star fields, cosmic rays, shot noise). This data will serve as the training and validation set for your deep learning detector and Kalman filter in later phases.

---

## 📚 1. Research & Theory

Before writing code, review these concepts to understand the math you are implementing.

### A. Relative Orbital Mechanics (Clohessy-Wiltshire / Hill's Frame)

When two spacecraft are close to each other in Low Earth Orbit (LEO), we don't calculate their absolute positions relative to the center of the Earth. Instead, we use a relative coordinate system called the Hill frame (or Local Vertical Local Horizontal - LVLH).

* **Math:** The CW equations are linearized differential equations that model the relative 3D motion $[x, y, z]$ of a "Deputy" satellite relative to a "Chief" satellite.
* **Read:** [Wikipedia: Clohessy-Wiltshire equations](https://en.wikipedia.org/wiki/Clohessy-Wiltshire_equations?utm_source=gemini)
* **Key Insight:** You will use the discrete-time State Transition Matrix ($\Phi$ or $F$) to propagate the target's position over time ($t$).

### B. Space Sensor Noise Models

Space cameras (Star Trackers and NavCams) do not look like standard phone cameras. They operate in extreme lighting conditions.

* **Point Spread Function (PSF):** Stars aren't single pixels; they bleed into adjacent pixels following a 2D Gaussian distribution.
* **Hot Pixels / Cosmic Rays:** High-energy particles hit the camera sensor, causing random pixels to permanently max out at 255 (salt-and-pepper noise).
* **Read:** [Image Noise Models (Towards Data Science)](https://www.google.com/search?q=https://towardsdatascience.com/image-noise-models-in-python-86f781df0342&utm_source=gemini) - Focus on Gaussian and Poisson noise generation using NumPy.

---

## 🧪 2. Colab Lab Assignment: `00_orbital_dynamics_and_synthetic_data.ipynb`

**Goal:** Create a notebook that visualizes the math and proves your synthetic generation logic works before moving it to your Python package.

### Step 1: Propagate the Orbit

1. Define the orbital mean motion $n = \sqrt{\frac{\mu}{a^3}}$. For a standard 400km LEO orbit, $n \approx 0.00113 \text{ rad/s}$.
2. Implement the $6 \times 6$ CW State Transition Matrix in NumPy.
3. Initialize a state vector $X_0 = [x, y, z, v_x, v_y, v_z]^T$ (e.g., target is 500m ahead and 100m above).
4. Write a loop to propagate the state over 1000 seconds with a timestep `dt = 1.0` seconds.
5. **Output:** Use `matplotlib` to plot a 3D trajectory of the target relative to the camera.

### Step 2: Generate the Star Field Background

1. Create a blank $1024 \times 1024$ NumPy array (black background).
2. Randomly select 100-200 $(x,y)$ coordinates to represent stars.
3. Assign them random intensities (brightness).
4. Apply `cv2.GaussianBlur` to simulate the Point Spread Function (PSF) of the optics.

### Step 3: Inject Sensor Noise

1. Add Gaussian read noise: `noise = np.random.normal(mean=0, std=5, size=(1024, 1024))`
2. Add "Hot Pixels": Randomly select 50 pixels and set their intensity to 255.

### Step 4: Render the Target Spacecraft

1. Take a small image of a satellite (e.g., a simple white rectangle or a transparent PNG of a CubeSat).
2. Map the 3D position $[x, y, z]$ from Step 1 to a 2D pixel coordinate $[u, v]$ using a simple pinhole projection: $u = f_x \frac{x}{z} + c_x$.
3. Overlay the satellite image onto the starfield at $(u, v)$.
4. Generate a YOLO-format bounding box `[class_id, center_x, center_y, width, height]` (normalized from 0 to 1) for the target.

---

## 💻 3. Programming Implementation: `src/sim_engine.py`

Once your Colab notebook works, refactor the code into clean, object-oriented Python inside your `src/` directory.

Create a class `SpaceDataSimulator`:

```python
import numpy as np
import cv2
import pandas as pd
import os

class SpaceDataSimulator:
    def __init__(self, output_dir="data/synthetic", img_size=(1024, 1024), fov_deg=45.0):
        self.output_dir = output_dir
        self.img_size = img_size
        # ... setup camera matrix K based on FOV ...

    def generate_cw_trajectory(self, state_0, time_steps, dt):
        """Propagates CW equations and returns a sequence of 6D states."""
        pass

    def render_frame(self, target_state, frame_id):
        """
        1. Generate noisy starfield background.
        2. Project target_state (3D) to 2D pixels.
        3. Overlay target and apply Gaussian blur/hot pixels.
        4. Save image to disk.
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

## ✅ Phase 1 Definition of Done (DoD) Checklist

* [ ] I understand the basics of the Clohessy-Wiltshire relative motion model.
* [ ] Colab notebook `00_orbital_dynamics_and_synthetic_data.ipynb` is created and successfully plots a 3D orbit.
* [ ] Colab notebook successfully generates a realistic space image with stars, noise, and a target overlay.
* [ ] Refactored the code into `src/sim_engine.py`.
* [ ] Ran the simulator to generate a dataset of 500+ sequential images.
* [ ] Generated `ground_truth.csv` mapping every image to its true 3D position, velocity, and 2D bounding box.

---

Let me know when you have saved this to your `planning/` folder and are ready for **Phase 2: Classical Space Image Processing & Geometry**!