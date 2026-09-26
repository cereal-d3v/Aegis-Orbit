# 🗺️ Project Roadmap: Aegis-Orbit

This project is developed using a research-to-production methodology modeled after aerospace flight software development. Each phase begins with literature review and mathematical derivation, transitions to interactive Jupyter/Colab notebooks for rapid prototyping ("Labs"), and concludes with modular, tested production code ("Programming").

## 📌 Phase 1: Synthetic Space Data & Astrodynamics Setup

*True Anomaly requires algorithms that work in space. Since we don't have a satellite in orbit, we must simulate realistic spacecraft targets, star fields, and orbital mechanics.*

* **Research:**
* Read up on the Clohessy-Wiltshire (CW) equations for relative orbital motion (Hill's frame).
* Study space sensor noise models: Poisson shot noise, read noise, and cosmic-ray "hot pixels."


* **Labs (`/notebooks`):**
* `[ ] 00_orbital_dynamics_and_synthetic_data.ipynb`: Plot 3D relative orbits using the CW equations. Generate synthetic 2D frames with black backgrounds, random star point-spread functions (PSFs), and rendered spacecraft bounding boxes.


* **Programming (`/src`):**
* `[ ] src/sim_engine.py`: A modular synthetic data generator to create labeled datasets for our ML models (images + YOLO format bounding boxes + ECI state vectors).



## 📌 Phase 2: Classical Space Image Processing & Geometry

*Before feeding data to neural networks, we must clean the raw sensor feed and establish rigorous geometric coordinate chains (Pixels $\to$ ECI).*

* **Research:**
* Study OpenCV mathematical morphology (Top-Hat filtering) for star background suppression.
* Read *Camera Calibration and 3D Reconstruction* (OpenCV Docs) and Quaternions for spacecraft attitude.


* **Labs (`/notebooks`):**
* `[ ] 01_astrometric_coordinate_chains.ipynb`: Step-by-step conversion of 2D pixels $(u,v)$ to 3D unit line-of-sight rays in the inertial frame.
* `[ ] 02_space_image_filtering_centroiding.ipynb`: Visualizing hot-pixel removal and sub-pixel centroiding of star fields.


* **Programming (`/src`):**
* `[ ] src/coordinates.py`: Pinhole camera matrix inversion and quaternion rotation logic.
* `[ ] src/image_processing.py`: Reusable filters for median hot-pixel cleaning and connected component analysis.



## 📌 Phase 3: Deep Edge Perception (Detection & Re-ID)

*Using machine learning to discriminate between the actual target spacecraft and space debris/decoys.*

* **Research:**
* Study YOLOv8-nano architecture for edge detection.
* Research Siamese Networks and Triplet Loss for visual metric learning (appearance embeddings).


* **Labs (`/notebooks`):**
* `[ ] 03_yolo_training_and_reid.ipynb`: Train YOLOv8 on the synthetic dataset generated in Phase 1. Implement a lightweight MobileNet feature extractor to output 128-dimensional appearance vectors for detected objects.


* **Programming (`/src`):**
* `[ ] src/detector.py`: PyTorch inference wrapper for the trained YOLO model.
* `[ ] src/reid.py`: Appearance embedding generator for track matching.



## 📌 Phase 4: Non-Linear Estimation & Multi-Object Tracking (MOT)

*The mathematical core of the project. Tracking targets across time using only 2D angular measurements.*

* **Research:**
* Study the Extended Kalman Filter (EKF), specifically calculating the Jacobian matrix ($H$) for angles-only bearing measurements.
* Read about the Hungarian Algorithm (Munkres) and DeepSORT for data association.


* **Labs (`/notebooks`):**
* `[ ] 04_angles_only_ekf_derivation.ipynb`: NumPy implementation of the EKF. Plot actual vs. estimated trajectories and verify 3-sigma covariance bounds.
* `[ ] 05_hungarian_multi_target_tracker.ipynb`: Implement Mahalanobis gating and the bipartite matching cost matrix. Simulate multiple targets crossing paths.


* **Programming (`/src`):**
* `[ ] src/ekf.py`: The `AnglesOnlyEKF` class with analytical Jacobians.
* `[ ] src/tracker.py`: The `MultiTargetTracker` class managing `TENTATIVE`, `CONFIRMED`, and `COASTED` track lifecycles.



## 📌 Phase 5: Edge Optimization & C++ Flight Software

*Space-qualified processors have strict SWaP (Size, Weight, Power) limits. We must quantize our models and write low-latency C++ flight code.*

* **Research:**
* Study ONNX Runtime C++ API documentation.
* Research INT8 dynamic quantization from floating-point 32 (FP32).
* Familiarize with the `Eigen` C++ library for matrix math.


* **Labs (`/notebooks`):**
* `[ ] 06_model_quantization_int8_export.ipynb`: Export PyTorch models to ONNX. Apply INT8 quantization and benchmark file size and inference speed reduction.


* **Programming (`/flight_cpp`):**
* `[ ] flight_cpp/CMakeLists.txt`: Build configuration linking ONNX Runtime and Eigen.
* `[ ] flight_cpp/src/ekf.cpp`: Port the Python EKF logic to high-performance C++.
* `[ ] flight_cpp/src/inference_engine.cpp`: C++ wrapper to run the quantized ONNX detector.



## 📌 Phase 6: Closed-Loop Validation & CI/CD

*Tying it all together into an autonomous pipeline and proving it works for recruiters and engineers.*

* **Research:**
* Software-in-the-Loop (SITL) testing paradigms for robotics.
* GitHub Actions for automated testing.


* **Labs (`/notebooks`):**
* `[ ] 07_end_to_end_closed_loop_simulation.ipynb`: The capstone notebook. Feed raw synthetic images in, run classical CV, YOLO detection, Hungarian tracking, and EKF estimation, outputting the final 3D orbital state determination.


* **Programming:**
* `[ ] .github/workflows/ci.yml`: Setup continuous integration to run `pytest` on the Python core and `cmake build` on the C++ flight code.
* `[ ] README.md`: Final polish with GIFs of the tracking pipeline and benchmark graphs.