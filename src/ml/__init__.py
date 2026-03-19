"""ML subsystem for progressive learning across scans.

Components:
    - replay_buffer: Stores representative data from completed scans
    - face_prior: Neural network that predicts Gaussian initialization from FLAME params
    - taichi_physics: Physics-aware mesh refinement (Laplacian smoothing, collision, anatomical bounds)
    - integrate: Glue functions connecting the ML system to the pipeline
"""
