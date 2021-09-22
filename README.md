# Matterport Bounding Box Dataset Generator

This script generates supervised datasets of the specified objects in [Matterport3D](https://niessner.github.io/Matterport/)'s house meshes, using [AIHabitat](https://aihabitat.org/) to render the views.

## Setup

Download Matterport3D. Inside the repo, use symlinks inside the `data/` folder to link to the `matterport` dataset, e.g.

```
data
├── datasets
│   └── pointnav
│       └── mp3d
│           └── v1 -> /data/matterport/pointnav/v1
└── scene_datasets
    └── mp3d -> /data/matterport/mp3d
```

## Build/Run

To build the Docker container, use the provided `Dockerfile`. Following the above setup, mount the project directory as `/project` and the `/data` directory as follows:

```
nvidia-docker run -v `pwd`:/project -v /data:/data --rm -it habitat_sampling
```

Then run `python main.py` with optional config flags.

## Citation

This code was used to generate the _Matterport-Chair_ dataset in our work _Sparse PointPillars: Maintaining and Exploiting Input Sparsity to Improve Runtime on Embedded Systems_. If you use this code for your research, please [cite Sparse PointPillars](https://github.com/kylevedder/SparsePointPillars#citation).
