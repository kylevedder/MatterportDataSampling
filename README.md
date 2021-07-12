# Matterport Bounding Box Dataset Generator

## Setup

Setup a `data/` folder with a path to the `matterport` dataset, e.g.

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

To build a Docker container, use the provided `Dockerfile`. Following the above setup, mount the project directory as `/project` and the `/data` directory as follows:

```
nvidia-docker run -v `pwd`:/project -v /data:/data --rm -it habitat_sampling
```
