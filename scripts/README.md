# Scripts in Instinct Onboard

This is the directory storing all entry-points of the Instinct Onboard repository.

Please DO NOT import any of the python files in this directory from other modules.

**The scripts in this directory are meant to be run as standalone programs.**

## Available Scripts

### g1_parkour.py
Autonomous parkour behaviors using depth camera perception.
- **Agent**: ParkourAgent + ParkourStandAgent
- **Features**: Depth perception, terrain navigation, velocity control
- **Usage**: `python g1_parkour.py --logdir /path/to/parkour/model --standdir /path/to/stand/model`

### g1_perceptive_track.py
Perceptive motion tracking with depth camera and NPZ motion files.
- **Agent**: PerceptiveTrackerAgent + WalkAgent
- **Features**: Depth perception, motion reference tracking, walk mode
- **Usage**: `python g1_perceptive_track.py --logdir /path/to/tracking/model --motion_dir /path/to/motions`

### g1_track.py
Basic motion tracking from NPZ files (no depth perception).
- **Agent**: TrackerAgent
- **Features**: Motion reference from files, simple tracking
- **Usage**: `python g1_track.py --logdir /path/to/tracking/model --motion_dir /path/to/motions`

### g1_interaction.py
Interaction task deployed from the sitting checkpoint truth.
- **Agent**: InteractionAgent + WalkAgent
- **Features**: Raw observation normalization, depth encoder, actor inference, motion reference tracking
- **Usage**: `python g1_interaction.py --logdir /path/to/interaction/model --walk_logdir /path/to/walk/model --motion_dir /path/to/motions`

## Common Arguments

| Argument | Description |
|----------|-------------|
| `--logdir` | Directory containing exported ONNX model and configs |
| `--nodryrun` | Disable dry run mode (enables actual robot control) |
| `--startup_step_size` | Cold start step size (default: 0.2) |
| `--kpkd_factor` | KP/KD gain multiplier for cold start (default: 2.0) |
| `--debug` | Enable debugpy for remote debugging |

## Visualization Options

| Argument | Description |
|----------|-------------|
| `--depth_vis` | Publish depth images to `/debug/depth_image` |
| `--pointcloud_vis` | Publish point clouds to `/debug/pointcloud` |
| `--motion_vis` | Publish motion sequence as JointState for RViz |
