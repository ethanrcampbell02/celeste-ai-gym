# Celeste AI Gym

A reinforcement learning environment for training AI agents to play Celeste using OpenAI Gym interface.

## Overview

This repository contains the Python-based reinforcement learning training infrastructure for teaching AI agents to play Celeste. It provides a Gym environment that interfaces with the Celeste game through TCP communication with the companion Celeste AI mod.

## Features

- **Custom Gym Environment**: Full gymnasium-compatible environment for Celeste gameplay
- **Deep RL Support**: Pre-configured for PPO training with Stable-Baselines3
- **Real-time Training**: Direct interface with Celeste game through TCP socket communication
- **Advanced Wrappers**: Frame skipping, action simplification, and reward shaping
- **Comprehensive Monitoring**: TensorBoard integration and training visualization
- **Model Persistence**: Automatic checkpointing and model saving

## Architecture

```
Python RL Environment (celeste-ai-gym)
    ↕ TCP Socket Communication (port 5000)
Celeste Game + AI Mod (celeste-ai-mod)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Celeste game with the companion AI mod installed

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd celeste-ai-gym
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a New Model

```bash
python main.py
```

### Environment Testing

```bash
python debug_env.py
```

### Custom Training Configuration

Edit the configuration parameters in `main.py`:

```python
TOTAL_TIMESTEPS = 1_000_000  # Training duration
LEARNING_RATE = 3e-4         # PPO learning rate
N_STEPS = 2048               # Steps per update
BATCH_SIZE = 64              # Minibatch size
```

## Environment Details

### Observation Space

- **Type**: RGB images
- **Shape**: (180, 320, 3) 
- **Data Type**: uint8
- **Range**: [0, 255]

### Action Space

- **Type**: MultiBinary(7)
- **Actions**: [up, down, left, right, jump, dash, grab]
- **Binary**: Each action is 0 (not pressed) or 1 (pressed)

### Reward Function

The environment supports multiple reward modes:

- **Distance-based**: Rewards based on progress through the level
- **Room exploration**: Additional rewards for discovering new areas
- **Time penalties**: Encourages faster completion
- **Death penalties**: Negative rewards for player death

## File Structure

```
celeste-ai-gym/
├── CelesteEnv.py          # Main Gym environment
├── main.py                # Training script
├── wrappers.py            # Environment wrappers
├── utils.py               # Utility functions
├── debug_env.py           # Environment testing
├── get_held_out_states.py # Validation data collection
├── test_wrappers.py       # Wrapper testing
├── requirements.txt       # Python dependencies
├── models/                # Trained model checkpoints
├── holdouts/              # Validation datasets
└── README.md             # This file
```

## Environment Wrappers

### SimplifiedActionSpace
Reduces the action space complexity by mapping discrete actions to multi-binary combinations.

### SkipFrame
Implements frame skipping to reduce computational overhead and improve temporal consistency.

### ClipReward
Clips rewards to a specific range to improve training stability.

## Model Training

### Hyperparameters

The default configuration uses PPO with the following settings:

```python
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.005
MAX_GRAD_NORM = 0.5
```

### Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=models
```

## TCP Communication Protocol

The environment communicates with the Celeste mod using JSON messages over TCP:

### Reset Message
```json
{"type": "reset"}
```

### Action Message
```json
{
  "type": "action",
  "actions": [0, 0, 1, 0, 1, 0, 0]
}
```

### State Response
```json
{
  "type": "state",
  "image": "base64_encoded_screenshot",
  "gameState": {
    "position": [x, y],
    "velocity": [vx, vy],
    "room": "room_name",
    "distance": 123.45,
    "isDead": false,
    "completedLevel": false
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

**Connection Refused**: Ensure the Celeste AI mod is loaded and the game is running.

**CUDA Out of Memory**: Reduce batch size or use CPU training:
```python
device = "cpu"  # In main.py
```

**Environment Freezing**: Check TCP connection stability and mod compatibility.

## Performance Tips

- Use GPU acceleration for faster training
- Adjust frame skip rate based on your hardware
- Monitor memory usage during long training sessions
- Use tensorboard for real-time training monitoring

## Dependencies

Key dependencies include:

- `gymnasium`: RL environment interface
- `stable-baselines3`: Deep RL algorithms
- `torch`: Deep learning framework
- `opencv-python`: Image processing
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization

## License

[Specify your license here]

## Related Projects

- [Celeste AI Mod](../celeste-ai-mod): Companion C# mod for Celeste integration
- [ProgrammaticInput](../ProgrammaticInput): Input automation framework

## Citation

If you use this work in research, please cite:

```bibtex
@misc{celeste-ai-gym,
  title={Celeste AI Gym: Reinforcement Learning Environment for Celeste},
  author={[Your Name]},
  year={2026},
  url={[Repository URL]}
}
```