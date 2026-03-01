# NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos

**Authors:** [Hongyu Li*](https://lhy.xyz/), [Lingfeng Sun*](https://lingfeng.moe/), [Yafei Hu](https://jeffreyyh.github.io/), [Duy Ta](https://www.linkedin.com/in/duynguyen-ta), [Jennifer Barry](https://www.linkedin.com/in/jennifer-barry-742a0799/), [George Konidaris](https://cs.brown.edu/people/gdk/), [Jiahui Fu](https://jiahui-fu.github.io/)  
**Affiliations:** Robotics and AI Institute, Brown University  
**\*Equal contribution**

<a href="https://arxiv.org/abs/2510.08568"><img src='https://img.shields.io/badge/arXiv-NovaFlow-red' alt='Paper PDF'></a>
<a href='https://novaflow.lhy.xyz'><img src='https://img.shields.io/badge/Project_Page-NovaFlow-green' alt='Project Page'></a>
<a href='https://novaflow.lhy.xyz'><img src='https://img.shields.io/badge/Interactive_Viewer-NovaFlow-blue' alt='Interactive Viewer'></a>

![NovaFlow Overview](static/overview.gif)

**NovaFlow** enables robots to execute novel manipulation tasks in a zero-shot manner without any demonstrations or embodiment-specific training. Given a natural language task description, NovaFlow autonomously synthesizes a video using state-of-the-art video generation models and distills it into 3D actionable object flow. This flow is then converted into precise robot actions through grasp proposals and trajectory optimization, enabling seamless transfer across different robotic platforms.

## ✨ Key Features

- **Zero-Shot Manipulation**: Execute novel tasks without demonstrations or training
- **Multi-Embodiment Transfer**: Naturally transfers across different robots (Franka arm, Spot quadruped)
- **Object Agnostic**: Handles rigid, articulated, and deformable objects
- **Language-to-Action**: Converts natural language task descriptions into precise robot trajectories
- **Actionable Flow**: Distills generated videos into 3D object motion plans
- **Robust Execution**: Grasp proposal + trajectory optimization for reliable manipulation

## 📋 Abstract

Enabling robots to execute novel manipulation tasks zero-shot is a central goal in robotics. Most existing methods assume in-distribution tasks or rely on fine-tuning with embodiment-matched data, limiting transfer across platforms. We present **NovaFlow**, an autonomous manipulation framework that converts a task description into an actionable plan for a target robot without any demonstrations. Given a task description, NovaFlow synthesizes a video using a video generation model and distills it into 3D actionable object flow using off-the-shelf perception modules. From the object flow, it computes relative poses for rigid objects and realizes them as robot actions via grasp proposals and trajectory optimization. For deformable objects, this flow serves as a tracking objective for model-based planning with a particle-based dynamics model. By decoupling task understanding from low-level control, NovaFlow naturally transfers across embodiments. We validate on rigid, articulated, and deformable object manipulation tasks using a table-top Franka arm and a Spot quadrupedal mobile robot, and achieve effective zero-shot execution without demonstrations or embodiment-specific training.

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Multi-GPU setup (H100/A100 recommended) for Wan2.1 video generation pipeline. For any GPU under A100/H100, the **Veo model is recommended** (requires only a single gaming GPU).
- **Software**: Python 3.8+, Docker (recommended, optional)
- **API Keys**: `GOOGLE_API_KEY` (required if using Veo model)
- **Robots**: Franka Panda arm or Boston Dynamics Spot (for physical execution)

### Installation

1. **Clone the repository**:
   ```bash
   git clone --recursive https://github.com/holi-rai/novaflow-private.git
   cd novaflow-private
   ```

2. **Setup environment**:
   ```bash
   # Option 1: Using Docker (Recommended)
   docker pull us-docker.pkg.dev/engineering-380817/bdai/holi_gizmo:main

   # Option 2: Local Docker build
   cd server/docker
   docker build -t novaflow .
   ```

3. **Download model weights**:
   ```bash
   cd server
   ./download_weights.sh
   ```

4. **Start the server**:
   To use prompt extention, you should set `GOOGLE_API_KEY` to your Google API key.

   <details>
   <summary><b>Option 1: Using Wan (Default)</b></summary>

   ```bash
   cd server
   ./start_ray_server.sh
   ```
   </details>

   <details>
   <summary><b>Option 2: Using Veo (Recommended for GPUs < A100/H100)</b></summary>

   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   cd server
   ./start_ray_server.sh --model veo
   ```
   </details>

5. **Run your first job**:

   <details>
   <summary><b>Option 1: Using Wan (Default)</b></summary>

   ```bash
   cd client
   python submit_jobs.py --num-jobs 1 --base-seed 42
   ```
   </details>

   <details>
   <summary><b>Option 2: Using Veo (Recommended for GPUs < A100/H100)</b></summary>

   ```bash
   cd client
   python submit_jobs.py --num-jobs 1 --base-seed 42 --use-veo
   ```
   </details>

## 📖 Pipeline Overview

NovaFlow operates through two main pipelines that convert language instructions into robot actions:

### 🎬 Flow Generator Pipeline
Converts task descriptions into 3D actionable object flow:

1. **Video Generation**: Synthesizes plausible object motion videos using state-of-the-art video models (WAN2.1)
2. **3D Lifting**: Converts 2D video to 3D using monocular depth estimation
3. **Depth Calibration**: Calibrates estimated depth against initial observations
4. **Point Tracking**: Tracks dense per-point motion using 3D point tracking (Tapip3D)
5. **Object Grounding**: Extracts object-centric 3D flow via segmentation (Grounded SAM 2)

### 🤖 Flow Executor Pipeline
Converts 3D flow into precise robot trajectories:

1. **Grasp Proposal**: Determines initial end-effector poses from grasp candidates
2. **Trajectory Planning**: Plans robot trajectories based on actionable flow with cost/constraint optimization
3. **Motion Execution**: Tracks planned trajectories on physical robots (Franka/Spot)

### 🎯 Object Types Supported
- **Rigid Objects**: Cup placement, block insertion, mug hanging
- **Articulated Objects**: Drawer opening, lid lifting
- **Deformable Objects**: Rope straightening, plant watering

### 📋 Pipeline Modules

- [x] **Video Generation**: WAN2.1 model integration for task video synthesis
- [x] **Depth Estimation**: Monocular depth lifting for 3D reconstruction
- [x] **Depth Calibration**: Calibration against initial depth observations
- [x] **Point Tracking**: Tapip3D integration for dense 3D motion tracking
- [x] **Object Grounding**: Grounded SAM 2 for object-centric flow extraction
- [ ] **Flow Visualization**: 3D motion flow visualization and analysis
- [ ] **Grasp Planning**: Candidate grasp pose generation and selection
- [ ] **Trajectory Optimization**: Motion planning with constraints and costs
- [ ] **Execution**: Real-time trajectory tracking on physical robots

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

NovaFlow builds upon several outstanding research projects and open-source implementations:

- **[Wan2.1](https://github.com/Wan-Video/Wan2.1)**: Video generation models
- **[TAPIP3D](https://github.com/zbw001/TAPIP3D)**: 3D point tracking
- **[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)**: Object segmentation and tracking
- **[Mega-SAM](https://github.com/mega-sam/mega-sam)**: Lift 2D video to 3D using depth estimation
- **[GraspGen](https://github.com/NVlabs/GraspGen)**: Grasp planning and generation
- **[Ray](https://github.com/ray-project/ray)**: Distributed computing framework

## 📚 Citations

If you find NovaFlow useful in your research, please cite our paper:

```bibtex
@article{li2025novaflow,
  title={Novaflow: Zero-shot manipulation via actionable flow from generated videos},
  author={Li, Hongyu and Sun, Lingfeng and Hu, Yafei and Ta, Duy and Barry, Jennifer and Konidaris, George and Fu, Jiahui},
  journal={arXiv preprint arXiv:2510.08568},
  year={2025}
}
```