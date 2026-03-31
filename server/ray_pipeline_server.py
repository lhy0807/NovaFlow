"""
Unified Ray Pipeline Server for WAN2.1, Tapip3D, and SAM processing
Supports multiple pipeline modes and distributed processing across GPUs.

Configuration:

WAN Models:
- FLF2V: ./wan2.1/Wan2.1-FLF2V-14B-720P (relative to server dir)
- I2V: ./wan2.1/Wan2.1-I2V-14B-720P (relative to server dir)
- Size: 1280*720
- Frame number: 21 (unless downsampled from Veo)
- FPS: From config (typically 16)

Tapip3D:
- Checkpoint: ./tapip3d/checkpoints/tapip3d_final.pth (relative to server dir)
- Resolution factor: 2
- Model image_size: [384, 512]
- Inference resolution: Calculated dynamically using model.image_size * sqrt(resolution_factor)
- Resolution setting: Uses model.set_image_size()

SAM:
- Checkpoint: ./grounded_sam_2/checkpoints/sam2.1_hiera_large.pt (relative to server dir)
- Config: ./grounded_sam_2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml (relative to server dir)
- Grounding DINO: IDEA-Research/grounding-dino-tiny
- Box threshold: 0.25
- Text threshold: 0.3
"""

import sys
import os
import asyncio
import threading
import queue
import time
import pickle
import base64
import tempfile
import argparse
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from rich.logging import RichHandler
import cv2
from google import genai
from google.genai import types
from google.genai.types import GenerateVideosConfig
# Lazy import to avoid hard dependency at module import time
# from google.cloud import storage # Removed as no longer needed for Veo Gemini API
import uuid
import shutil
import torchvision.transforms as T

# Configure rich logging
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)

# Add necessary paths for imports
# Note: insert(0, ...) puts items at front, so last inserted = first in sys.path
# tapip3d must be inserted LAST so it's searched FIRST (it has 'training' package needed by its modules)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'SpaTrackerV2'))
sys.path.insert(0, os.path.join(current_dir, 'grounded_sam_2'))
sys.path.insert(0, os.path.join(current_dir, 'wan2.1'))
sys.path.insert(0, os.path.join(current_dir, 'tapip3d'))  # Last = first in sys.path

# Set environment variables for proper module loading
os.environ['PYTHONPATH'] = current_dir + ':' + os.environ.get('PYTHONPATH', '')

try:
    import ray
except ImportError:
    logger.error("Ray not available. Please install: pip install ray")
    sys.exit(1)

try:
    import zmq
    import zmq.asyncio
except ImportError:
    logger.error("ZMQ not available. Please install: pip install pyzmq")
    sys.exit(1)

try:
    import torch
    import numpy as np
    from PIL import Image
    import cv2
except ImportError as e:
    logger.error(f"Required packages not available: {e}")
    sys.exit(1)

try:
    import threading
    import time
except ImportError as e:
    logger.error(f"Threading packages not available: {e}")
    sys.exit(1)

# Try to import pipeline components
TAPIP3D_AVAILABLE = False
SAM_AVAILABLE = False
WAN_AVAILABLE = False

try:
    # For Tapip3D, we'll load models directly in the Ray workers
    TAPIP3D_AVAILABLE = True
    logger.info("Tapip3D will be loaded directly in Ray workers")
except ImportError as e:
    logger.warning(f"Tapip3D not available: {e}")

try:
    # For SAM, we'll load models directly in the Ray workers
    SAM_AVAILABLE = True
    logger.info("SAM will be loaded directly in Ray workers")
except ImportError as e:
    logger.warning(f"SAM not available: {e}")

try:
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    WAN_AVAILABLE = True
    logger.info("WAN imported successfully")
except ImportError as e:
    logger.warning(f"WAN not available: {e}")

class PipelineMode(Enum):
    """Pipeline execution modes"""
    FULL = "full"                    # WAN -> Tapip3D -> SAM
    WAN_ONLY = "wan_only"            # Only WAN video generation
    TAPIP3D_ONLY = "tapip3d_only"    # Only Tapip3D tracking (requires input video)
    SAM_ONLY = "sam_only"            # Only SAM segmentation (requires input npz)
    WAN_TAPIP3D = "wan_tapip3d"      # WAN -> Tapip3D
    TAPIP3D_SAM = "tapip3d_sam"      # Tapip3D -> SAM (requires input video)
    WAN_SAM = "wan_sam"              # WAN -> SAM (requires input video for SAM)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PipelineJob:
    """Represents a pipeline job"""
    job_id: str
    request_data: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    gpu_id: Optional[int] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    future: Optional[ray.ObjectRef] = None  # Ray future for async execution

# LoadBalancer class removed - using Ray's built-in load balancing instead

@ray.remote(num_gpus=1)
class UnifiedGPUWorker:
    """Unified GPU worker that handles all models on a single GPU"""
    
    def __init__(self, gpu_id: int, tracker: str = "SpaTrackerV2", model_type: str = "wan"):
        """Initialize unified GPU worker"""
        self.gpu_id = gpu_id
        self.tracker = tracker
        self.model_type = model_type
        
        # Activate conda environment in the worker
        import os
        import subprocess
        import sys
        
        # Set up conda environment
        os.environ['CONDA_DEFAULT_ENV'] = 'gizmo'
        os.environ['CONDA_PREFIX'] = '/opt/conda/envs/gizmo'
        os.environ['PATH'] = '/opt/conda/envs/gizmo/bin:' + os.environ.get('PATH', '')
        os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/gizmo/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        
        # Add paths to sys.path (use current_dir which points to the server directory)
        base_path = current_dir
        paths_to_add = [
            base_path,
            os.path.join(base_path, 'wan2.1'),
            os.path.join(base_path, 'grounded_sam_2'),
            os.path.join(base_path, 'tapip3d'),
            os.path.join(base_path, 'tapip3d', 'training'),
            os.path.join(base_path, 'tapip3d', 'datasets'),
            os.path.join(base_path, 'SpaTrackerV2')
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Use Ray's GPU assignment
        try:
            import torch
            # Get the number of available GPUs (Ray assigns one GPU per worker)
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                # Use the GPU assigned by Ray (should be device 0 since Ray assigns one GPU per worker)
                self.device = "cuda:0"
                logger.info(f"GPU worker {gpu_id} using CUDA device 0 (Ray-assigned GPU)")
                print(f"[GPU {self.gpu_id}] Using device cuda:0 (Ray-assigned)", flush=True)
            else:
                self.device = "cpu"
                logger.warning(f"GPU worker {gpu_id} falling back to CPU - no CUDA devices available")
                print(f"[GPU {self.gpu_id}] Falling back to CPU - no CUDA devices available", flush=True)
        except Exception as e:
            self.device = "cuda:0"  # Fallback
            logger.warning(f"GPU worker {gpu_id} using fallback device assignment: {e}")
            print(f"[GPU {self.gpu_id}] Fallback device assignment to cuda:0 due to: {e}", flush=True)
        
        # Model availability flags
        self.wan_available = True
        self.tapip3d_available = True
        self.sam_available = True

        # SAM lazy loading synchronization
        self.sam_loaded = False
        self.sam_loading = False
        self.sam_loading_lock = threading.Lock()
        self.sam_loading_event = threading.Event()
        
        # Model paths (use absolute paths)
        self.wan_flf2v_model_dir = os.path.join(current_dir, 'wan2.1', 'Wan2.1-FLF2V-14B-720P')
        self.wan_i2v_model_dir = os.path.join(current_dir, 'wan2.1', 'Wan2.1-I2V-14B-720P')
        self.tapip3d_checkpoint = os.path.join(current_dir, 'tapip3d', 'checkpoints', 'tapip3d_final.pth')
        self.sam2_checkpoint = os.path.join(current_dir, 'grounded_sam_2', 'checkpoints', 'sam2.1_hiera_large.pt')
        self.sam2_config = os.path.join(current_dir, 'grounded_sam_2', 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_l.yaml')
        
        # Tapip3D settings
        self.tapip3d_resolution_factor = 2
        self.tapip3d_num_iters = 6
        self.tapip3d_support_grid_size = 16
        self.tapip3d_num_threads = 8
        
        # Initialize models
        self._load_all_models()
        
    def _load_all_models(self):
        """Load all models (WAN, Tapip3D) on this GPU"""
        logger.info(f"Loading all models on GPU {self.gpu_id}...")
        print(f"[GPU {self.gpu_id}] Loading models for tracker={self.tracker}...", flush=True)
        
        if self.tracker == 'tapip3d':
            # Load Tapip3D model (directly in this process)
            self._load_tapip3d_model()
        elif self.tracker == 'SpaTrackerV2':
            # Load SpaTrackerV2 model
            self._load_spatrackerv2_model()

        # Load WAN models (most time-consuming)
        if self.model_type == "veo":
            logger.info(f"Skipping WAN models loading on GPU {self.gpu_id} (model_type='veo')")
        else:
            self._load_wan_models()
        
        logger.info(f"All models loaded directly on GPU {self.gpu_id}")
        
        self.models_loaded = True
    
    def get_gpu_id(self) -> int:
        """Get GPU ID - used to check if worker is ready"""
        return self.gpu_id
    
    def _load_wan_models(self):
        """Load WAN models on this GPU"""
        try:
            logger.info(f"Loading WAN models on GPU {self.gpu_id}...")
            
            # Load FLF2V model
            cfg = WAN_CONFIGS["flf2v-14B"]
            self.wan_flf2v_model = wan.WanFLF2V(
                config=cfg,
                checkpoint_dir=self.wan_flf2v_model_dir,
                device_id=0,  # Ray assigns one GPU per worker, so use device_id=0
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
            )
            
            # Load I2V model
            cfg = WAN_CONFIGS["i2v-14B"]
            self.wan_i2v_model = wan.WanI2V(
                config=cfg,
                checkpoint_dir=self.wan_i2v_model_dir,
                device_id=0,  # Ray assigns one GPU per worker, so use device_id=0
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
            )
            
            logger.info(f"WAN models loaded successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            logger.error(f"Error loading WAN models on GPU {self.gpu_id}: {e}")
            self.wan_flf2v_model = None
            self.wan_i2v_model = None
    
    def _load_tapip3d_model(self):
        """Load Tapip3D model on this GPU"""
        if not TAPIP3D_AVAILABLE:
            logger.warning(f"Tapip3D not available on GPU {self.gpu_id}")
            self.tapip3d_model = None
            return
        
        try:
            logger.info(f"Loading Tapip3D model on GPU {self.gpu_id}...")
            print(f"[GPU {self.gpu_id}] Loading Tapip3D model...", flush=True)
            
            # Change working directory to tapip3d for proper relative imports
            import os
            original_cwd = os.getcwd()
            os.chdir(os.path.join(current_dir, 'tapip3d'))

            # Import Tapip3D modules
            from tapip3d.utils.inference_utils import load_model
            from tapip3d.annotation.megasam import MegaSAMAnnotator
            
            # Load Tapip3D model
            self.tapip3d_model = load_model(self.tapip3d_checkpoint)
            self.tapip3d_model.to(self.device)
            
            # Calculate inference resolution using model's image_size
            inference_res = (
                int(self.tapip3d_model.image_size[0] * np.sqrt(self.tapip3d_resolution_factor)), 
                int(self.tapip3d_model.image_size[1] * np.sqrt(self.tapip3d_resolution_factor))
            )
            
            # Set model's image size
            self.tapip3d_model.set_image_size(inference_res)
            
            self.tapip3d_model.eval()
            
            # Store the calculated inference resolution for later use
            self.tapip3d_inference_res = inference_res
            
            # Load MegaSAM annotator
            self.megasam_model = MegaSAMAnnotator(
                script_path=Path(os.path.join(current_dir, 'tapip3d', 'third_party', 'megasam', 'inference.py')),
                depth_model="moge",
                resolution=inference_res[0] * inference_res[1],
                use_gt_intrinsics=True
            )
            self.megasam_model.to(self.device)
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            print(f"[GPU {self.gpu_id}] Tapip3D + MegaSAM ready. inference_res={inference_res}", flush=True)
            
        except Exception as e:
            logger.error(f"Error loading Tapip3D model on GPU {self.gpu_id}: {e}")
            self.tapip3d_model = None
            self.megasam_model = None
            import traceback as _tb
            print(f"[GPU {self.gpu_id}] Error loading Tapip3D: {e}\n{_tb.format_exc()}", flush=True)
    
    def _load_sam_models(self):
        """Load SAM models on this GPU"""
        if not SAM_AVAILABLE:
            logger.warning(f"SAM not available on GPU {self.gpu_id}")
            self.video_predictor = None
            self.image_predictor = None
            self.processor = None
            self.grounding_model = None
            return
        
        try:
            logger.info(f"Loading SAM models on GPU {self.gpu_id}...")
            
            # Add SAM path to sys.path
            import sys
            import os
            sam_path = os.path.join(current_dir, 'grounded_sam_2')
            if sam_path not in sys.path:
                sys.path.append(sam_path)
            
            # Change working directory to grounded_sam_2
            original_cwd = os.getcwd()
            os.chdir(os.path.join(current_dir, 'grounded_sam_2'))
            
            # Import SAM modules
            from sam2.build_sam import build_sam2_video_predictor, build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            # Load SAM2 models using relative config path
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.video_predictor = build_sam2_video_predictor(model_cfg, self.sam2_checkpoint)
            self.video_predictor.to(self.device)
            
            sam2_image_model = build_sam2(model_cfg, self.sam2_checkpoint)
            sam2_image_model.to(self.device)  # Move the model to device first
            self.image_predictor = SAM2ImagePredictor(sam2_image_model)  # SAM2ImagePredictor doesn't have .to() method
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            # Load Grounding DINO model
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            
            logger.info(f"SAM models loaded successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            logger.error(f"Error loading SAM models on GPU {self.gpu_id}: {e}")
            self.video_predictor = None
            self.image_predictor = None
            self.processor = None
            self.grounding_model = None

    def _preload_sam_models(self):
        """Start loading SAM models in background without waiting"""
        with self.sam_loading_lock:
            if self.sam_loaded or self.sam_loading:
                return  # Already loaded or loading

            # Start loading in background
            self.sam_loading = True
            self.sam_loading_event.clear()

            def load_sam_thread():
                try:
                    logger.info(f"Starting background SAM model loading on GPU {self.gpu_id}...")
                    self._load_sam_models()
                    self.sam_loaded = True
                    logger.info(f"SAM models loaded successfully in background on GPU {self.gpu_id}")
                except Exception as e:
                    logger.error(f"Error loading SAM models in background on GPU {self.gpu_id}: {e}")
                    self.sam_loaded = False
                finally:
                    self.sam_loading = False
                    self.sam_loading_event.set()

            thread = threading.Thread(target=load_sam_thread, daemon=True)
            thread.start()

    def _ensure_sam_loaded(self):
        """Ensure SAM models are loaded, waiting if necessary"""
        with self.sam_loading_lock:
            if self.sam_loaded:
                return True
            if self.sam_loading:
                # Wait for loading to complete
                logger.info(f"Waiting for SAM models to finish loading on GPU {self.gpu_id}...")
                self.sam_loading_event.wait()
                return self.sam_loaded

            # If not loaded and not loading, load now (blocking)
            logger.info(f"Loading SAM models synchronously on GPU {self.gpu_id}...")
            try:
                self._load_sam_models()
                self.sam_loaded = True
                return True
            except Exception as e:
                logger.error(f"Error loading SAM models on GPU {self.gpu_id}: {e}")
                self.sam_loaded = False
                return False

    def _unload_sam_models(self):
        """Unload SAM models to free memory"""
        with self.sam_loading_lock:
            if not self.sam_loaded:
                return

            logger.info(f"Unloading SAM models on GPU {self.gpu_id}...")
            try:
                # Clear model references
                self.video_predictor = None
                self.image_predictor = None
                self.processor = None
                self.grounding_model = None

                # Force garbage collection
                import gc
                gc.collect()

                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.sam_loaded = False
                logger.info(f"SAM models unloaded successfully on GPU {self.gpu_id}")

            except Exception as e:
                logger.error(f"Error unloading SAM models on GPU {self.gpu_id}: {e}")

    def _load_spatrackerv2_model(self):
        """Load SpaTrackerV2 model on this GPU"""
        logger.info(f"Loading SpaTrackerV2 model on GPU {self.gpu_id}...")
        from st_models.SpaTrackV2.models.predictor import Predictor
        from st_models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

        self.spatrackerv2_vggt_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        self.spatrackerv2_vggt_model.eval()
        self.spatrackerv2_vggt_model = self.spatrackerv2_vggt_model.to(self.device)

        self.spatrackerv2_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        self.spatrackerv2_model.eval()
        self.spatrackerv2_model.to(self.device)
        logger.info(f"SpaTrackerV2 models loaded successfully on GPU {self.gpu_id}")
    
    def process_pipeline(self, job_id: str, mode: str, first_frame_path: str, prompt: str, output_path: str, 
                        last_frame_path: Optional[str] = None, seed: int = 42, sam_prompt: str = "mug",
                        input_video_path: Optional[str] = None, input_npz_path: Optional[str] = None,
                        config_file_content: Optional[str] = None, use_veo: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Process pipeline based on specified mode"""
        logger.info(f"🎯 Starting {mode} pipeline processing for job {job_id} on GPU {self.gpu_id}")
        print(f"[GPU {self.gpu_id}] Start pipeline job={job_id} mode={mode} seed={seed}", flush=True)
        
        try:
            logger.info(f"🌱 Using seed: {seed}")
            pipeline_start_time = time.time()
            
            # Initialize outputs and timings dictionary
            outputs = {}
            timings = {}
            wan_output_path = None
            
            # Step 1: WAN Video Generation (if needed)
            if mode in [PipelineMode.FULL.value, PipelineMode.WAN_ONLY.value, 
                       PipelineMode.WAN_TAPIP3D.value, PipelineMode.WAN_SAM.value]:
                logger.info(f"📹 Step 1: Video Generation")
                wan_output_path = os.path.join(output_path, f'wan_output_{job_id}.mp4')
                
                wan_start_time = time.time()
                try:
                    if use_veo:
                        self._generate_veo_video(job_id, first_frame_path, prompt, wan_output_path, last_frame_path, seed)
                    else:
                        self._generate_wan_video(first_frame_path, prompt, wan_output_path, last_frame_path, seed)
                finally:
                    timings['wan_video_generation'] = time.time() - wan_start_time
                
                if os.path.exists(wan_output_path):
                    with open(wan_output_path, 'rb') as f:
                        wan_video_bytes = f.read()
                    outputs['wan_video'] = base64.b64encode(wan_video_bytes).decode('utf-8')
                    outputs['wan_video_filename'] = f'wan_output_{job_id}.mp4'

            # Pre-load SAM models in background after video generation (if SAM will be needed)
            sam_needed_modes = [PipelineMode.FULL.value, PipelineMode.SAM_ONLY.value,
                              PipelineMode.TAPIP3D_SAM.value, PipelineMode.WAN_SAM.value]
            if mode in sam_needed_modes and SAM_AVAILABLE:
                logger.info(f"🔄 Starting background SAM model loading after video generation...")
                self._preload_sam_models()

            # Step 2: Tapip3D Tracking (if needed)
            tapip3d_output_path = None
            if mode in [PipelineMode.FULL.value, PipelineMode.TAPIP3D_ONLY.value, 
                       PipelineMode.WAN_TAPIP3D.value, PipelineMode.TAPIP3D_SAM.value]:
                if self.tracker == 'tapip3d':
                    logger.info(f"🎯 Step 2: Tapip3D Tracking")
                    
                    # Determine input video path (use absolute paths for Ray workers)
                    video_input = input_video_path if input_video_path else wan_output_path
                    if not video_input:
                        raise ValueError("No input video provided for Tapip3D")
                    
                    # Convert to absolute path if it's a relative path
                    if not os.path.isabs(video_input):
                        video_input = os.path.abspath(video_input)
                    
                    tapip3d_output_path = os.path.join(output_path, f'tapip3d_output_{job_id}.npz')
                    
                    tapip3d_output_path, tapip3d_timings = self._run_tapip3d(video_input, tapip3d_output_path, config_file_content)
                    if tapip3d_timings:
                        timings.update(tapip3d_timings)

                    if not tapip3d_output_path:
                        logger.error(f"Tapip3D processing failed for job {job_id}")
                    
                    if tapip3d_output_path and os.path.exists(tapip3d_output_path):
                        with open(tapip3d_output_path, 'rb') as f:
                            tapip3d_bytes = f.read()
                        outputs['tapip3d_results'] = base64.b64encode(tapip3d_bytes).decode('utf-8')
                        outputs['tapip3d_results_filename'] = f'tapip3d_output_{job_id}.npz'
                
                elif self.tracker == 'SpaTrackerV2':
                    logger.info(f"🎯 Step 2: SpaTrackerV2 Tracking")
                    
                    # Determine input video path (use absolute paths for Ray workers)
                    video_input = input_video_path if input_video_path else wan_output_path
                    if not video_input:
                        raise ValueError("No input video provided for SpaTrackerV2")
                    
                    # Convert to absolute path if it's a relative path
                    if not os.path.isabs(video_input):
                        video_input = os.path.abspath(video_input)
                    
                    spatracker_output_path = os.path.join(output_path, f'spatracker_output_{job_id}.npz')
                    
                    spatracker_output_path, spatracker_timings = self._run_spatrackerv2(video_input, spatracker_output_path, config_file_content)
                    if spatracker_timings:
                        timings.update(spatracker_timings)

                    if not spatracker_output_path:
                        logger.error(f"SpaTrackerV2 processing failed for job {job_id}")
                    
                    if spatracker_output_path and os.path.exists(spatracker_output_path):
                        with open(spatracker_output_path, 'rb') as f:
                            spatracker_bytes = f.read()
                        outputs['spatracker_results'] = base64.b64encode(spatracker_bytes).decode('utf-8')
                        outputs['spatracker_results_filename'] = f'spatracker_output_{job_id}.npz'
                        tapip3d_output_path = spatracker_output_path # for SAM
            
            # Step 3: SAM Segmentation (if needed)
            sam_output_path = None
            sam_video_path = None
            if mode in [PipelineMode.FULL.value, PipelineMode.SAM_ONLY.value, 
                       PipelineMode.TAPIP3D_SAM.value, PipelineMode.WAN_SAM.value]:
                logger.info(f"🎨 Step 3: SAM Segmentation")

                # Run segmentation on Tapip3D result if available
                if mode in [PipelineMode.FULL.value, PipelineMode.TAPIP3D_SAM.value, PipelineMode.SAM_ONLY.value]:
                    npz_input = input_npz_path if input_npz_path else tapip3d_output_path
                    if npz_input:
                        logger.info("Running SAM on Tapip3D output")
                        
                        sam_start_time = time.time()
                        try:
                            sam_output_path = self._run_sam_segmentation(npz_input, sam_prompt, output_path, job_id)
                        finally:
                            timings['sam_segmentation_tapip3d'] = time.time() - sam_start_time
                        
                        if sam_output_path and os.path.exists(sam_output_path):
                            with open(sam_output_path, 'rb') as f:
                                sam_bytes = f.read()
                            outputs['sam_segmentation'] = base64.b64encode(sam_bytes).decode('utf-8')
                            outputs['sam_segmentation_filename'] = f'segmentation_masks_{job_id}.npz'
                        
                        sam_video_path = os.path.join(output_path, f'segmentation_video_{job_id}.mp4')
                        if os.path.exists(sam_video_path):
                            with open(sam_video_path, 'rb') as f:
                                sam_video_bytes = f.read()
                            outputs['sam_segmentation_video'] = base64.b64encode(sam_video_bytes).decode('utf-8')
                            outputs['sam_segmentation_video_filename'] = f'segmentation_video_{job_id}.mp4'
                            logger.info(f"✅ SAM segmentation video included in response: {sam_video_path}")
                        else:
                            logger.warning(f"⚠️  SAM segmentation video not found: {sam_video_path}")
                    else:
                        logger.warning("No NPZ input for SAM in a mode that expects it.")

                # Run segmentation on WAN result if available
                if mode in [PipelineMode.FULL.value, PipelineMode.WAN_SAM.value]:
                    video_input_for_sam = wan_output_path if 'wan_output_path' in locals() and wan_output_path else input_video_path
                    if video_input_for_sam:
                        logger.info("Running SAM on WAN output")
                        
                        sam_wan_start_time = time.time()
                        try:
                            sam_output_full_path = self._run_sam_segmentation(video_input_for_sam, sam_prompt, output_path, job_id)
                        finally:
                            timings['sam_segmentation_wan_visualization'] = time.time() - sam_wan_start_time

                        if sam_output_full_path and os.path.exists(sam_output_full_path):
                             with open(sam_output_full_path, 'rb') as f:
                                sam_bytes = f.read()
                             outputs['sam_segmentation_full'] = base64.b64encode(sam_bytes).decode('utf-8')
                             outputs['sam_segmentation_full_filename'] = f'segmentation_masks_full_{job_id}.npz'

                        sam_video_full_path = os.path.join(output_path, f'segmentation_video_full_{job_id}.mp4')
                        if os.path.exists(sam_video_full_path):
                            with open(sam_video_full_path, 'rb') as f:
                                sam_video_bytes = f.read()
                            outputs['sam_segmentation_video_full'] = base64.b64encode(sam_video_bytes).decode('utf-8')
                            outputs['sam_segmentation_video_full_filename'] = f'segmentation_video_full_{job_id}.mp4'
                            logger.info(f"✅ SAM segmentation full video included in response: {sam_video_full_path}")
                        else:
                            logger.warning(f"⚠️  SAM segmentation full video not found: {sam_video_full_path}")
                    else:
                        logger.warning("No video input for SAM in a mode that expects it.")

            pipeline_elapsed_time = time.time() - pipeline_start_time
            logger.info(f"🎉 {mode} pipeline completed successfully for job {job_id} in {pipeline_elapsed_time:.2f} seconds!")
            
            return job_id, {
                'status': 'success',
                'mode': mode,
                'outputs': outputs,
                'processing_time': pipeline_elapsed_time,
                'timings': timings
            }
            
        except Exception as e:
            logger.error(f"Error in {mode} pipeline on GPU {self.gpu_id}: {e}")
            return job_id, {
                'status': 'error',
                'mode': mode,
                'error': str(e)
            }
        finally:
            # Clear GPU memory cache after job completion
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"🧹 Cleared GPU {self.gpu_id} memory cache after job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to clear GPU {self.gpu_id} cache: {e}")
            
            logger.info(f"Released job {job_id} from GPU {self.gpu_id}")
    
    def _generate_wan_video(self, first_frame_path: str, prompt: str, output_path: str, 
                           last_frame_path: Optional[str] = None, seed: int = 42) -> Optional[str]:
        """Generate video using WAN2.1 models directly loaded in GPU worker"""
        logger.info(f"Generating video with WAN2.1 from {first_frame_path} on GPU {self.gpu_id}")
        
        # clean torch cache
        torch.cuda.empty_cache()
        
        try:
            # Load input image
            first_frame = Image.open(first_frame_path).convert("RGB")
            
            # Determine model type and generate
            use_flf2v = last_frame_path is not None
            
            if use_flf2v:
                # FLF2V mode
                if self.wan_flf2v_model is None:
                    logger.error(f"FLF2V model not loaded on GPU {self.gpu_id}")
                    self._create_dummy_video(output_path)
                    return output_path
                
                logger.info(f"Using FLF2V model (first + last frame) on GPU {self.gpu_id}")
                last_frame = Image.open(last_frame_path).convert("RGB")
                
                logger.info(f"🚀 Starting FLF2V generation on GPU {self.gpu_id}...")
                start_time = time.time()
                
                video = self.wan_flf2v_model.generate(
                    prompt,
                    first_frame,
                    last_frame,
                    max_area=MAX_AREA_CONFIGS["1280*720"],
                    frame_num=41,
                    shift=5.0,  # Default for FLF2V
                    sample_solver="unipc",
                    sampling_steps=40,  # Default for I2V tasks
                    guide_scale=5.0,
                    seed=seed,
                    offload_model=True
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ FLF2V generation completed on GPU {self.gpu_id} in {elapsed_time:.2f} seconds")
            else:
                # I2V mode
                if self.wan_i2v_model is None:
                    logger.error(f"I2V model not loaded on GPU {self.gpu_id}")
                    self._create_dummy_video(output_path)
                    return output_path
                
                logger.info(f"Using I2V model (first frame only) on GPU {self.gpu_id}")
                
                logger.info(f"🚀 Starting I2V generation on GPU {self.gpu_id}...")
                start_time = time.time()
                
                video = self.wan_i2v_model.generate(
                    prompt,
                    first_frame,
                    max_area=MAX_AREA_CONFIGS["1280*720"],
                    frame_num=41,
                    shift=5.0,  # Default for I2V
                    sample_solver="unipc",
                    sampling_steps=40,  # Default for I2V tasks
                    guide_scale=5.0,
                    seed=seed,
                    offload_model=True
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ I2V generation completed on GPU {self.gpu_id} in {elapsed_time:.2f} seconds")
            
            # Save video
            if video is not None:
                from wan.utils.utils import cache_video
                # Get FPS from config (default to 24 if not available)
                if last_frame_path is not None:
                    cfg = WAN_CONFIGS["flf2v-14B"]
                else:
                    cfg = WAN_CONFIGS["i2v-14B"]
                fps = getattr(cfg, 'sample_fps', 24)
                cache_video(
                    tensor=video[None],
                    save_file=output_path,
                    fps=fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                logger.info(f"WAN video saved to {output_path} from GPU {self.gpu_id}")
            else:
                logger.error(f"Video generation failed on GPU {self.gpu_id}")
                self._create_dummy_video(output_path)
                
        except Exception as e:
            logger.error(f"Error generating video on GPU {self.gpu_id}: {e}")
            self._create_dummy_video(output_path)
        
        return output_path
        
    def _create_dummy_video(self, output_path: str):
        """Create a dummy video for demonstration purposes"""
        # Create a simple test video (in practice, this would be the actual WAN output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, 16.0, (1280, 720))
        
        # Create 21 frames of a simple animation
        for i in range(21):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add some simple animation
            cv2.circle(frame, (640 + i*10, 360), 50, (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
    
    def _run_tapip3d(self, video_path: str, output_npz: str, config_file_content: Optional[str] = None) -> Optional[str]:
        """Run Tapip3D on the generated video using loaded model"""
        if not self.tapip3d_model:
            logger.warning(f"Tapip3D model not loaded on GPU {self.gpu_id}")
            return None
        
        tapip3d_timings = {}
        try:
            logger.info(f"🚀 Starting Tapip3D processing on GPU {self.gpu_id}...")
            print(f"[GPU {self.gpu_id}] Tapip3D: start on {video_path}", flush=True)
            start_time = time.time()
            
            # Change working directory to tapip3d for proper relative imports
            import os
            import sys
            original_cwd = os.getcwd()
            tapip3d_path = os.path.join(current_dir, 'tapip3d')
            os.chdir(tapip3d_path)
            
            # Add tapip3d to sys.path so modules can be found
            if tapip3d_path not in sys.path:
                sys.path.insert(0, tapip3d_path)
            
            # Import Tapip3D inference functions
            from tapip3d.utils.inference_utils import inference, read_video, get_grid_queries, resize_depth_bilinear
            from tapip3d.datasets.data_ops import _filter_one_depth
            from einops import repeat
            import cv2
            from concurrent.futures import ThreadPoolExecutor
            
            # Prepare inputs for Tapip3D (custom implementation to avoid args dependency)
            # Use the pre-calculated inference resolution from model loading
            inference_res = self.tapip3d_inference_res
            support_grid_size = self.tapip3d_support_grid_size
            num_threads = self.tapip3d_num_threads
            
            # Ensure video_path is absolute and exists
            if not os.path.isabs(video_path):
                video_path = os.path.abspath(video_path)
            
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                os.chdir(original_cwd)
                return None
            
            logger.info(f"Processing video: {video_path}")
            
            logger.info(f"Reading video from: {video_path}")
            
            # Read video
            video = read_video(video_path)
            
            # Parse intrinsics from config file if provided
            gt_intrinsics = None

            # If config_file_content is not provided, try to read from output directory
            if not config_file_content:
                config_path = os.path.join(os.path.dirname(output_npz), "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_file_content = f.read()

            if config_file_content:
                try:
                    config_data = json.loads(config_file_content)
                    # Following the structure from the provided config.json example
                    if 'camera_intrinsics_and_extrinsics' in config_data and 'depth_intrinsics' in config_data['camera_intrinsics_and_extrinsics']:
                        intrinsics_data = config_data['camera_intrinsics_and_extrinsics']['depth_intrinsics']
                        fx = intrinsics_data['fx']
                        fy = intrinsics_data['fy']
                        cx = intrinsics_data['ppx']
                        cy = intrinsics_data['ppy']
                        intrinsic_matrix = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        # As seen in the megasam.py snippet, gt_intrinsics is expected to be an array of matrices.
                        # We repeat the same intrinsic matrix for all frames in the video.
                        num_frames = len(video)
                        gt_intrinsics = repeat(intrinsic_matrix, 'i j -> t i j', t=num_frames)
                        logger.info("Successfully parsed intrinsics from config file and repeated for all frames.")

                        # get the generated video shape
                        video_H, video_W = video.shape[1:3]
                        # get the original image shape (correspond to the gt intrinsics)
                        original_H, original_W = intrinsics_data["height"], intrinsics_data["width"]
                        gt_intrinsics[:, 0, :] *= (video_W) / (original_W)
                        gt_intrinsics[:, 1, :] *= (video_H) / (original_H)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse config_file_content as JSON: {e}")
                except KeyError as e:
                    logger.warning(f"Key not found when parsing intrinsics from config file: {e}")

            # Generate proper depths using MegaSAM
            logger.info(f"Generating depths using MegaSAM for Tapip3D on GPU {self.gpu_id}")
            print(f"[GPU {self.gpu_id}] MegaSAM: processing video for depths...", flush=True)
            megasam_start_time = time.time()

            if not self.megasam_model:
                logger.error(f"MegaSAM model not loaded on GPU {self.gpu_id}")
                os.chdir(original_cwd)
                return None, {}
                
            # Process video to get depths, intrinsics, and extrinsics
            depths, intrinsics, extrinsics = self.megasam_model.process_video(video, gt_intrinsics=gt_intrinsics, return_raw_depths=True)
            tapip3d_timings['megasam'] = time.time() - megasam_start_time
            try:
                print(f"[GPU {self.gpu_id}] MegaSAM done in {tapip3d_timings['megasam']:.2f}s; depths={None if depths is None else depths.shape}, intrinsics={None if intrinsics is None else intrinsics.shape}, extrinsics={None if extrinsics is None else extrinsics.shape}", flush=True)
            except Exception:
                pass

            # Ensure intrinsics have per-frame shape (T, 3, 3)
            try:
                if intrinsics is not None and getattr(intrinsics, 'ndim', 0) == 2:
                    intrinsics = np.repeat(intrinsics[None, :, :], depths.shape[0], axis=0)
                    logger.info(f"Expanded intrinsics to per-frame shape: {intrinsics.shape}")
            except Exception as _shape_err:
                logger.warning(f"Failed to expand intrinsics shape; proceeding as-is: {getattr(_shape_err, 'args', '')}")

            # Log shapes before resizing/scaling
            try:
                logger.info(f"Shapes before resize: video={video.shape}, depths={depths.shape}, intrinsics={None if intrinsics is None else intrinsics.shape}, extrinsics={None if extrinsics is None else extrinsics.shape}")
            except Exception:
                pass
            _original_res = video.shape[1:3]
            
            # Scale intrinsics for new resolution
            intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
            intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)
            
            # Resize video and depths
            with ThreadPoolExecutor(num_threads) as executor:
                video_futures = [executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR) for rgb in video]
                depths_futures = [executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) for depth in depths]
                
                video = np.stack([future.result() for future in video_futures])
                depths = np.stack([future.result() for future in depths_futures])
                
                # Apply depth filtering
                depths_futures = [executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) for depth, intrinsic in zip(depths, intrinsics)]
                depths = np.stack([future.result() for future in depths_futures])
            
            # Convert to tensors
            video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(self.device)
            depths = torch.from_numpy(depths).float().to(self.device)
            intrinsics = torch.from_numpy(intrinsics).float().to(self.device)
            extrinsics = torch.from_numpy(extrinsics).float().to(self.device)
            
            # Generate query points
            query_point = get_grid_queries(grid_size=32, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
            
            # Run inference
            tapip3d_inference_start_time = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coords, visibs = inference(
                    model=self.tapip3d_model,
                    video=video,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    query_point=query_point,
                    num_iters=self.tapip3d_num_iters,
                    grid_size=support_grid_size,
                )
            
            tapip3d_timings['tapip3d_inference'] = time.time() - tapip3d_inference_start_time
            # Save results
            video = video.cpu().numpy()
            depths = depths.cpu().numpy()
            intrinsics = intrinsics.cpu().numpy()
            extrinsics = extrinsics.cpu().numpy()
            coords = coords.cpu().numpy()
            visibs = visibs.cpu().numpy()
            query_point = query_point.cpu().numpy()
            
            logger.info(f"Saving Tapip3D results to: {output_npz}")
            print(f"[GPU {self.gpu_id}] Saving Tapip3D results to {output_npz}", flush=True)
            np.savez(
                output_npz,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                coords=coords,
                visibs=visibs,
                query_points=query_point,
            )
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            elapsed_time = time.time() - start_time
            
            # Verify file was created
            if os.path.exists(output_npz):
                file_size = os.path.getsize(output_npz)
                logger.info(f"✅ Tapip3D processing completed on GPU {self.gpu_id} in {elapsed_time:.2f} seconds")
                logger.info(f"Tapip3D results saved to {output_npz} (size: {file_size} bytes) from GPU {self.gpu_id}")
                return output_npz, tapip3d_timings
            else:
                logger.error(f"❌ Tapip3D processing failed - output file not created: {output_npz}")
                return None, {}
            
        except Exception as e:
            logger.error(f"Error running Tapip3D on GPU {self.gpu_id}: {e}")
            import traceback as _tb
            print(f"[GPU {self.gpu_id}] Tapip3D error: {e}\n{_tb.format_exc()}", flush=True)
            return None, {}
    
    def _run_sam_segmentation(self, input_path: str, text_prompt: str, output_dir: str, job_id: str) -> Optional[str]:
        """Run SAM segmentation on the Tapip3D results or a video file using lazy-loaded models"""
        # Ensure SAM models are loaded before proceeding
        if not self._ensure_sam_loaded():
            logger.error(f"Failed to load SAM models on GPU {self.gpu_id}")
            return None

        try:
            # Ensure text prompt ends with a period for SAM
            if not text_prompt.endswith("."):
                text_prompt = text_prompt + "."
            
            logger.info(f"🚀 Starting SAM segmentation on GPU {self.gpu_id} with text prompt: '{text_prompt}' on input: {input_path}")
            start_time = time.time()
            
            is_npz_input = input_path.endswith('.npz')

            # Load data from npz file or mp4 file
            if is_npz_input:
                data = np.load(input_path)
                video = data['video']  # Shape: (T, C, H, W)
            else: # mp4 input
                cap = cv2.VideoCapture(input_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                cap.release()
                video_np = np.array(frames) # T, H, W, C
                from einops import rearrange
                video = rearrange(video_np, 't h w c -> t c h w') / 255.0
            
            # Convert video to RGB format
            from einops import rearrange
            T, C, H, W = video.shape
            rgb_video = (rearrange(video, "T C H W -> T H W C") * 255).astype(np.uint8)
            
            # Process RGB data
            # Use fixed size default for npz
            if is_npz_input:
                fixed_size = (543, 724)
                assert rgb_video.shape[1:3] == fixed_size, f"RGB video shape {rgb_video.shape[1:3]} does not match fixed size {fixed_size}"
            else:
                # For mp4, the size is likely 720x1280, we don't need to assert
                logger.info(f"Processing MP4 video with shape {rgb_video.shape[1:3]}")

            # Save frames to a temporary directory for SAM2 video predictor
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            try:
                # Save frames to temporary directory
                for frame_idx in range(len(rgb_video)):
                    frame_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
                    if is_npz_input:
                        cv2.imwrite(frame_path, cv2.cvtColor(rgb_video[frame_idx], cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(frame_path, rgb_video[frame_idx])
                
                # Initialize video predictor state with directory path
                inference_state = self.video_predictor.init_state(video_path=temp_dir)
                
                # Process first frame for object detection
                first_frame = rgb_video[0]
                image = Image.fromarray(first_frame)
                
                # Run Grounding DINO on the first frame
                inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)
                
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.25,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]]
                )
                
                # Check if any objects were detected
                if len(results[0]["boxes"]) == 0:
                    logger.warning(f"No objects detected for prompt '{text_prompt}' on GPU {self.gpu_id}")
                    return None
                
                # Select highest-confidence detection
                scores_tensor = results[0]["scores"]
                max_idx = int(torch.argmax(scores_tensor).item())
                highest_box = results[0]["boxes"][max_idx].unsqueeze(0).cpu().numpy()
                labels_list = results[0].get("text_labels", results[0]["labels"])
                highest_label = labels_list[max_idx]
                logger.info(
                    f"Using highest-confidence detection: '{highest_label}' (score={scores_tensor[max_idx].item():.3f})"
                )

                # Use only the highest-confidence box and label downstream
                input_boxes = highest_box
                OBJECTS = [highest_label]
                
                # Set image for SAM image predictor
                self.image_predictor.set_image(first_frame)
                
                # Get masks for the detected objects
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                
                # Process masks for video
                if masks.ndim == 3:
                    masks = masks[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                # Add boxes to video predictor for the first frame
                ann_frame_idx = 0
                for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        box=box,
                    )
                
                # Propagate through video to get segmentation results
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()  # Ensure 2D mask
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
                # Convert to list of masks for each frame
                all_binary_masks = []
                for frame_idx in range(len(rgb_video)):
                    if frame_idx in video_segments:
                        # Combine all object masks for this frame
                        frame_masks = list(video_segments[frame_idx].values())
                        if frame_masks:
                            # Combine all masks into a single binary mask
                            binary_mask = np.zeros((H, W), dtype=np.uint8)
                            for mask in frame_masks:
                                # Ensure mask has the same dimensions as the frame
                                if mask.shape != (H, W):
                                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                                binary_mask = np.logical_or(binary_mask, mask)
                            binary_mask = binary_mask.astype(np.uint8)
                            all_binary_masks.append(binary_mask)
                        else:
                            all_binary_masks.append(np.zeros((H, W), dtype=np.uint8))
                    else:
                        all_binary_masks.append(np.zeros((H, W), dtype=np.uint8))
                
                # Save results based on input type
                if is_npz_input:
                    segmentation_path = os.path.join(output_dir, f"segmentation_masks_{job_id}.npz")
                    visualization_path = os.path.join(output_dir, f"segmentation_video_{job_id}.mp4")
                else:
                    segmentation_path = os.path.join(output_dir, f"segmentation_masks_full_{job_id}.npz")
                    visualization_path = os.path.join(output_dir, f"segmentation_video_full_{job_id}.mp4")

                binary_masks_array = np.array(all_binary_masks, dtype=np.uint8)  # Shape: (T, H, W)
                np.savez(segmentation_path, masks=binary_masks_array)

                # Generate visualization video
                logger.info(f"🎬 Creating segmentation video at: {visualization_path}")
                
                try:
                    self._create_segmentation_video(rgb_video, binary_masks_array, visualization_path)
                    
                    # Verify the video was created
                    if os.path.exists(visualization_path):
                        file_size = os.path.getsize(visualization_path)
                        logger.info(f"✅ Segmentation video created successfully: {visualization_path} (size: {file_size} bytes)")
                    else:
                        logger.error(f"❌ Segmentation video was not created: {visualization_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Error creating segmentation video: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ SAM segmentation completed on GPU {self.gpu_id} in {elapsed_time:.2f} seconds")
                logger.info(f"SAM segmentation results saved to {segmentation_path} from GPU {self.gpu_id}")
                logger.info(f"Binary masks shape: {binary_masks_array.shape} (frames, height, width)")
                return segmentation_path
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Error running SAM segmentation on GPU {self.gpu_id}: {e}")
            return None
        finally:
            # Unload SAM models to free memory
            self._unload_sam_models()
    
    def _create_segmentation_video(self, rgb_video, binary_masks_array, output_path):
        """Create a segmentation visualization video from RGB video and binary masks"""
        try:
            import cv2
            import numpy as np
            import os
            import tempfile
            import shutil
            
            # Try to import tqdm, fallback to range if not available
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda x, **kwargs: x  # Fallback to regular range
            
            logger.info(f"Creating segmentation visualization video on GPU {self.gpu_id}")
            
            # Create temporary directory for annotated frames
            temp_dir = tempfile.mkdtemp()
            try:
                # Process each frame
                for frame_idx in tqdm(range(len(rgb_video)), desc="Creating annotated frames"):
                    # Get original frame and mask
                    frame = rgb_video[frame_idx]  # RGB format
                    mask = binary_masks_array[frame_idx]  # Binary mask
                    
                    # Get frame dimensions
                    frame_height, frame_width = frame.shape[:2]
                    mask_height, mask_width = mask.shape[:2]
                    
                    # Ensure mask has the same dimensions as the frame
                    if mask.shape != (frame_height, frame_width):
                        logger.info(f"Resizing mask from {mask.shape} to {(frame_height, frame_width)}")
                        mask = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Create colored overlay for segmentation
                    overlay = np.zeros_like(frame_bgr)
                    
                    # Use the mask for boolean indexing (now dimensions should match)
                    mask_indices = mask > 0  # Get boolean mask (H, W)
                    
                    # Apply green overlay to segmented regions
                    overlay[mask_indices] = [0, 255, 0]  # Green color for segmented regions
                    
                    # Blend original frame with overlay (50% transparency)
                    alpha = 0.5
                    annotated_frame = cv2.addWeighted(frame_bgr, 1-alpha, overlay, alpha, 0)
                    
                    # Add text label
                    cv2.putText(annotated_frame, f"Frame {frame_idx}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save annotated frame
                    frame_path = os.path.join(temp_dir, f"annotated_frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(frame_path, annotated_frame)
                
                # Create video from annotated frames using the proper function
                self._create_video_from_images(temp_dir, output_path, frame_rate=16)
                
                logger.info(f"Segmentation video created: {output_path}")
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Error creating segmentation video on GPU {self.gpu_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def _create_video_from_images(self, image_folder, output_video_path, frame_rate=16):
        """Create video from images in a folder (similar to utils.video_utils.create_video_from_images)"""
        try:
            import imageio
            import os
            
            # Try to import tqdm, fallback to range if not available
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda x, **kwargs: x  # Fallback to regular range
            
            # Define valid extensions
            valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
            
            # Get all image files in the folder
            image_files = [f for f in os.listdir(image_folder) 
                          if os.path.splitext(f)[1] in valid_extensions]
            image_files.sort()  # Sort the files in alphabetical order
            
            if not image_files:
                raise ValueError("No valid image files found in the specified folder.")
            
            # Create a video writer using imageio
            writer = imageio.get_writer(output_video_path, fps=frame_rate, codec='libx264', quality=8)
            
            # Write each image to the video
            for image_file in tqdm(image_files, desc="Creating video"):
                image_path = os.path.join(image_folder, image_file)
                image = imageio.imread(image_path)
                writer.append_data(image)
            
            # Release the writer
            writer.close()
            logger.info(f"Video saved at {output_video_path}")
            
        except Exception as e:
            logger.error(f"Error creating video from images: {e}")
            raise

    def _generate_veo_video(self, job_id: str, first_frame_path: str, prompt: str, output_path: str, 
                             last_frame_path: Optional[str] = None, seed: int = 42) -> Optional[str]:
        """Generate video using Google's Veo API via Vertex AI (GCS upload + poll + download)."""
        logger.info(f"Generating video with Veo (Vertex) from {first_frame_path} on GPU {self.gpu_id}")

        try:

            # 1) Upload first (and last, if provided) frames to use with Gemini API
            # For Veo 3.1 Preview, we use inline image bytes to simplify and avoid GCS logic
            
            # Read first frame
            if not os.path.exists(first_frame_path):
                logger.error(f"First frame not found at {first_frame_path}")
                self._create_dummy_video(output_path)
                return output_path
                
            with open(first_frame_path, "rb") as f:
                first_frame_bytes = f.read()

            # Initialize client with API key from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY environment variable not set")
                self._create_dummy_video(output_path)
                return output_path
                
            client = genai.Client(api_key=api_key)

            first_frame_image = types.Image(
                image_bytes=first_frame_bytes,
                mime_type="image/png",
            )
            
            config_kwargs = dict(
                duration_seconds=4,
            )
            
            # Handle last frame if present
            # Note: The test script didn't cover last frame, but the logic is similar.
            # Veo 3.1 supports last frame via config? 
            # Docs: config=types.GenerateVideosConfig(last_frame=last_image)
            if last_frame_path:
                if not os.path.exists(last_frame_path):
                    logger.error(f"Last frame not found at {last_frame_path}")
                else:
                    with open(last_frame_path, "rb") as f:
                        last_frame_bytes = f.read()
                    
                    last_frame_image = types.Image(
                        image_bytes=last_frame_bytes,
                        mime_type="image/png",
                    )
                    config_kwargs["last_frame"] = last_frame_image

            # 2) Call Veo API
            logger.info(f"Calling Veo 3.1 API for job {job_id}...")
            
            # Use appropriate model
            model_name = "veo-3.1-generate-preview"
            
            operation = client.models.generate_videos(
                model=model_name,
                prompt=prompt,
                image=first_frame_image,
                config=GenerateVideosConfig(**config_kwargs),
            )

            logger.info(f"Veo generation started. Operation: {operation.name}")

            # 3) Poll the operation
            while not operation.done:
                time.sleep(10) # 10s poll interval as per docs example
                operation = client.operations.get(operation)
                logger.info(f"Waiting for Veo generation... Status: {operation.metadata if hasattr(operation, 'metadata') else 'unknown'}")

            # 4) Download result
            if getattr(operation, "result", None):
                generated_videos = operation.result.generated_videos
                if not generated_videos:
                    logger.error("No videos were generated by Veo")
                    self._create_dummy_video(output_path)
                    return output_path

                video_result = generated_videos[0]
                logger.info(f"Generated video URI: {getattr(video_result.video, 'uri', 'unknown')}")

                try:
                    logger.info(f"Downloading video to {output_path}...")
                    video_bytes = client.files.download(file=video_result.video)
                    
                    with open(output_path, "wb") as f:
                        f.write(video_bytes)
                    
                    logger.info(f"Video downloaded to {output_path}")

                    # Downsample the video to 41 frames, replacing the original
                    temp_downsampled_output_path = f"{os.path.splitext(output_path)[0]}_temp_downsampled.mp4"
                    self._downsample_video(output_path, temp_downsampled_output_path, 41)
                    
                    # Replace original with downsampled version
                    if os.path.exists(temp_downsampled_output_path):
                        shutil.move(temp_downsampled_output_path, output_path)
                        logger.info(f"Replaced original video with downsampled version at {output_path}")

                except Exception as download_err:
                    logger.error(f"Failed to download generated video: {download_err}")
                    self._create_dummy_video(output_path)
                    return output_path
            else:
                if hasattr(operation, "error") and operation.error:
                    raise Exception(f"Veo operation failed: {operation.error}")
                logger.error("Veo operation completed without response.")
                self._create_dummy_video(output_path)
                return output_path

        except Exception as e:
            logger.error(f"Error generating Veo video on GPU {self.gpu_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._create_dummy_video(output_path)
            return output_path

    def _create_dummy_video(self, output_path: str):
        """Create a dummy video for demonstration purposes"""
        # Create a simple test video (in practice, this would be the actual WAN output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, 16.0, (1280, 720))
        
        # Create 21 frames of a simple animation
        for i in range(21):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add some simple animation
            cv2.circle(frame, (640 + i*10, 360), 50, (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
    
    def _downsample_video(self, input_path: str, output_path: str, target_frame_count: int):
        """Downsamples a video to a specific frame count using ffmpeg."""
        if not os.path.exists(input_path):
            logger.error(f"Error: Input video not found at {input_path}")
            return
        
        try:
            # Get video duration
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", input_path
            ]
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())

            fps = target_frame_count / duration

            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vf",
                f"fps={fps}",
                "-y",
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Video downsampled to ~{target_frame_count} frames and saved to {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Error downsampling video: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                logger.error(f"ffmpeg stdout: {e.stdout}")
                logger.error(f"ffmpeg stderr: {e.stderr}")
    
    def _run_spatrackerv2(self, video_path: str, output_npz: str, config_file_content: Optional[str] = None) -> Optional[str]:
        """Run SpaTrackerV2 on the generated video using loaded model"""
        if not hasattr(self, 'spatrackerv2_model') or self.spatrackerv2_model is None or \
           not hasattr(self, 'spatrackerv2_vggt_model') or self.spatrackerv2_vggt_model is None:
            logger.error(f"SpaTrackerV2 models are not available on GPU {self.gpu_id}, skipping tracking.")
            return None, {}
        
        spatracker_timings = {}
        try:
            logger.info(f"🚀 Starting SpaTrackerV2 processing on GPU {self.gpu_id}...")
            start_time = time.time()
            
            from tapip3d.utils.inference_utils import read_video
            from st_models.SpaTrackV2.models.utils import get_points_on_a_grid
            from st_models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

            video = read_video(video_path)
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(self.device)

            # Preprocess video and predict depth
            video_tensor_processed = preprocess_image(video_tensor.cpu())[None].to(self.device)
            print("Predicting depth on video shape: ", video_tensor_processed.shape)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    predictions = self.spatrackerv2_vggt_model(video_tensor_processed / 255.0)
                    depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
            
            video_tensor_processed = video_tensor_processed.squeeze(0)
            depth_tensor = depth_map.squeeze(0)
            unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
            
            # Clear memory
            del predictions
            del depth_map
            torch.cuda.empty_cache()

            # Set tracker point number
            self.spatrackerv2_model.spatrack.track_num = 1024

            gt_intrinsics = None
            if config_file_content:
                try:
                    config_data = json.loads(config_file_content)
                    if 'camera_intrinsics_and_extrinsics' in config_data and 'depth_intrinsics' in config_data['camera_intrinsics_and_extrinsics']:
                        intrinsics_data = config_data['camera_intrinsics_and_extrinsics']['depth_intrinsics']
                        fx = intrinsics_data['fx']
                        fy = intrinsics_data['fy']
                        cx = intrinsics_data['ppx']
                        cy = intrinsics_data['ppy']
                        intrinsic_matrix = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        num_frames = len(video)
                        gt_intrinsics = np.repeat(intrinsic_matrix[np.newaxis, :, :], num_frames, axis=0)
                        
                        video_H, video_W = video_tensor_processed.shape[2:]
                        original_H, original_W = intrinsics_data["height"], intrinsics_data["width"]
                        gt_intrinsics[:, 0, :] *= (video_W) / (original_W)
                        gt_intrinsics[:, 1, :] *= (video_H) / (original_H)
                        logger.info("Successfully parsed intrinsics from config file.")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse intrinsics from config file: {e}")

            extrinsics = np.eye(4)[np.newaxis, :, :].repeat(len(video_tensor), axis=0)

            frame_H, frame_W = video_tensor_processed.shape[2:]
            grid_pts = get_points_on_a_grid(32, (frame_H, frame_W), device="cpu")
            query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                (
                    c2w_traj, intrs, point_map, conf_depth,
                    track3d_pred, track2d_pred, vis_pred, conf_pred, video_out
                ) = self.spatrackerv2_model.forward(video_tensor_processed, depth=depth_tensor,
                                    intrs=gt_intrinsics, extrs=extrinsics, 
                                    queries=query_xyt,
                                    fps=1, full_point=False, iters_track=4,
                                    query_no_BA=True, fixed_cam=True, stage=1, unc_metric=unc_metric,
                                    support_frame=len(video_tensor_processed)-1, replace_ratio=0.2)

            # Resize results
            max_size = 1280
            h, w = video_out.shape[2:]
            scale = min(max_size / h, max_size / w)
            if scale < 1:
                new_h, new_w = int(h * scale), int(w * scale)
                video_out = T.Resize((new_h, new_w))(video_out)
                video_tensor_processed = T.Resize((new_h, new_w))(video_tensor_processed)
                point_map = T.Resize((new_h, new_w))(point_map)
                conf_depth = T.Resize((new_h, new_w))(conf_depth)
                track2d_pred[...,:2] = track2d_pred[...,:2] * scale
                intrs[:,:2,:] = intrs[:,:2,:] * scale

            data_to_save = {}
            data_to_save["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
            data_to_save["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            data_to_save["intrinsics"] = intrs.cpu().numpy()
            depth_save = point_map[:,2,...]
            depth_save[conf_depth<0.5] = 0
            data_to_save["depths"] = depth_save.cpu().numpy()
            data_to_save["video"] = (video_tensor_processed).cpu().numpy() / 255.0
            data_to_save["visibs"] = vis_pred.cpu().numpy()
            data_to_save["unc_metric"] = conf_depth.cpu().numpy()
            np.savez(output_npz, **data_to_save)

            elapsed_time = time.time() - start_time
            logger.info(f"✅ SpaTrackerV2 processing completed on GPU {self.gpu_id} in {elapsed_time:.2f} seconds")
            
            return output_npz, spatracker_timings
            
        except Exception as e:
            logger.error(f"Error running SpaTrackerV2 on GPU {self.gpu_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, {}
    
class UnifiedRayPipelineServer:
    """Unified Ray-based pipeline server with all models on each GPU"""
    
    def __init__(self, port: int = 5555, num_gpus: int = 8, enable_cleanup: bool = True, tracker: str = "SpaTrackerV2", model_type: str = "wan"):
        """Initialize the unified Ray pipeline server"""
        self.port = port
        self.num_gpus = num_gpus
        self.enable_cleanup = enable_cleanup  # Control whether to clean up intermediate files
        self.tracker = tracker
        self.model_type = model_type
        
        # Initialize ZMQ socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Initialize Ray with simple multi-GPU configuration
        if not ray.is_initialized():
            # Set up runtime environment
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": f"{os.path.join(current_dir, 'SpaTrackerV2')}:{current_dir}:{os.path.join(current_dir, 'wan2.1')}:{os.path.join(current_dir, 'grounded_sam_2')}:{os.path.join(current_dir, 'tapip3d')}:{os.path.join(current_dir, 'tapip3d', 'training')}:{os.path.join(current_dir, 'tapip3d', 'datasets')}",
                    "CONDA_DEFAULT_ENV": "gizmo",
                    "CONDA_PREFIX": "/opt/conda/envs/gizmo",
                    "PATH": "/opt/conda/envs/gizmo/bin:" + os.environ.get('PATH', ''),
                    "LD_LIBRARY_PATH": "/opt/conda/envs/gizmo/lib:" + os.environ.get('LD_LIBRARY_PATH', ''),
                    "RAY_DISABLE_IMPORT_WARNING": "1"
                }
            }
            
            # Ray initialization with automatic CPU detection
            import multiprocessing
            total_cpus = multiprocessing.cpu_count()
            logger.info(f"Initializing Ray with {total_cpus} CPU cores and {num_gpus} GPUs")
            
            ray.init(
                num_cpus=total_cpus,
                num_gpus=num_gpus,
                ignore_reinit_error=True,
                runtime_env=runtime_env
            )
        
        # Initialize components
        self.job_queue = queue.Queue()
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        
        # Round-robin counter for worker selection
        self.worker_round_robin_counter = 0
        self.worker_round_robin_lock = threading.Lock()
        
        # Create output directories
        # Note: Server always uses server_outputs for intermediate processing
        # Client output_dir parameter is only used for response metadata
        os.makedirs('./server_outputs', exist_ok=True)
        os.makedirs('./client_outputs', exist_ok=True)
        os.makedirs('./test_outputs', exist_ok=True)
        
        # Create worker pools
        self._create_worker_pools()
        
        # Wait for workers to be ready
        self._wait_for_workers_ready()
        
        # Start job queue processor in a separate thread
        self.job_processor_thread = threading.Thread(target=self._process_job_queue, daemon=True)
        self.job_processor_thread.start()
    
    def _create_worker_pools(self):
        """Create unified GPU workers with Ray's built-in GPU scheduling"""
        logger.info(f"Creating {self.num_gpus} unified GPU workers with Ray's built-in GPU scheduling")
        
        import multiprocessing
        
        # Automatically detect total CPU cores and allocate optimally
        total_cpus = multiprocessing.cpu_count()
        cpu_cores_per_worker = max(1, total_cpus // self.num_gpus)  # Distribute evenly
        
        logger.info(f"Detected {total_cpus} total CPU cores, allocating {cpu_cores_per_worker} cores per worker")
        
        # Create workers with Ray's built-in GPU scheduling
        self.gpu_workers = []
        for i in range(self.num_gpus):
            # Use Ray's built-in GPU scheduling without explicit device assignment
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": f"{os.path.join(current_dir, 'SpaTrackerV2')}:{current_dir}:{os.path.join(current_dir, 'wan2.1')}:{os.path.join(current_dir, 'grounded_sam_2')}:{os.path.join(current_dir, 'tapip3d')}:{os.path.join(current_dir, 'tapip3d', 'training')}:{os.path.join(current_dir, 'tapip3d', 'datasets')}",
                    "CONDA_DEFAULT_ENV": "gizmo",
                    "CONDA_PREFIX": "/opt/conda/envs/gizmo",
                    "PATH": "/opt/conda/envs/gizmo/bin:" + os.environ.get('PATH', ''),
                    "LD_LIBRARY_PATH": "/opt/conda/envs/gizmo/lib:" + os.environ.get('LD_LIBRARY_PATH', ''),
                    "RAY_DISABLE_IMPORT_WARNING": "1"
                }
            }
            
            worker = UnifiedGPUWorker.options(
                num_cpus=cpu_cores_per_worker,
                num_gpus=1,
                runtime_env=runtime_env
            ).remote(i, self.tracker, self.model_type)
            self.gpu_workers.append(worker)
        
        logger.info(f"Created {len(self.gpu_workers)} unified GPU workers with Ray's built-in GPU scheduling")
        logger.info("Ray will automatically distribute workers across available GPUs...")
    
    def _wait_for_workers_ready(self):
        """Wait for all workers to finish loading models"""
        logger.info("Waiting for all workers to finish model loading...")
        
        # Wait for models to actually load by checking model loading completion
        max_wait_time = 600  # 10 minutes max wait for model loading
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Try to ping all workers to see if they respond (meaning models are loaded)
                ping_futures = [worker.get_gpu_id.remote() for worker in self.gpu_workers]
                gpu_ids = ray.get(ping_futures, timeout=10)
                
                if len(gpu_ids) == len(self.gpu_workers):
                    logger.info(f"✅ All {len(self.gpu_workers)} GPU workers are ready with models loaded!")
                    return
                    
            except Exception as e:
                logger.info(f"⏳ Workers still loading models... ({int(time.time() - start_time)}s elapsed)")
                time.sleep(10)
        
        logger.warning(f"⚠️  Timeout waiting for workers after {max_wait_time}s. Proceeding anyway...")
    
    def _get_gpu_worker(self):
        """Get a GPU worker using simple round-robin selection"""
        with self.worker_round_robin_lock:
            # Get worker using round-robin
            worker_idx = self.worker_round_robin_counter
            worker = self.gpu_workers[worker_idx]
            
            # Update round-robin counter for next time
            self.worker_round_robin_counter = (worker_idx + 1) % self.num_gpus
            
            logger.info(f"🔍 DEBUG: Selected GPU {worker_idx} for next job (round-robin)")
            return worker
    
    def _process_job_queue(self):
        """Process jobs from the queue asynchronously"""
        logger.info("Job queue processor started")
        while True:
            try:
                # Check for jobs and process them immediately for better concurrency
                try:
                    job = self.job_queue.get_nowait()
                    logger.info(f"🔍 DEBUG: Processing job {job.job_id} immediately")
                    # Process job immediately to allow concurrent execution
                    self._execute_job_async(job)
                except queue.Empty:
                    # No jobs in queue, check for completed jobs
                    self._check_completed_jobs()
                    time.sleep(0.1)  # Short sleep to avoid busy waiting
                    continue
                
            except Exception as e:
                logger.error(f"Error in job processor: {e}")
                time.sleep(1)
    
    def _cleanup_job_files(self, job_id: str):
        """Clean up intermediate files for a completed job"""
        try:
            output_dir = './server_outputs'
            files_to_remove = [
                f'first_frame_{job_id}.png',
                f'last_frame_{job_id}.png',
                f'wan_output_{job_id}.mp4',
                f'tapip3d_output_{job_id}.npz',
                f'segmentation_masks_{job_id}.npz',
                f'segmentation_video_{job_id}.mp4',
                f'segmentation_masks_full_{job_id}.npz',
                f'segmentation_video_full_{job_id}.mp4'
            ]
            
            removed_count = 0
            total_size_freed = 0
            
            for filename in files_to_remove:
                file_path = os.path.join(output_dir, filename)
                if os.path.exists(file_path):
                    try:
                        # Get file size before removal for logging
                        file_size = os.path.getsize(file_path)
                        total_size_freed += file_size
                        
                        # Additional safety check: ensure file is in server_outputs directory
                        if not os.path.abspath(file_path).startswith(os.path.abspath(output_dir)):
                            logger.warning(f"Skipping {filename} - not in server_outputs directory (safety check)")
                            continue
                        
                        os.remove(file_path)
                        removed_count += 1
                        
                        # Convert to human readable size
                        if file_size > 1024 * 1024:  # MB
                            size_str = f"{file_size / (1024 * 1024):.1f} MB"
                        elif file_size > 1024:  # KB
                            size_str = f"{file_size / 1024:.1f} KB"
                        else:
                            size_str = f"{file_size} B"
                        
                        logger.info(f"🧹 Cleaned up: {filename} ({size_str})")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove {filename}: {e}")
            
            if removed_count > 0:
                # Convert total size to human readable
                if total_size_freed > 1024 * 1024:  # MB
                    total_size_str = f"{total_size_freed / (1024 * 1024):.1f} MB"
                elif total_size_freed > 1024:  # KB
                    total_size_str = f"{total_size_freed / 1024:.1f} KB"
                else:
                    total_size_str = f"{total_size_freed} B"
                
                logger.info(f"🧹 Cleaned up {removed_count} intermediate files for job {job_id} (freed {total_size_str})")
            else:
                logger.info(f"🧹 No intermediate files found to clean up for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error during cleanup for job {job_id}: {e}")
            import traceback
            logger.error(f"Cleanup traceback: {traceback.format_exc()}")
    
    def _cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old intermediate files that are older than max_age_hours"""
        try:
            output_dir = './server_outputs'
            if not os.path.exists(output_dir):
                return
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            removed_count = 0
            total_size_freed = 0
            
            # Get all files in server_outputs
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                
                # Check if file is old enough to be cleaned up
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size_freed += file_size
                        
                        os.remove(file_path)
                        removed_count += 1
                        
                        logger.info(f"🧹 Cleaned up old file: {filename} (age: {file_age/3600:.1f}h)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {filename}: {e}")
            
            if removed_count > 0:
                if total_size_freed > 1024 * 1024:  # MB
                    total_size_str = f"{total_size_freed / (1024 * 1024):.1f} MB"
                elif total_size_freed > 1024:  # KB
                    total_size_str = f"{total_size_freed / 1024:.1f} KB"
                else:
                    total_size_str = f"{total_size_freed} B"
                
                logger.info(f"🧹 Cleaned up {removed_count} old files (freed {total_size_str})")
                
        except Exception as e:
            logger.error(f"Error during old file cleanup: {e}")
    
    def _check_completed_jobs(self):
        """Check for completed jobs and update their status"""
        with self.jobs_lock:
            jobs_to_check = [job for job in self.jobs.values() if job.future is not None and job.status == JobStatus.RUNNING]
        
        for job in jobs_to_check:
            try:
                # Check if job is ready (properly non-blocking using ray.wait)
                ready, not_ready = ray.wait([job.future], timeout=0)
                
                if ready:
                    # Job completed, get the result
                    job_id, result = ray.get(ready[0])
                    job.result = result
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    
                    logger.info(f"🎉 Job {job.job_id} completed successfully!")
                    
                    # Log timings if available
                    if result.get('status') == 'success' and 'timings' in result:
                        timings = result['timings']
                        timings_log = " | ".join([f"{k}: {v:.2f}s" for k, v in timings.items()])
                        logger.info(f"📊 Timings for job {job.job_id}: {timings_log}")

                    # Clean up intermediate files if job was successful
                    if result.get('status') == 'success' and self.enable_cleanup:
                        logger.info(f"🧹 Starting cleanup for successful job {job.job_id}")
                        self._cleanup_job_files(job.job_id)
                    else:
                        logger.info(f"⚠️  Job {job.job_id} completed but failed, keeping intermediate files for debugging")
                        
                # If not ready, job is still running (this is normal, do nothing)
                    
            except Exception as e:
                # Job actually failed
                logger.error(f"❌ Job {job.job_id} failed: {e}")
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
    
    def _execute_job(self, job: PipelineJob):
        """Execute a pipeline job asynchronously"""
        try:
            # Update job status
            with self.jobs_lock:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
            
            # Save first frame to file
            first_frame_data = job.request_data.get('first_frame')
            if first_frame_data:
                first_frame_bytes = base64.b64decode(first_frame_data)
                # Server always uses server_outputs for intermediate files
                output_dir = './server_outputs'
                os.makedirs(output_dir, exist_ok=True)
                first_frame_path = os.path.join(output_dir, f'first_frame_{job.job_id}.png')
                with open(first_frame_path, 'wb') as f:
                    f.write(first_frame_bytes)
                logger.info(f"Saved first frame to {first_frame_path}")
            else:
                first_frame_path = None
            
            # Save last frame to file if provided
            last_frame_path = None
            last_frame_data = job.request_data.get('last_frame')
            if last_frame_data:
                last_frame_bytes = base64.b64decode(last_frame_data)
                # Server always uses server_outputs for intermediate files
                output_dir = './server_outputs'
                os.makedirs(output_dir, exist_ok=True)
                last_frame_path = os.path.join(output_dir, f'last_frame_{job.job_id}.png')
                with open(last_frame_path, 'wb') as f:
                    f.write(last_frame_bytes)
                logger.info(f"Saved last frame to {last_frame_path}")
            
            # Get worker using simple round-robin selection
            worker = self._get_gpu_worker()
            
            # Assign job to GPU
            gpu_id = self.gpu_workers.index(worker)
            job.gpu_id = gpu_id
            
            logger.info(f"Assigned job {job.job_id} to GPU {gpu_id}")
            
            # Submit pipeline job asynchronously (don't wait for completion)
            # The worker will handle its own load tracking within process_pipeline
            # Server always uses server_outputs for intermediate processing
            job.future = worker.process_pipeline.remote(
                job.job_id, 
                job.request_data.get('mode', PipelineMode.FULL.value),
                os.path.abspath(first_frame_path) if first_frame_path else "",
                job.request_data.get('wan_prompt', 'a human hand pick up the mug using fingertips and smoothly pour the water into the basket.'),
                os.path.abspath('./server_outputs'),  # Server always uses server_outputs
                os.path.abspath(last_frame_path) if last_frame_path else None,
                job.request_data.get('seed', 42),
                job.request_data.get('sam_prompt', 'mug.'),
                job.request_data.get('input_video_path'),
                job.request_data.get('input_npz_path'),
                job.request_data.get('config_file_content'),
                job.request_data.get('use_veo', False)
            )
            
            logger.info(f"Job {job.job_id} submitted asynchronously to GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            
            # Update job with error
            with self.jobs_lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
    
    def _execute_job_async(self, job: PipelineJob):
        """Execute a pipeline job asynchronously (simplified version for better concurrency)"""
        try:
            # Update job status
            with self.jobs_lock:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
            
            # Save first frame to file
            first_frame_data = job.request_data.get('first_frame')
            if first_frame_data:
                first_frame_bytes = base64.b64decode(first_frame_data)
                # Server always uses server_outputs for intermediate files
                output_dir = './server_outputs'
                os.makedirs(output_dir, exist_ok=True)
                first_frame_path = os.path.join(output_dir, f'first_frame_{job.job_id}.png')
                with open(first_frame_path, 'wb') as f:
                    f.write(first_frame_bytes)
                logger.info(f"Saved first frame to {first_frame_path}")
            else:
                first_frame_path = None
            
            # Save last frame to file if provided
            last_frame_path = None
            last_frame_data = job.request_data.get('last_frame')
            if last_frame_data:
                last_frame_bytes = base64.b64decode(last_frame_data)
                # Server always uses server_outputs for intermediate files
                output_dir = './server_outputs'
                os.makedirs(output_dir, exist_ok=True)
                last_frame_path = os.path.join(output_dir, f'last_frame_{job.job_id}.png')
                with open(last_frame_path, 'wb') as f:
                    f.write(last_frame_bytes)
                logger.info(f"Saved last frame to {last_frame_path}")
            
            # Get worker using simple round-robin selection
            worker = self._get_gpu_worker()
            
            # Assign job to GPU
            gpu_id = self.gpu_workers.index(worker)
            job.gpu_id = gpu_id
            
            logger.info(f"🚀 Assigned job {job.job_id} to GPU {gpu_id} (async)")
            
            # Submit pipeline job asynchronously to Ray (this allows concurrent execution)
            # Server always uses server_outputs for intermediate processing
            job.future = worker.process_pipeline.remote(
                job.job_id, 
                job.request_data.get('mode', PipelineMode.FULL.value),
                os.path.abspath(first_frame_path) if first_frame_path else "",
                job.request_data.get('wan_prompt', 'a human hand pick up the mug using fingertips and smoothly pour the water into the basket.'),
                os.path.abspath('./server_outputs'),  # Server always uses server_outputs
                os.path.abspath(last_frame_path) if last_frame_path else None,
                job.request_data.get('seed', 42),
                job.request_data.get('sam_prompt', 'mug.'),
                job.request_data.get('input_video_path'),
                job.request_data.get('input_npz_path'),
                job.request_data.get('config_file_content'),
                job.request_data.get('use_veo', False)
            )
            
            logger.info(f"🔥 Job {job.job_id} submitted to GPU {gpu_id} for CONCURRENT execution!")
            
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            
            # Update job with error
            with self.jobs_lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a pipeline request"""
        try:
            # Generate unique job ID with datetime and uuid
            import uuid
            job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create job
            job = PipelineJob(job_id=job_id, request_data=request_data)
            
            # Add job to tracking
            with self.jobs_lock:
                self.jobs[job_id] = job
            
            # Add job to queue
            self.job_queue.put(job)
            
            # Log request
            mode = request_data.get('mode', PipelineMode.FULL.value)
            logger.info(f"Received {mode} pipeline request: {request_data.get('wan_prompt', 'No WAN prompt')}")
            logger.info(f"Queued job {job_id}: {request_data.get('wan_prompt', 'No WAN prompt')}")
            
            # Return job ID for tracking
            return {
                'status': 'queued',
                'job_id': job_id,
                'message': f'Job {job_id} queued for {mode} pipeline processing'
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job"""
        try:
            with self.jobs_lock:
                job = self.jobs.get(job_id)
            
            if job is None:
                return {
                    'status': 'error',
                    'error': f'Job {job_id} not found'
                }
            
            response = {
                'job_id': job_id,
                'status': job.status.value,
                'created_at': job.created_at.isoformat(),
                'gpu_id': job.gpu_id
            }
            
            if job.started_at:
                response['started_at'] = job.started_at.isoformat()
            
            if job.completed_at:
                response['completed_at'] = job.completed_at.isoformat()
            
            if job.error:
                response['error'] = job.error
            
            if job.result:
                response['result'] = job.result
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            # Get GPU status from workers
            gpu_status = {}
            for i, worker in enumerate(self.gpu_workers):
                # Simplified status without manual load tracking
                gpu_status[i] = {
                    'load': 'unknown',  # Ray manages load internally
                    'available': True   # Assume available since Ray handles scheduling
                }
            
            with self.jobs_lock:
                job_counts = {
                    'pending': len([j for j in self.jobs.values() if j.status == JobStatus.PENDING]),
                    'running': len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
                    'completed': len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
                    'failed': len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
                }
            
            return {
                'status': 'success',
                'gpu_status': gpu_status,
                'total_jobs': len(self.jobs),
                'queue_size': self.job_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def run(self):
        """Run the server"""
        logger.info(f"Unified Ray pipeline server started on port {self.port}")
        logger.info(f"Using {self.num_gpus} GPUs with Ray's built-in GPU scheduling")
        logger.info("Each GPU worker preloads WAN, Tapip3D, and SAM models for instant processing")
        logger.info("Ray automatically distributes jobs across available GPUs for optimal load balancing")
        logger.info("✅ Server is ready! Waiting for requests...")
        
        while True:
            try:
                # Receive request (synchronous)
                message = self.socket.recv()
                request_data = pickle.loads(message)
                
                # Process request based on type
                request_type = request_data.get('type', 'pipeline')
                
                if request_type == 'pipeline':
                    response = await self.process_request(request_data)
                elif request_type == 'status':
                    job_id = request_data.get('job_id')
                    response = await self.get_job_status(job_id)
                elif request_type == 'system_status':
                    response = await self.get_system_status()
                else:
                    response = {
                        'status': 'error',
                        'error': f'Unknown request type: {request_type}'
                    }
                
                # Send response (synchronous)
                response_data = pickle.dumps(response)
                self.socket.send(response_data)
                
            except Exception as e:
                logger.error(f"Error handling request: {str(e)}")
                error_response = {
                    'status': 'error',
                    'error': str(e)
                }
                self.socket.send(pickle.dumps(error_response))

def main():
    """Main function
    
    Server Configuration:
    - Server always saves intermediate outputs to ./server_outputs
    - Client output_dir parameter is only used for response metadata
    - Final results are returned to client as base64-encoded data
    - Intermediate files are cleaned up after successful job completion
    """
    parser = argparse.ArgumentParser(description='Unified Ray Pipeline Server')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--no-cleanup', action='store_true', 
                       help='Disable automatic cleanup of intermediate files (useful for debugging)')
    parser.add_argument('--tracker', type=str, default='tapip3d', choices=['tapip3d', 'SpaTrackerV2'],
                        help='Choose the tracker model to use')
    parser.add_argument('--model', type=str, default='wan', choices=['wan', 'veo'],
                        help='Choose the video generation model to use')
    
    args = parser.parse_args()
    
    # Create and run server
    enable_cleanup = not args.no_cleanup
    server = UnifiedRayPipelineServer(port=args.port, num_gpus=args.num_gpus, enable_cleanup=enable_cleanup, tracker=args.tracker, model_type=args.model)
    
    if enable_cleanup:
        logger.info("🧹 Automatic cleanup enabled - intermediate files will be removed after successful jobs")
    else:
        logger.info("⚠️  Automatic cleanup disabled - intermediate files will be preserved for debugging")
    
    # Run the server
    asyncio.run(server.run())

if __name__ == "__main__":
    main() 
