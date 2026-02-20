#!/usr/bin/env python3
"""
Client for the Advanced Ray Pipeline Server
Supports job queuing and status checking
"""

import sys
import os
import pickle
import base64
import argparse
import time
import zmq
from typing import Dict, Any, Optional
import json
from prompt_extension import generate
import wandb
import cv2
import numpy as np

# Import flow visualization
try:
    from visualize_flow import visualize_flow
    FLOW_VISUALIZATION_AVAILABLE = True
    print("Flow visualization module found")
except ImportError as e:
    FLOW_VISUALIZATION_AVAILABLE = False
    print(f"Warning: Flow visualization module not found: {e}")

def _save_wandb_run_id_file(output_dir: str, run_id: Optional[str]):
    """Save the W&B run id into output_dir/wandb_run_id.txt if available."""
    if not run_id:
        return
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        run_id_file = os.path.join(output_dir, "wandb_run_id.txt")
        with open(run_id_file, "w") as f:
            f.write(run_id)
        print(f"Saved W&B run ID to: {run_id_file}")
        print(f"Run ID: {run_id}")
        print(f"To resume this run in the future, use: wandb.init(project='gizmo', id='{run_id}', resume='must')")
    except Exception as e:
        print(f"Warning: Failed to save W&B run ID: {e}")

def modify_wandb_run(run, **kwargs):
    """
    Helper function to modify W&B run attributes after initialization.
    
    Args:
        run: W&B run object
        **kwargs: Attributes to modify (name, tags, notes, etc.)
    """
    if run is None:
        return
    
    try:
        for attr, value in kwargs.items():
            if hasattr(run, attr):
                setattr(run, attr, value)
                print(f"Modified W&B run {attr}: {value}")
            else:
                print(f"Warning: W&B run has no attribute '{attr}'")
    except Exception as e:
        print(f"Error modifying W&B run attributes: {e}")

def resume_wandb_run_from_file(run_id_file: str, project: str = "gizmo", entity: str = None):
    """
    Resume a W&B run from a saved run ID file.
    
    Args:
        run_id_file: Path to the file containing the run ID
        project: W&B project name
        entity: W&B entity (username/team)
        
    Returns:
        W&B run object or None if failed
    """
    try:
        # Read run ID from file
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        
        print(f"Resuming W&B run with ID: {run_id}")
        
        # Resume the run
        run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="must"
        )
        
        print(f"Successfully resumed W&B run: {run.name}")
        return run
        
    except FileNotFoundError:
        print(f"Run ID file not found: {run_id_file}")
        return None
    except Exception as e:
        print(f"Failed to resume W&B run: {e}")
        return None

class RayPipelineClient:
    """Client for the Advanced Ray Pipeline Server"""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 5555):
        self.server_host = server_host
        self.server_port = server_port
        
        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_host}:{server_port}")
        
    def _send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the server and get response"""
        try:
            # Send request
            message = pickle.dumps(request_data)
            self.socket.send(message)
            
            # Receive response
            response = self.socket.recv()
            return pickle.loads(response)
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Communication error: {str(e)}'
            }
    
    def submit_pipeline_job(self, first_frame_path: str, wan_prompt: str, 
                           sam_prompt: str = "mug.", last_frame_path: Optional[str] = None,
                           output_dir: str = "./client_outputs", seed: int = 42,
                           config_file: Optional[str] = None, use_veo: bool = False,
                           input_video_path: Optional[str] = None, mode: str = 'full') -> Dict[str, Any]:
        """Submit a pipeline job to the server
        
        Note: output_dir is only used for client-side final result storage.
        Server always uses its own server_outputs directory for intermediate processing.
        """
        
        # Read and encode first frame
        with open(first_frame_path, 'rb') as f:
            first_frame_bytes = f.read()
        first_frame_data = base64.b64encode(first_frame_bytes).decode('utf-8')
        
        # Read and encode last frame if provided
        last_frame_data = None
        if last_frame_path is not None:
            with open(last_frame_path, 'rb') as f:
                last_frame_bytes = f.read()
            last_frame_data = base64.b64encode(last_frame_bytes).decode('utf-8')
        
        # Read config file if provided
        config_file_content = None
        if config_file:
            with open(config_file, 'r') as f:
                config_file_content = f.read()

        # Also check for automatic config file in sequence directory
        if config_file_content is None:
            sequence_dir = os.path.dirname(first_frame_path)
            auto_config_path = os.path.join(sequence_dir, "config.json")
            if os.path.exists(auto_config_path):
                with open(auto_config_path, 'r') as f:
                    config_file_content = f.read()
                print(f"Loaded config file content from: {auto_config_path}")

        # Prepare request
        # Note: output_dir is passed to server but only used for response metadata
        # Server always uses server_outputs for actual file processing
        request_data = {
            'type': 'pipeline',
            'wan_prompt': wan_prompt,
            'sam_prompt': sam_prompt,
            'output_dir': output_dir,  # Used for response metadata only
            'seed': seed,
            'use_veo': use_veo,
            'first_frame': first_frame_data,
            'mode': mode,
        }

        if last_frame_data is not None:
            request_data['last_frame'] = last_frame_data

        if config_file_content is not None:
            request_data['config_file_content'] = config_file_content

        if input_video_path is not None:
            request_data['input_video_path'] = input_video_path
        
        # Send request
        return self._send_request(request_data)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job"""
        request_data = {
            'type': 'status',
            'job_id': job_id
        }
        return self._send_request(request_data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        request_data = {
            'type': 'system_status'
        }
        return self._send_request(request_data)
    
    def wait_for_job_completion(self, job_id: str, poll_interval: float = 5.0, 
                               timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a job to complete and return the result"""
        start_time = time.time()
        
        while True:
            # Check if timeout exceeded
            if timeout is not None and (time.time() - start_time) > timeout:
                return {
                    'status': 'error',
                    'error': f'Timeout waiting for job {job_id} to complete'
                }
            
            # Get job status
            status_response = self.get_job_status(job_id)
            
            if status_response['status'] == 'error':
                return status_response
            
            # Check if job is completed
            if 'status' in status_response:
                job_status = status_response['status']
                if job_status == 'completed':
                    # Job is completed, return the result from the job
                    if 'result' in status_response:
                        return status_response['result']
                    else:
                        return {
                            'status': 'error',
                            'error': f'Job {job_id} completed but no result found'
                        }
                elif job_status == 'failed':
                    return {
                        'status': 'error',
                        'error': status_response.get('error', f'Job {job_id} failed')
                    }
                else:
                    # Job is still pending or running
                    print(f"Job {job_id} status: {job_status}")
                    time.sleep(poll_interval)
            else:
                # Unexpected response format
                return {
                    'status': 'error',
                    'error': f'Unexpected response format: {status_response}'
                }
    
    def run_pipeline_and_wait(self, first_frame_path: str, wan_prompt: str,
                             sam_prompt: str = "mug.", last_frame_path: Optional[str] = None,
                             output_dir: str = "./client_outputs", seed: int = 42,
                             config_file: Optional[str] = None,
                             poll_interval: float = 5.0, timeout: Optional[float] = None,
                             use_veo: bool = False, input_video_path: Optional[str] = None,
                             mode: str = 'full') -> Dict[str, Any]:
        """Submit a pipeline job and wait for completion
        
        Note: output_dir is only used for client-side final result storage.
        Server always uses its own server_outputs directory for intermediate processing.
        """
        
        # Submit job
        print(f"Submitting pipeline job...")
        submit_response = self.submit_pipeline_job(
            first_frame_path, wan_prompt, sam_prompt, last_frame_path, output_dir, seed, config_file,
            use_veo, input_video_path, mode
        )
        
        if submit_response['status'] != 'queued':
            return submit_response
        
        job_id = submit_response['job_id']
        print(f"Job submitted successfully with ID: {job_id}")
        
        # Wait for completion
        print(f"Waiting for job completion...")
        return self.wait_for_job_completion(job_id, poll_interval, timeout)
    
    def save_results(self, result: Dict[str, Any], output_dir: str = "./client_outputs", first_frame_path: Optional[str] = None, last_frame_path: Optional[str] = None):
        """Save the results from a completed job to client-side directory
        
        This method saves the final results (WAN video, Tapip3D results, SAM segmentation)
        that were returned from the server to the client's local output directory.
        The server's intermediate files remain in the server's server_outputs directory.
        
        Args:
            result: Dictionary containing the pipeline results
            output_dir: Directory to save results to (must already exist)
            first_frame_path: Path to the first frame image
            last_frame_path: Path to the last frame image
        """
        # Ensure output directory exists for saving artifacts
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create output_dir '{output_dir}': {e}")
        
        if result['status'] != 'success' or 'outputs' not in result:
            print(f"Cannot save results: {result.get('error', 'Unknown error')}")
            return
        
        # Note: W&B run ID is now saved immediately after initialization; do not save it here
        
        outputs = result['outputs']
        
        # Debug: Show what outputs are available
        print(f"📋 Available outputs: {list(outputs.keys())}")
        
        # Copy depth image of first frame if available
        if first_frame_path is not None:
            depth_path = first_frame_path.replace('rgb', 'depth')
            if os.path.exists(depth_path):
                import shutil
                depth_filename = os.path.basename(depth_path)
                output_depth_path = os.path.join(output_dir, "start_depth.png")
                shutil.copy2(depth_path, output_depth_path)
                print(f"Copied depth image: {output_depth_path}")
            else:
                print(f"⚠️  Depth image not found at: {depth_path}")
            
            # Copy first frame RGB image as start.png
            if os.path.exists(first_frame_path):
                import shutil
                output_start_path = os.path.join(output_dir, "start.png")
                shutil.copy2(first_frame_path, output_start_path)
                print(f"Copied start frame: {output_start_path}")
            else:
                print(f"⚠️  Start frame not found at: {first_frame_path}")
            
            # Copy transform config file of first frame if available
            config_path = first_frame_path.replace('rgb', 'transform').replace('.png', '.json')
            if os.path.exists(config_path):
                import shutil
                output_config_path = os.path.join(output_dir, "config.json")
                shutil.copy2(config_path, output_config_path)
                print(f"Copied config file: {output_config_path}")
            else:
                print(f"⚠️  Config file not found at: {config_path}")
        
        # Copy depth image of last frame if available
        if last_frame_path is not None:
            depth_path = last_frame_path.replace('rgb', 'depth')
            if os.path.exists(depth_path):
                import shutil
                output_depth_path = os.path.join(output_dir, "end_depth.png")
                shutil.copy2(depth_path, output_depth_path)
                print(f"Copied last frame depth image: {output_depth_path}")
            else:
                print(f"⚠️  Last frame depth image not found at: {depth_path}")
            
            # Copy last frame RGB image as end.png
            if os.path.exists(last_frame_path):
                import shutil
                output_end_path = os.path.join(output_dir, "end.png")
                shutil.copy2(last_frame_path, output_end_path)
                print(f"Copied end frame: {output_end_path}")
            else:
                print(f"⚠️  End frame not found at: {last_frame_path}")
        
        # Save WAN video
        if 'wan_video' in outputs:
            wan_video_data = base64.b64decode(outputs['wan_video'])
            wan_video_path = os.path.join(output_dir, outputs['wan_video_filename'])
            with open(wan_video_path, 'wb') as f:
                f.write(wan_video_data)
            print(f"Saved WAN video: {wan_video_path}")
        
        # Save Tapip3D results
        if 'tapip3d_results' in outputs:
            tapip3d_data = base64.b64decode(outputs['tapip3d_results'])
            tapip3d_path = os.path.join(output_dir, outputs['tapip3d_results_filename'])
            with open(tapip3d_path, 'wb') as f:
                f.write(tapip3d_data)
            print(f"Saved Tapip3D results: {tapip3d_path}")

        # Save SpaTrackerV2 results
        if 'spatracker_results' in outputs:
            spatracker_data = base64.b64decode(outputs['spatracker_results'])
            spatracker_path = os.path.join(output_dir, outputs['spatracker_results_filename'])
            with open(spatracker_path, 'wb') as f:
                f.write(spatracker_data)
            print(f"Saved SpaTrackerV2 results: {spatracker_path}")
        
        # Save SAM segmentation results
        if 'sam_segmentation' in outputs:
            sam_data = base64.b64decode(outputs['sam_segmentation'])
            sam_path = os.path.join(output_dir, outputs['sam_segmentation_filename'])
            with open(sam_path, 'wb') as f:
                f.write(sam_data)
            print(f"Saved SAM segmentation masks: {sam_path}")
        
        # Save SAM segmentation video
        if 'sam_segmentation_video' in outputs:
            sam_video_data = base64.b64decode(outputs['sam_segmentation_video'])
            sam_video_path = os.path.join(output_dir, outputs['sam_segmentation_video_filename'])
            with open(sam_video_path, 'wb') as f:
                f.write(sam_video_data)
            print(f"Saved SAM segmentation video: {sam_video_path}")
        else:
            print("⚠️  SAM segmentation video not found in response")

        # Save SAM segmentation full results
        if 'sam_segmentation_full' in outputs:
            sam_data = base64.b64decode(outputs['sam_segmentation_full'])
            sam_path = os.path.join(output_dir, outputs['sam_segmentation_full_filename'])
            with open(sam_path, 'wb') as f:
                f.write(sam_data)
            print(f"Saved SAM segmentation full masks: {sam_path}")

        # Save SAM segmentation full video
        if 'sam_segmentation_video_full' in outputs:
            sam_video_data = base64.b64decode(outputs['sam_segmentation_video_full'])
            sam_video_path = os.path.join(output_dir, outputs['sam_segmentation_video_full_filename'])
            with open(sam_video_path, 'wb') as f:
                f.write(sam_video_data)
            print(f"Saved SAM segmentation full video: {sam_video_path}")
        else:
            print("⚠️  SAM segmentation full video not found in response")
        
        # Generate flow visualization if Tapip3D results are available
        flow_visualization_path = None
        flow_visualization_object_path = None
        flow_visualization_all_path = None
        if 'tapip3d_results' in outputs and FLOW_VISUALIZATION_AVAILABLE:
            try:
                print("Generating flow visualization from Tapip3D results...")
                
                # Save Tapip3D results first (if not already saved)
                if 'tapip3d_results_filename' in outputs:
                    tapip3d_path = os.path.join(output_dir, outputs['tapip3d_results_filename'])
                    
                    # Generate flow visualization
                    flow_visualization_path = visualize_flow(
                        tapip3d_output_path=tapip3d_path,
                        output_dir=output_dir,
                        width=724,  # Default processing width
                        height=543,  # Default processing height
                        end_frame=-1,  # Use all frames
                        start_image_path=os.path.join(output_dir, "start.png") if os.path.exists(os.path.join(output_dir, "start.png")) else None
                    )

                    # Derive both expected output paths based on tapip3d filename
                    base_name = os.path.basename(tapip3d_path)
                    name_part = base_name.replace('tapip3d_output_', '').replace('.npz', '')
                    flow_visualization_object_path = os.path.join(output_dir, f"flow_visualization_{name_part}.png")
                    flow_visualization_all_path = os.path.join(output_dir, f"flow_visualization_all_{name_part}.png")
                    
                    if flow_visualization_path:
                        print(f"✅ Flow visualization generated: {flow_visualization_path}")
                    else:
                        print("⚠️  Flow visualization generation failed")
                        
            except Exception as e:
                print(f"⚠️  Error generating flow visualization: {e}")
        elif not FLOW_VISUALIZATION_AVAILABLE:
            print("⚠️  Flow visualization not available (module not found)")
        
        print(f"All results saved to {output_dir}")
        
        # Return additional information for W&B logging
        return {
            'flow_visualization_path': flow_visualization_path,
            'flow_visualization_object_path': flow_visualization_object_path,
            'flow_visualization_all_path': flow_visualization_all_path,
        }

def main():
    parser = argparse.ArgumentParser(description='Ray Pipeline Client')
    parser.add_argument('--server-host', default='localhost', help='Server host')
    parser.add_argument('--server-port', type=int, default=5555, help='Server port')
    parser.add_argument('--first-frame', help='Path to first frame image')
    parser.add_argument('--last-frame', help='Path to last frame image (optional)')
    parser.add_argument('--wan-prompt', default='a human hand pick up the mug using fingertips and smoothly pour the water into the basket.', 
                       help='WAN prompt for video generation')
    parser.add_argument('--sam-prompt', default='mug.', help='SAM prompt for segmentation')
    parser.add_argument('--output-dir', default='./client_outputs', 
                       help='Client-side output directory for final results (server uses server_outputs for intermediate processing)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config-file', help='Path to camera intrinsics config file (optional)')
    parser.add_argument('--poll-interval', type=float, default=10.0, help='Poll interval for status checking')
    parser.add_argument('--timeout', type=float, help='Timeout in seconds (optional)')
    parser.add_argument('--job-id', help='Check status of specific job ID')
    parser.add_argument('--system-status', action='store_true', help='Get system status')
    parser.add_argument('--prompt-extension', action='store_true', help='Use prompt extension')
    parser.add_argument('--task_name', default=None, help='Task name to use as W&B group for runs')
    parser.add_argument('--run', default=None, help='Run identifier to log to W&B and tag as {task_name}_{run}')
    parser.add_argument('--resume-run', help='Path to wandb_run_id.txt file to resume an existing run')
    parser.add_argument('--veo-prompt', help='Custom prompt for Veo video generation (overrides --wan-prompt when --use-veo is enabled)')
    parser.add_argument('--use-veo', action='store_true', help='Use Google Veo for video generation')
    parser.add_argument('--mode', default='full', help='Pipeline mode')
    parser.add_argument('--input-video', help='Path to input video for tapip3d_only mode')
    
    args = parser.parse_args()

    if args.use_veo:
        # ignore prompt extension if using veo since it will be handled by the Veo API
        args.prompt_extension = False
        args.wan_prompt = args.veo_prompt
    
    # Create client
    client = RayPipelineClient(args.server_host, args.server_port)
    
    if args.system_status:
        # Get system status
        print("Getting system status...")
        status = client.get_system_status()
        print(json.dumps(status, indent=2))
        
    elif args.job_id:
        # Check specific job status
        print(f"Checking status of job {args.job_id}...")
        status = client.get_job_status(args.job_id)
        print(json.dumps(status, indent=2))
        
    else:
        wan_prompt = args.wan_prompt
        original_wan_prompt = args.wan_prompt
        if args.prompt_extension:
            print("Performing prompt extension...")
            if not args.first_frame:
                print("Error: --first-frame is required for prompt extension.")
                sys.exit(1)

            try:
                print(f"Original prompt: {wan_prompt}")
                extended_prompt = generate(
                    image1_path=args.first_frame,
                    prompt=wan_prompt,
                    image2_path=args.last_frame,
                ).strip()

                if extended_prompt and extended_prompt != wan_prompt:
                    wan_prompt = extended_prompt
                    print(f"Extended prompt: {wan_prompt}")
                else:
                    print("Prompt extension returned an empty or identical result. Using original prompt.")
            except Exception as e:
                print(f"An error occurred during prompt extension: {e}")
                print("Proceeding with the original prompt.")

        # Initialize Weights & Biases
        run = None
        combined_task_run_tag = f"{args.task_name}_{args.run}" if (args.task_name and args.run) else None
        
        # Check if we should resume an existing run
        if args.resume_run:
            print(f"Attempting to resume W&B run from: {args.resume_run}")
            run = resume_wandb_run_from_file(args.resume_run, project="gizmo")
            # If resume succeeded, ensure the id file matches the resumed id
            try:
                _save_wandb_run_id_file(args.output_dir, run.id if run is not None else None)
            except Exception:
                pass
            if run is None:
                print("Failed to resume run, proceeding with new run initialization")
                args.resume_run = None  # Reset to create new run
            else:
                # Update resumed run with provided identifiers and tag
                try:
                    update_payload = {}
                    if args.run is not None:
                        update_payload["run"] = args.run
                    if args.task_name is not None:
                        update_payload["task_name"] = args.task_name
                    if update_payload:
                        run.config.update(update_payload, allow_val_change=True)
                except Exception:
                    pass
                try:
                    if combined_task_run_tag:
                        existing_tags = list(getattr(run, "tags", []))
                        new_tags = sorted(set(existing_tags + [combined_task_run_tag]))
                        modify_wandb_run(run, tags=new_tags)
                except Exception:
                    pass
                # Log identifiers explicitly as metrics for traceability
                try:
                    wandb.log({
                        "task_name": args.task_name,
                        "run": args.run,
                    })
                except Exception:
                    pass
        
        # Initialize new run if not resuming
        if run is None:
            try:
                # Generate a deterministic run id so we can save it immediately and resume reliably
                generated_run_id = wandb.util.generate_id()
                # Save run id before init (we assume output_dir exists per contract)
                try:
                    _save_wandb_run_id_file(args.output_dir, generated_run_id)
                except Exception:
                    pass
                run = wandb.init(project="gizmo", group=args.task_name, id=generated_run_id, resume="allow", config={
                    "server_host": args.server_host,
                    "server_port": args.server_port,
                    "seed": args.seed,
                    "output_dir": args.output_dir,
                    "run": args.run,
                    "task_name": args.task_name,
                })
                
                # Modify run attributes after initialization
                if run is not None:
                    # Use helper function to modify run attributes
                    base_tags = []
                    if combined_task_run_tag:
                        base_tags.append(combined_task_run_tag)
                    modify_wandb_run(run,
                        name=f"{args.task_name}_seed_{args.seed}",
                        tags=base_tags,
                        notes=f"Pipeline run with seed {args.seed} and task {args.task_name}"
                    )
                    # Log identifiers explicitly as metrics for traceability
                    try:
                        wandb.log({
                            "task_name": args.task_name,
                            "run": args.run,
                        })
                    except Exception:
                        pass
                    
            except Exception as e:
                print(f"W&B init failed, attempting offline mode: {e}")
                try:
                    # Reuse the previously generated run id for offline as well
                    run = wandb.init(project="gizmo", group=args.task_name, mode="offline", id=generated_run_id, resume="allow", config={
                        "server_host": args.server_host,
                        "server_port": args.server_port,
                        "seed": args.seed,
                        "output_dir": args.output_dir,
                        "run": args.run,
                        "task_name": args.task_name,
                    })
                    
                    # Modify run attributes for offline mode too
                    if run is not None:
                        base_tags = ["offline"]
                        if combined_task_run_tag:
                            base_tags.append(combined_task_run_tag)
                        modify_wandb_run(run,
                            name=f"{args.task_name}_seed_{args.seed}_offline",
                            tags=base_tags,
                            notes=f"Offline pipeline run with seed {args.seed} and task {args.task_name}"
                        )
                        # Log identifiers explicitly as metrics for traceability
                        try:
                            wandb.log({
                                "task_name": args.task_name,
                                "run": args.run,
                            })
                        except Exception:
                            pass
                        
                except Exception as e2:
                    print(f"W&B offline init also failed, proceeding without W&B logging: {e2}")
                    run = None

        # Log prompts
        if run is not None:
            try:
                wandb.log({
                    "wan_prompt_original": original_wan_prompt,
                    "wan_prompt_extended": wan_prompt,
                    "sam_prompt": args.sam_prompt,
                })
            except Exception as e:
                print(f"W&B prompt logging failed: {e}")

            # Log input frames (RGB and depth if available)
            inputs_to_log = {}
            input_files_for_artifact = []

            def _colorize_depth_image(depth_file_path: str, vmin: Optional[float] = None, vmax: Optional[float] = None):
                try:
                    depth_raw = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
                    if depth_raw is None:
                        return None
                    depth = depth_raw.astype(np.float32)
                    valid = depth > 0
                    if not np.any(valid):
                        return None
                    # Determine normalization bounds
                    if vmin is None or vmax is None:
                        # Robust normalization using percentiles from the current depth if bounds not provided
                        vmin_local = float(np.percentile(depth[valid], 5.0))
                        vmax_local = float(np.percentile(depth[valid], 95.0))
                        if vmax_local <= vmin_local:
                            vmin_local = float(np.min(depth[valid]))
                            vmax_local = float(np.max(depth[valid]))
                            if vmax_local <= vmin_local:
                                vmax_local = vmin_local + 1.0
                    else:
                        vmin_local = float(vmin)
                        vmax_local = float(vmax)
                        if vmax_local <= vmin_local:
                            # Fallback to ensure a positive range
                            vmax_local = vmin_local + 1.0

                    depth_norm = np.clip((depth - vmin_local) / (vmax_local - vmin_local), 0.0, 1.0)
                    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
                    depth_color_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                    depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
                    return depth_color_rgb
                except Exception:
                    return None
            if args.first_frame:
                try:
                    inputs_to_log["first_rgb"] = wandb.Image(args.first_frame, caption="first_rgb")
                    if os.path.exists(args.first_frame):
                        input_files_for_artifact.append((args.first_frame, "first_rgb"))
                    depth_path = args.first_frame.replace('rgb', 'depth')
                    if os.path.exists(depth_path):
                        depth_color = _colorize_depth_image(depth_path)
                        if depth_color is not None:
                            inputs_to_log["first_depth_colormap"] = wandb.Image(depth_color, caption="first_depth_colormap")
                        input_files_for_artifact.append((depth_path, "first_depth"))
                    # Also consider transform config inferred from first frame
                    inferred_config = args.first_frame.replace('rgb', 'transform').replace('.png', '.json')
                    if os.path.exists(inferred_config):
                        input_files_for_artifact.append((inferred_config, "inferred_transform_config"))
                except Exception as e:
                    print(f"W&B first frame logging failed: {e}")

            if args.last_frame:
                try:
                    inputs_to_log["last_rgb"] = wandb.Image(args.last_frame, caption="last_rgb")
                    if os.path.exists(args.last_frame):
                        input_files_for_artifact.append((args.last_frame, "last_rgb"))
                    depth_path = args.last_frame.replace('rgb', 'depth')
                    if os.path.exists(depth_path):
                        depth_color = _colorize_depth_image(depth_path)
                        if depth_color is not None:
                            inputs_to_log["last_depth_colormap"] = wandb.Image(depth_color, caption="last_depth_colormap")
                        input_files_for_artifact.append((depth_path, "last_depth"))
                except Exception as e:
                    print(f"W&B last frame logging failed: {e}")

            if inputs_to_log:
                try:
                    wandb.log(inputs_to_log)
                except Exception as e:
                    print(f"W&B input frames logging failed: {e}")

            # Add explicitly provided config file
            if args.config_file and os.path.exists(args.config_file):
                input_files_for_artifact.append((args.config_file, "config_file"))

            # Log all input files as a single artifact for reproducibility
            try:
                if input_files_for_artifact:
                    inputs_artifact = wandb.Artifact(
                        name=f"inputs_{int(time.time())}",
                        type="inputs",
                        metadata={
                            "wan_prompt_original": original_wan_prompt,
                            "wan_prompt_extended": wan_prompt,
                            "sam_prompt": args.sam_prompt,
                        }
                    )
                    for path, alias in input_files_for_artifact:
                        if os.path.exists(path):
                            inputs_artifact.add_file(path, name=os.path.basename(path))
                    run.log_artifact(inputs_artifact)
            except Exception as e:
                print(f"W&B inputs artifact logging failed: {e}")

        # Run pipeline
        print("Running pipeline...")
        result = client.run_pipeline_and_wait(
            args.first_frame,
            wan_prompt,
            args.sam_prompt,
            args.last_frame,
            args.output_dir,
            args.seed,
            args.config_file,
            args.poll_interval,
            args.timeout,
            args.use_veo,
            args.input_video,
            args.mode,
        )
        
        if result['status'] == 'success':
            print("Pipeline completed successfully!")
            save_info = client.save_results(result, args.output_dir, args.first_frame, args.last_frame)

            # Modify W&B run after successful completion
            if run is not None:
                try:
                    existing_tags = list(getattr(run, "tags", []))
                except Exception:
                    existing_tags = []
                new_tags = existing_tags + ["completed"]
                if combined_task_run_tag:
                    new_tags.append(combined_task_run_tag)
                # Deduplicate while preserving order
                seen = set()
                deduped_tags = []
                for t in new_tags:
                    if t not in seen:
                        deduped_tags.append(t)
                        seen.add(t)
                modify_wandb_run(run,
                    notes=f"Pipeline completed successfully with seed {args.seed}",
                    tags=deduped_tags
                )

            # Log returned outputs to W&B
            if run is not None:
                try:
                    outputs = result.get('outputs', {})
                    # WAN video
                    if 'wan_video_filename' in outputs:
                        wan_video_path = os.path.join(args.output_dir, outputs['wan_video_filename'])
                        if os.path.exists(wan_video_path):
                            wandb.log({"wan_video": wandb.Video(wan_video_path)})
                    # SAM segmentation video
                    if 'sam_segmentation_video_filename' in outputs:
                        sam_video_path = os.path.join(args.output_dir, outputs['sam_segmentation_video_filename'])
                        if os.path.exists(sam_video_path):
                            wandb.log({"sam_segmentation_video": wandb.Video(sam_video_path)})
                    # SAM segmentation full video
                    if 'sam_segmentation_video_full_filename' in outputs:
                        sam_video_path = os.path.join(args.output_dir, outputs['sam_segmentation_video_full_filename'])
                        if os.path.exists(sam_video_path):
                            wandb.log({"sam_segmentation_video_full": wandb.Video(sam_video_path)})
                    # Tapip3D result file
                    if 'tapip3d_results_filename' in outputs:
                        tapip3d_path = os.path.join(args.output_dir, outputs['tapip3d_results_filename'])
                        if os.path.exists(tapip3d_path):
                            artifact = wandb.Artifact(
                                name=f"tapip3d_result_{int(time.time())}",
                                type="tapip3d_result"
                            )
                            artifact.add_file(tapip3d_path)
                            run.log_artifact(artifact)
                    # SAM segmentation result file
                    if 'sam_segmentation_filename' in outputs:
                        sam_path = os.path.join(args.output_dir, outputs['sam_segmentation_filename'])
                        if os.path.exists(sam_path):
                            artifact = wandb.Artifact(
                                name=f"sam_segmentation_result_{int(time.time())}",
                                type="sam_segmentation_result"
                            )
                            artifact.add_file(sam_path)
                            run.log_artifact(artifact)
                    
                    # Flow visualization images (object-only and all-points)
                    if save_info:
                        # Object-only flow
                        flow_viz_object = save_info.get('flow_visualization_object_path')
                        if flow_viz_object and os.path.exists(flow_viz_object):
                            wandb.log({"flow_visualization_object": wandb.Image(flow_viz_object)})
                            print(f"✅ Logged object flow visualization to W&B: {flow_viz_object}")
                        elif flow_viz_object:
                            print(f"⚠️  Object flow visualization file not found: {flow_viz_object}")

                        # All-points flow
                        flow_viz_all = save_info.get('flow_visualization_all_path')
                        if flow_viz_all and os.path.exists(flow_viz_all):
                            wandb.log({"flow_visualization_all": wandb.Image(flow_viz_all)})
                            print(f"✅ Logged all-points flow visualization to W&B: {flow_viz_all}")
                        elif flow_viz_all:
                            print(f"⚠️  All-points flow visualization file not found: {flow_viz_all}")
                except Exception as e:
                    print(f"W&B output logging failed: {e}")
                finally:
                    try:
                        wandb.finish()
                    except Exception:
                        pass
        else:
            print(f"Pipeline failed: {result.get('error', 'Unknown error')}")
            if run is not None:
                try:
                    wandb.log({"status": "failed", "error": result.get('error', 'Unknown error')})
                    wandb.finish()
                except Exception:
                    pass

if __name__ == "__main__":
    main() 