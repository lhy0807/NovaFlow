#!/usr/bin/env python3
"""
Job submission script using ray_pipeline_client.py directly
Runs multiple concurrent requests with different seeds using ray_pipeline_client.py
"""

import subprocess
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import argparse
from get_cmds import get_cmd

class RunJobSubmitter:
    def __init__(self):
        self.results = []
        
    def run_single_job(self, seed: int, display_output: bool = False, use_veo: bool = False) -> Dict[str, Any]:
        """Run a single job with the given seed using ray_pipeline_client.py directly"""
        start_time = time.time()

        # Prepare command to run ray_pipeline_client.py with the same arguments as run.sh
        cmd = get_cmd(seed, "drawer_open", 99, use_veo=use_veo)
        
        print(f"🚀 Starting job with seed {seed} using ray_pipeline_client.py...")
        
        try:
            # Run the job using ray_pipeline_client.py directly
            result = subprocess.run(
                cmd,
                capture_output=not display_output,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ Job with seed {seed} completed successfully in {duration:.2f}s")
                return {
                    'seed': seed,
                    'status': 'success',
                    'duration': duration,
                    'output_dir': f"./demo_data/{seed}",
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"❌ Job with seed {seed} failed after {duration:.2f}s")
                return {
                    'seed': seed,
                    'status': 'failed',
                    'duration': duration,
                    'output_dir': f"./demo_data/{seed}",
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Job with seed {seed} timed out after 3600s")
            return {
                'seed': seed,
                'status': 'timeout',
                'duration': 3600,
                'output_dir': f"./demo_data/{seed}",
                'error': 'Timeout after 3600 seconds'
            }
        except Exception as e:
            print(f"💥 Job with seed {seed} crashed: {e}")
            return {
                'seed': seed,
                'status': 'crashed',
                'duration': time.time() - start_time,
                'output_dir': f"./demo_data/{seed}",
                'error': str(e)
            }
    
    def submit_jobs(self, num_jobs: int = 8, base_seed: int = 42, max_workers: Optional[int] = None, display_output: bool = False, use_veo: bool = False) -> List[Dict[str, Any]]:
        """Submit multiple jobs with different seeds"""
        if max_workers is None:
            max_workers = num_jobs

        print(f"🔥 Starting job submission with {num_jobs} jobs...")
        print(f"🌱 Base seed: {base_seed}")
        print(f"📁 Output directory: ./demo_data/<seed>")
        print(f"⚡ Max concurrent workers: {max_workers}")
        print(f"🎬 Use Veo: {use_veo}")
        print("🚀 Submitting all jobs immediately for concurrent execution")
        print("=" * 60)

        start_time = time.time()

        # Create seeds for all jobs
        seeds = [base_seed + i for i in range(num_jobs)]

        # Submit all jobs immediately for concurrent execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks immediately
            future_to_seed = {}
            for i, seed in enumerate(seeds):
                future = executor.submit(self.run_single_job, seed, display_output, use_veo)
                future_to_seed[future] = seed
                print(f"🚀 Submitted job {i+1}/{num_jobs} (seed {seed}) immediately")
            
            # Collect results as they complete
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    print(f"💥 Exception for seed {seed}: {e}")
                    self.results.append({
                        'seed': seed,
                        'status': 'exception',
                        'error': str(e)
                    })
        
        total_duration = time.time() - start_time
        
        # Print summary
        self.print_summary(total_duration)
        
        return self.results
    
    def print_summary(self, total_duration: float):
        """Print job submission summary"""
        print("\n" + "=" * 60)
        print("📊 JOB SUBMISSION SUMMARY")
        print("=" * 60)
        
        # Count results by status
        status_counts = {}
        successful_durations = []
        
        for result in self.results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status == 'success' and 'duration' in result:
                successful_durations.append(result['duration'])
        
        print(f"⏱️  Total execution time: {total_duration:.2f}s")
        print(f"📈 Total jobs: {len(self.results)}")
        
        for status, count in status_counts.items():
            percentage = (count / len(self.results)) * 100
            print(f"   {status.upper()}: {count} ({percentage:.1f}%)")
        
        if successful_durations:
            avg_duration = sum(successful_durations) / len(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            
            print(f"\n⏱️  Successful jobs timing:")
            print(f"   Average: {avg_duration:.2f}s")
            print(f"   Min: {min_duration:.2f}s")
            print(f"   Max: {max_duration:.2f}s")
        
        # Check outputs
        print(f"\n📁 Output directories:")
        for result in self.results:
            if 'output_dir' in result:
                output_dir = result['output_dir']
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    print(f"   Seed {result['seed']}: {len(files)} files in {output_dir}")
                else:
                    print(f"   Seed {result['seed']}: No output directory")
        
        # Save detailed results
        # results_file = f"job_results_{int(time.time())}.json"
        # with open(results_file, 'w') as f:
        #     json.dump({
        #         'timestamp': time.time(),
        #         'total_duration': total_duration,
        #         'num_jobs': len(self.results),
        #         'status_counts': status_counts,
        #         'results': self.results,
        #         'test_type': 'job_submission'
        #     }, f, indent=2)
        
        # print(f"\n💾 Detailed results saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Job submission using run.sh')
    parser.add_argument('--num-jobs', type=int, default=1, help='Number of jobs to submit')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed for jobs')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum concurrent workers (default: num_jobs)')
    parser.add_argument('--display-output', action='store_true', help='Display output of the jobs')
    parser.add_argument('--use-veo', action='store_true', default=False, help='Use Veo video generation')
    
    args = parser.parse_args()
    
    # Create job submitter
    submitter = RunJobSubmitter()
    
    # Submit jobs
    results = submitter.submit_jobs(args.num_jobs, args.base_seed, args.max_workers, args.display_output, args.use_veo)
    
    # Exit with error code if any jobs failed
    failed_count = sum(1 for r in results if r['status'] != 'success')
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} jobs failed!")
        exit(1)
    else:
        print(f"\n🎉 All {len(results)} jobs completed successfully!")

if __name__ == "__main__":
    main() 