#!/usr/bin/env python3
"""
NBA PRA Feature Engineering Pipeline Executor
Reads pipeline.yaml and orchestrates all feature generation steps

Usage:
    uv run feature_engineering/run_pipeline.py                    # Run full pipeline
    uv run feature_engineering/run_pipeline.py --skip-completed   # Resume from last failure
    uv run feature_engineering/run_pipeline.py --dry-run          # Show execution plan
    uv run feature_engineering/run_pipeline.py --only rolling,matchup  # Run specific stages
    uv run feature_engineering/run_pipeline.py --clean-start      # Reset state and run all
"""

import yaml
import subprocess
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import time
import os

class PipelineExecutor:
    def __init__(self, config_path: str = "pipeline.yaml", args: Optional[argparse.Namespace] = None):
        """Initialize pipeline executor with configuration."""
        self.script_dir = Path(__file__).parent
        self.config_path = self.script_dir / config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.args = args or argparse.Namespace()
        self.state_file = self.script_dir / self.config['config']['state_file']
        self.log_dir = self.script_dir / self.config['config']['log_dir']
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.setup_logging()
        self.load_state()

    def setup_logging(self):
        """Configure logging based on pipeline config."""
        log_level = getattr(logging, self.config['config']['log_level'])

        # Create formatters
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup logger
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        # Console handler
        if self.config['config']['log_to_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.config['config']['log_to_file']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"pipeline_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to: {log_file}")

    def load_state(self):
        """Load pipeline execution state from disk."""
        if self.state_file.exists() and not getattr(self.args, 'clean_start', False):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            self.logger.info(f"Loaded pipeline state from {self.state_file}")
        else:
            self.state = {
                'version': self.config['metadata']['version'],
                'last_run_start': None,
                'last_run_complete': None,
                'stages': {}
            }

    def save_state(self):
        """Persist pipeline state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def check_dependencies(self, stage: Dict[str, Any]) -> bool:
        """Check if all dependencies for a stage exist."""
        depends_on = stage.get('depends_on', [])

        for dep in depends_on:
            dep_path = self.script_dir / dep
            if not dep_path.exists():
                self.logger.error(f"Dependency not found: {dep}")
                self.logger.error(f"  Expected at: {dep_path}")

                # Provide helpful guidance
                if 'player_games.parquet' in str(dep):
                    self.logger.error("  → Run data_loader.py first: uv run feature_engineering/data_loader.py")
                elif 'feature_tables/' in str(dep):
                    feature_name = Path(dep).stem
                    self.logger.error(f"  → Run {feature_name}.py first: uv run feature_engineering/{feature_name}.py")

                return False

        return True

    def should_skip_stage(self, stage_name: str) -> bool:
        """Determine if stage should be skipped (already completed)."""
        if not getattr(self.args, 'skip_completed', False):
            return False

        if stage_name in self.state['stages']:
            stage_state = self.state['stages'][stage_name]
            if stage_state.get('status') == 'completed':
                self.logger.info(f"⏭️  Skipping {stage_name} (already completed)")
                return True

        return False

    def run_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        stage_name = stage['name']
        script = stage['script']

        # Check if stage is enabled
        if not stage.get('enabled', True):
            self.logger.info(f"⏭️  Skipping {stage_name} (disabled in config)")
            return {'stage': stage_name, 'status': 'skipped', 'reason': 'disabled'}

        # Check if should skip (already completed)
        if self.should_skip_stage(stage_name):
            return {'stage': stage_name, 'status': 'skipped', 'reason': 'already_completed'}

        # Check dependencies
        if not self.check_dependencies(stage):
            return {
                'stage': stage_name,
                'status': 'failed',
                'error': 'Missing dependencies',
                'duration': 0
            }

        # Log stage start
        self.logger.info("=" * 60)
        self.logger.info(f"Starting: {stage_name}")
        self.logger.info(f"Description: {stage.get('description', 'N/A')}")
        self.logger.info(f"Script: {script}")
        if 'estimated_duration' in stage:
            self.logger.info(f"Estimated duration: {stage['estimated_duration']}")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            # Build command - run from project root for proper imports
            project_root = self.script_dir.parent
            script_path = self.script_dir / script
            cmd = [sys.executable, str(script_path)]

            # Execute from project root so feature_engineering package imports work
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )

            duration = time.time() - start_time

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    self.logger.info(f"  {line}")

            if result.returncode != 0:
                self.logger.error(f"❌ Stage {stage_name} FAILED (exit code {result.returncode})")

                if result.stderr:
                    self.logger.error("Error output:")
                    for line in result.stderr.strip().split('\n'):
                        self.logger.error(f"  {line}")

                # Update state
                self.state['stages'][stage_name] = {
                    'status': 'failed',
                    'completed_at': datetime.now().isoformat(),
                    'duration_seconds': duration,
                    'error': result.stderr[:500]  # Truncate long errors
                }
                self.save_state()

                # Check if critical
                if stage.get('critical', False) or self.config['config']['fail_fast']:
                    raise RuntimeError(f"Critical stage {stage_name} failed")

                return {
                    'stage': stage_name,
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr
                }

            # Success
            self.logger.info(f"✅ Stage {stage_name} completed in {duration:.1f}s")

            # Validate outputs
            outputs = stage.get('outputs', [])
            for output in outputs:
                output_path = self.script_dir / output
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    self.logger.info(f"  Created: {output} ({size_mb:.1f} MB)")
                else:
                    self.logger.warning(f"  ⚠️  Expected output not found: {output}")

            # Update state
            self.state['stages'][stage_name] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'outputs': outputs
            }
            self.save_state()

            return {
                'stage': stage_name,
                'status': 'success',
                'duration': duration
            }

        except (subprocess.CalledProcessError, FileNotFoundError, ImportError, ValueError, KeyError) as e:
            # Subprocess failed, module not found, or data validation error
            duration = time.time() - start_time
            self.logger.error(f"❌ Stage {stage_name} raised exception: {type(e).__name__}: {e}")

            self.state['stages'][stage_name] = {
                'status': 'error',
                'completed_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'error': f"{type(e).__name__}: {str(e)}"
            }
            self.save_state()

            if stage.get('critical', False) or self.config['config']['fail_fast']:
                raise

            return {
                'stage': stage_name,
                'status': 'error',
                'duration': duration,
                'error': str(e)
            }

    def filter_stages(self) -> List[Dict[str, Any]]:
        """Filter stages based on command-line arguments."""
        stages = self.config['stages']

        # --only flag: run only specified stages
        if hasattr(self.args, 'only') and self.args.only:
            only_stages = [s.strip() for s in self.args.only.split(',')]
            stages = [s for s in stages if s['name'] in only_stages]
            self.logger.info(f"Running only: {', '.join(only_stages)}")

        # --from flag: run from specified stage onward
        elif hasattr(self.args, 'from_stage') and self.args.from_stage:
            start_idx = next((i for i, s in enumerate(stages) if s['name'] == self.args.from_stage), 0)
            stages = stages[start_idx:]
            self.logger.info(f"Running from {self.args.from_stage} onward")

        # --to flag: run up to specified stage
        if hasattr(self.args, 'to_stage') and self.args.to_stage:
            end_idx = next((i for i, s in enumerate(stages) if s['name'] == self.args.to_stage), len(stages) - 1)
            stages = stages[:end_idx + 1]
            self.logger.info(f"Running up to {self.args.to_stage}")

        return stages

    def dry_run(self):
        """Show execution plan without running."""
        self.logger.info("=" * 60)
        self.logger.info("DRY RUN - Execution Plan")
        self.logger.info("=" * 60)

        stages = self.filter_stages()

        for i, stage in enumerate(stages, 1):
            stage_name = stage['name']

            # Check status
            will_run = stage.get('enabled', True)
            will_skip = self.should_skip_stage(stage_name)

            status = "✓ Will run" if will_run and not will_skip else "⏭️  Will skip"

            self.logger.info(f"{i}. {stage_name}: {status}")
            self.logger.info(f"   Description: {stage.get('description', 'N/A')}")

            if 'estimated_duration' in stage:
                self.logger.info(f"   Estimated duration: {stage['estimated_duration']}")

            # Check dependencies
            depends_on = stage.get('depends_on', [])
            if depends_on:
                self.logger.info(f"   Dependencies:")
                for dep in depends_on:
                    dep_path = self.script_dir / dep
                    exists = "✓" if dep_path.exists() else "✗"
                    self.logger.info(f"     {exists} {dep}")

            self.logger.info("")

        self.logger.info("=" * 60)
        self.logger.info("Use --help to see available options")
        self.logger.info("Run without --dry-run to execute pipeline")

    def run(self):
        """Execute the complete pipeline."""
        # Dry run mode
        if getattr(self.args, 'dry_run', False):
            self.dry_run()
            return True

        # Start pipeline
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline: {self.config['metadata']['name']}")
        self.logger.info(f"Version: {self.config['metadata']['version']}")
        self.logger.info(f"Description: {self.config['metadata']['description']}")
        self.logger.info("=" * 60)

        self.state['last_run_start'] = datetime.now().isoformat()
        self.save_state()

        pipeline_start = time.time()
        results = []

        try:
            stages = self.filter_stages()

            for i, stage in enumerate(stages, 1):
                self.logger.info(f"\n[{i}/{len(stages)}] Processing stage: {stage['name']}")

                result = self.run_stage(stage)
                results.append(result)

                # Stop if failed and fail_fast enabled
                if result['status'] in ['failed', 'error']:
                    if stage.get('critical', False) or self.config['config']['fail_fast']:
                        self.logger.error(f"Stopping pipeline due to {stage['name']} failure")
                        break

            pipeline_duration = time.time() - pipeline_start

            # Summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PIPELINE SUMMARY")
            self.logger.info("=" * 60)

            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = sum(1 for r in results if r['status'] in ['failed', 'error'])
            skipped_count = sum(1 for r in results if r['status'] == 'skipped')

            self.logger.info(f"Total stages: {len(results)}")
            self.logger.info(f"✅ Successful: {success_count}")
            self.logger.info(f"❌ Failed: {failed_count}")
            self.logger.info(f"⏭️  Skipped: {skipped_count}")
            self.logger.info(f"⏱️  Total time: {pipeline_duration:.1f}s ({pipeline_duration / 60:.1f} min)")

            # Detailed results
            for result in results:
                if result['status'] == 'success':
                    self.logger.info(f"  ✅ {result['stage']}: {result['duration']:.1f}s")
                elif result['status'] in ['failed', 'error']:
                    self.logger.error(f"  ❌ {result['stage']}: {result.get('error', 'Unknown error')[:100]}")
                elif result['status'] == 'skipped':
                    reason = result.get('reason', 'unknown')
                    self.logger.info(f"  ⏭️  {result['stage']}: {reason}")

            # Update state
            self.state['last_run_complete'] = datetime.now().isoformat()
            self.save_state()

            # Check if all succeeded
            all_success = failed_count == 0

            if all_success:
                self.logger.info("\n🎉 Pipeline completed successfully!")
                self.logger.info(f"State saved to: {self.state_file}")
            else:
                self.logger.error(f"\n❌ Pipeline completed with {failed_count} failures")
                self.logger.info("Use --skip-completed to resume from last successful stage")

            return all_success

        except (subprocess.CalledProcessError, FileNotFoundError, ImportError, ValueError, KeyError, OSError) as e:
            # Pipeline-level error - could be any stage failure
            pipeline_duration = time.time() - pipeline_start
            self.logger.error(f"\n❌ Pipeline FAILED after {pipeline_duration:.1f}s")
            self.logger.error(f"Error ({type(e).__name__}): {e}")

            self.state['last_run_complete'] = datetime.now().isoformat()
            self.state['last_error'] = f"{type(e).__name__}: {str(e)}"
            self.save_state()

            return False


def main():
    parser = argparse.ArgumentParser(
        description='NBA PRA Feature Engineering Pipeline Executor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full pipeline:
    uv run feature_engineering/run_pipeline.py

  Resume from last failure:
    uv run feature_engineering/run_pipeline.py --skip-completed

  Show execution plan (dry run):
    uv run feature_engineering/run_pipeline.py --dry-run

  Run only specific stages:
    uv run feature_engineering/run_pipeline.py --only rolling_features,matchup_features

  Run from a specific stage onward:
    uv run feature_engineering/run_pipeline.py --from contextual_features

  Clean start (reset state):
    uv run feature_engineering/run_pipeline.py --clean-start
        """
    )

    parser.add_argument(
        '--config',
        default='pipeline.yaml',
        help='Path to pipeline configuration file (default: pipeline.yaml)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show execution plan without running'
    )

    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip stages that completed successfully in previous run'
    )

    parser.add_argument(
        '--clean-start',
        action='store_true',
        help='Reset pipeline state and run all stages'
    )

    parser.add_argument(
        '--only',
        type=str,
        help='Run only specified stages (comma-separated, e.g., "rolling_features,matchup_features")'
    )

    parser.add_argument(
        '--from',
        dest='from_stage',
        type=str,
        help='Run from this stage onward'
    )

    parser.add_argument(
        '--to',
        dest='to_stage',
        type=str,
        help='Run up to and including this stage'
    )

    args = parser.parse_args()

    # Execute pipeline
    executor = PipelineExecutor(config_path=args.config, args=args)
    success = executor.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
