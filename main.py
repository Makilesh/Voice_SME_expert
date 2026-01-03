"""Application entry point and CLI interface."""
import argparse
import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, AppConfig
from src.utils import setup_logging, MemoryMonitor, PerformanceTracker
from orchestrator import MeetingOrchestrator

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Subject Matter Expert Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["microphone", "virtual", "zoom", "meet", "teams"],
        help="Audio capture mode"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="default",
        help="Audio device name or index (for microphone mode)"
    )
    
    parser.add_argument(
        "--application",
        type=str,
        choices=["zoom", "meet", "teams"],
        help="Application to capture audio from (for virtual mode)"
    )
    
    parser.add_argument(
        "--meeting-id",
        type=str,
        help="Meeting ID (for zoom mode)"
    )
    
    parser.add_argument(
        "--meeting-url",
        type=str,
        help="Meeting URL (for meet/teams mode)"
    )
    
    parser.add_argument(
        "--password",
        type=str,
        help="Meeting password (for zoom mode)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="AI Expert",
        help="Display name in meeting"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to .env configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


async def run_assistant(config: AppConfig, mode: str, args: argparse.Namespace) -> None:
    """
    Runs the assistant in specified mode.
    
    Parameters:
        config: Application configuration
        mode: Operating mode
        args: Command line arguments
    """
    logger.info(f"Starting assistant in {mode} mode")
    
    # Initialize orchestrator
    orchestrator = MeetingOrchestrator(config)
    
    try:
        # Determine audio source
        if mode == "microphone":
            audio_source = f"microphone:{args.device}"
        elif mode == "virtual":
            if args.application:
                audio_source = f"virtual:{args.application}"
            else:
                audio_source = "virtual"
        elif mode == "zoom":
            if not args.meeting_id:
                logger.error("--meeting-id required for zoom mode")
                return
            audio_source = f"zoom:{args.meeting_id}"
        elif mode == "meet":
            if not args.meeting_url:
                logger.error("--meeting-url required for meet mode")
                return
            audio_source = f"meet:{args.meeting_url}"
        elif mode == "teams":
            if not args.meeting_url:
                logger.error("--meeting-url required for teams mode")
                return
            audio_source = f"teams:{args.meeting_url}"
        else:
            logger.error(f"Unknown mode: {mode}")
            return
        
        # Start the orchestrator
        await orchestrator.start(audio_source)
        
        # Keep running until interrupted
        logger.info("Assistant is running. Press Ctrl+C to stop.")
        
        # Wait indefinitely
        try:
            while True:
                await asyncio.sleep(1)
                
                # Periodic status check
                if logger.isEnabledFor(logging.DEBUG):
                    status = orchestrator.get_meeting_status()
                    logger.debug(f"Status: {status}")
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
    
    finally:
        # Clean shutdown
        await orchestrator.stop()
        logger.info("Assistant stopped")


def print_banner():
    """Print application banner."""
    banner = """
==================================================
    Subject Matter Expert Voice Assistant v1.0
==================================================
"""
    print(banner)


def main():
    """
    Main entry point.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set log level
        log_level = "DEBUG" if args.debug else args.log_level
        
        # Setup logging
        setup_logging(level=log_level)
        
        # Print banner
        print_banner()
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Override config with CLI args
        if args.name:
            config.meeting_display_name = args.name
        
        logger.info("Configuration loaded successfully")
        
        # Run the assistant
        asyncio.run(run_assistant(config, args.mode, args))
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
