# accessicommand/core/engine.py
import sys # Import sys first for path debugging
import os # Import os for path calculation and cwd
import traceback
import time # Moved import down slightly

# --- DEBUG: Print Python Path and Working Directory ---
# Add these lines RIGHT AT THE TOP before other imports that might fail
print(f"--- Debug Info from engine.py ---")
print(f"Current Working Directory: {os.getcwd()}")
print("Python sys.path:")
for p in sys.path:
    print(f"  - {p}")
print("---------------------------------")
# --- END DEBUG ---


# --- Use ABSOLUTE Imports (Requires running with -m from project root) ---
try:
    # Import directly from the 'accessicommand' package namespace
    print("Attempting absolute imports...") # Add debug print
    from accessicommand.config.manager import ConfigManager
    print(" -> Imported ConfigManager OK") # Add OK message
    from accessicommand.detectors.voice_detector import VoiceDetector
    print(" -> Imported VoiceDetector OK")
    from accessicommand.actions.registry import get_action_function, ACTION_REGISTRY
    print(" -> Imported Action Registry OK")

    # --- FACIAL DETECTOR PLACEHOLDER (using absolute import path) ---
    try:
        from accessicommand.detectors.facial_detector import FacialDetector
        print(" -> Imported FacialDetector OK")
    except ImportError:
        # Keep the dummy if the real one isn't created yet
        print("WARN: Real FacialDetector (accessicommand.detectors.facial_detector) not found, using dummy.")
        class FacialDetector:
            def __init__(self, **kwargs): print("Dummy FacialDetector created")
            def start(self): print("Dummy FacialDetector started")
            def stop(self): print("Dummy FacialDetector stopped")
    # ----------------------------------------------------------------

    _imports_ok = True
    print("Absolute imports successful.")
except ImportError as e:
    print(f"ERROR: Could not perform absolute imports in engine.py: {e}")
    print("       Make sure you run this script using 'python -m accessicommand.core.engine' from the project root directory (FacialGestures/).")
    print("       Also ensure all __init__.py files exist in the necessary directories.")
    _imports_ok = False
    # Exit if imports fail, as the rest of the script depends on them
    sys.exit(1)


# --- Configuration Constants (Defaults for Engine Initialization) ---
# These might be used if settings are missing from the config file
DEFAULT_VOICE_PAUSE_THRESHOLD = 0.4
DEFAULT_VOICE_ENERGY_THRESHOLD = 350
# Add defaults for facial detector here later if needed

class Engine:
    """
    The core engine that manages detectors, loads configuration,
    handles events, and executes corresponding actions.
    """
    def __init__(self, config_path="config.json"): # Path relative to where script is run/project root
        print("--- Engine Initializing ---")
        # Use the imported ConfigManager directly
        self.config_manager = ConfigManager(config_path)
        self.detectors = {} # Dictionary to hold active detectors {detector_type: instance}
        self.bindings = [] # Initialize as empty list
        self.settings = {} # Initialize as empty dict
        self.is_running = False

        self._load_configuration()
        self._initialize_actions() # Checks registry access
        self._initialize_detectors()
        print("--- Engine Initialized ---")

    def _load_configuration(self):
        """Loads bindings and settings from the config manager."""
        print(f"Engine: Loading configuration from '{self.config_manager.config_path}'...")
        try:
            # ConfigManager handles loading or creating default internally now
            self.config_data = self.config_manager.get_config() # Get current data
            self.bindings = self.config_manager.get_bindings()
            self.settings = self.config_manager.get_settings()
            print(f"Engine: Loaded {len(self.bindings)} bindings.")
        except Exception as e:
            print(f"ERROR: Failed to load configuration via ConfigManager: {e}")
            traceback.print_exc()
            self.bindings = [] # Reset on error
            self.settings = {} # Reset on error

    def _initialize_actions(self):
        """Verifies access to the action registry."""
        if not callable(get_action_function):
             print("ERROR: Action registry function 'get_action_function' not available!")
        else:
            print(f"Engine: Action registry access initialized ({len(ACTION_REGISTRY)} actions available).")


    def _initialize_detectors(self):
        """Creates and initializes detector instances based on configuration."""
        print("Engine: Initializing detectors...")
        self.detectors = {} # Clear previous detectors

        # --- Voice Detector Initialization ---
        # Extract trigger words needed for voice detector from bindings
        # Ensure trigger_event exists and is treated as a string before lowercasing
        voice_trigger_words = set(
            str(b.get("trigger_event")).lower()
            for b in self.bindings
            if b.get("trigger_type") == "voice" and b.get("trigger_event") is not None
        )

        if voice_trigger_words:
            try:
                print(f"Engine: Initializing VoiceDetector...")
                # Get voice settings from loaded config, use engine defaults if missing
                voice_settings = self.settings.get('voice_detector', {})
                self.detectors['voice'] = VoiceDetector(
                    trigger_words=list(voice_trigger_words), # Pass the identified trigger words
                    event_handler=self.handle_event, # Pass the engine's handler method
                    # Use loaded setting OR engine default
                    energy_threshold=voice_settings.get('energy_threshold', DEFAULT_VOICE_ENERGY_THRESHOLD),
                    pause_threshold=voice_settings.get('pause_threshold', DEFAULT_VOICE_PAUSE_THRESHOLD),
                )
            except Exception as e:
                 print(f"ERROR: Failed to initialize VoiceDetector: {e}")
                 traceback.print_exc()
        else:
            print("Engine: No voice bindings found, VoiceDetector not initialized.")


        # --- Facial Detector Initialization ---
        # Check if any bindings need the facial detector
        facial_triggers_needed = any(b.get("trigger_type") == "face" for b in self.bindings)

        if facial_triggers_needed:
            # Check if the real FacialDetector class was imported successfully
            # Need a more robust check than just __name__ if dummy is also named FacialDetector
            # Let's assume if _imports_ok is True and FacialDetector exists, it's the real one for now
            if 'FacialDetector' in globals() and _imports_ok:
                try:
                    print("Engine: Initializing FacialDetector...")
                    facial_settings = self.settings.get('facial_detector', {})
                    # --- TODO: Pass relevant settings from config to the real FacialDetector ---
                    self.detectors['face'] = FacialDetector(event_handler=self.handle_event) # Pass handler
                except Exception as e:
                     print(f"ERROR: Failed to initialize FacialDetector: {e}")
                     traceback.print_exc()
            else:
                # If the real FacialDetector class wasn't imported or is the dummy
                print("WARN: Facial bindings found, but using Dummy FacialDetector.")
                # Initialize the dummy, still passing the handler
                # Make sure dummy class is defined if initial import failed
                if 'FacialDetector' not in globals():
                     class FacialDetector: # Define dummy inline if needed
                         def __init__(self, **kwargs): print("Inline Dummy FacialDetector created")
                         def start(self): print("Inline Dummy FacialDetector started")
                         def stop(self): print("Inline Dummy FacialDetector stopped")
                self.detectors['face'] = FacialDetector(event_handler=self.handle_event)
        else:
             print("Engine: No facial bindings found, FacialDetector not initialized.")

        print(f"Engine: Initialized active detectors: {list(self.detectors.keys())}")


    def handle_event(self, detector_type, event_data):
        """
        Callback method for detectors. Receives events and triggers actions based on bindings.
        Args:
            detector_type (str): The type of detector sending the event (e.g., "voice", "face").
            event_data (any): The specific event data (e.g., "next", "LEFT_WINK").
        """
        if not self.is_running: return # Don't process events if detection isn't active

        print(f"Engine: Event received - Type: '{detector_type}', Data: '{event_data}'")
        # Ensure event_data is treated as a lowercase string for matching
        event_key = str(event_data).lower()
        action_id_to_execute = None

        # --- Find Matching Binding ---
        # Iterate through the list of binding dictionaries loaded from config
        for binding in self.bindings:
            # Check type and ensure trigger_event from config is also lowercased string
            if binding.get("trigger_type") == detector_type and \
               str(binding.get("trigger_event", "")).lower() == event_key:
                action_id_to_execute = binding.get("action_id")
                if action_id_to_execute: # Ensure action_id is not empty/None
                     print(f"Engine: Found binding -> Action ID '{action_id_to_execute}'")
                     break # Found the first match, stop searching
                else:
                     print(f"WARN: Binding found for event '{event_key}', but 'action_id' is missing or empty.")

        # --- Execute Action ---
        if action_id_to_execute:
            action_func = get_action_function(action_id_to_execute) # Get function from registry
            if action_func:
                try:
                    print(f"Engine: Executing action '{action_id_to_execute}'...")
                    action_func() # Call the actual action function (e.g., the lambda calling pyautogui)
                except Exception as e:
                    print(f"ERROR: executing action '{action_id_to_execute}': {e}")
                    traceback.print_exc()
            else:
                # This means the action_id from config doesn't exist in actions/registry.py
                print(f"WARN: Action ID '{action_id_to_execute}' (from config) not found in action registry!")


    def start(self):
        """Starts all initialized detectors."""
        if self.is_running:
            print("Engine: Already running.")
            return
        print("--- Engine Starting Detectors ---")
        self.is_running = True
        if not self.detectors:
             print("Engine: No detectors initialized to start.")
             return

        start_success_count = 0
        for detector_type, detector_instance in self.detectors.items():
            try:
                print(f"Engine: Starting {detector_type} detector...")
                detector_instance.start()
                start_success_count += 1
            except Exception as e:
                print(f"ERROR: Failed to start detector '{detector_type}': {e}")
                traceback.print_exc()
        print(f"Engine: Started {start_success_count}/{len(self.detectors)} detectors.")


    def stop(self):
        """Stops all initialized detectors."""
        # Check if there are detectors to stop, even if running flag might be false
        if not self.detectors:
             print("Engine: No detectors initialized to stop.")
             if not self.is_running: # If also not running, definitely nothing to do
                 print("Engine: Already stopped.")
             return
        # Proceed if there are detectors OR if running flag is true
        if not self.is_running and self.detectors:
             print("Engine: Was not marked as running, but attempting to stop initialized detectors...")
        elif not self.is_running: # No detectors and not running
             print("Engine: Already stopped.")
             return

        print("--- Engine Stopping Detectors ---")
        self.is_running = False # Set flag immediately
        stop_success_count = 0
        for detector_type, detector_instance in self.detectors.items():
            try:
                # Check if detector has a 'stop' method before calling
                if callable(getattr(detector_instance, "stop", None)):
                    print(f"Engine: Stopping {detector_type} detector...")
                    detector_instance.stop()
                    stop_success_count += 1
                else:
                     print(f"WARN: Detector '{detector_type}' has no stop() method.")
            except Exception as e:
                print(f"ERROR: Failed to stop detector '{detector_type}': {e}")
                traceback.print_exc()
        print(f"Engine: Stopped {stop_success_count}/{len(self.detectors)} detectors.")


    def reload_configuration(self):
        """Stops detectors, reloads config, re-initializes, and restarts detectors."""
        print("--- Engine Reloading Configuration ---")
        self.stop()
        # Optional delay might be needed if detectors take time to fully stop threads
        # time.sleep(0.2)
        self._load_configuration()
        # Re-initialize detectors based on potentially new config
        self._initialize_detectors()
        self.start() # Restart the engine and its detectors
        print("--- Engine Reload Complete ---")


# --- Integration Test Block (To be run via 'python -m accessicommand.core.engine') ---
if __name__ == '__main__':
    # Ensure imports worked before proceeding
    if not _imports_ok:
        print("Exiting due to import errors.")
        sys.exit(1)

    print("--- Running Engine Directly (Integration Test) ---")

    # --- Use REAL Components ---
    # Calculate path to config.json relative to this file's location
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root (FacialGestures/) relative to core/
    project_root = os.path.dirname(os.path.dirname(current_script_dir))
    # Assume config.json is directly in the project root
    config_file_path = os.path.join(project_root, "config.json")
    print(f"Using config file: {config_file_path}")

    # Check if the config file actually exists
    if not os.path.exists(config_file_path):
        print(f"ERROR: Config file not found at '{config_file_path}'!")
        print("       Please create 'config.json' in the project root directory (e.g., FacialGestures/).")
        sys.exit(1)

    # --- Create the Engine using the REAL ConfigManager and config path ---
    # The Engine will now use the real components imported via ABSOLUTE paths at the top.
    engine = Engine(config_path=config_file_path)

    # --- Start the engine ---
    # This will start the threads for initialized detectors (VoiceDetector, maybe dummy FacialDetector)
    engine.start()

    # --- User Interaction / Keep Alive ---
    if not engine.detectors:
        print("\nWARN: Engine started, but no detectors were initialized based on config.")
        print("      Check 'bindings' in config.json.")
    else:
        print(f"\nEngine running with detectors: {list(engine.detectors.keys())}")
        print("Speak one of the configured commands (e.g., 'next', 'go', 'record').")
        print("Watch the console output for event handling and action execution.")
        print("Check if the corresponding pyautogui action happens!")

    print("\nPress Ctrl+C in this terminal to stop.")      

    try:
        # Keep main thread alive while the engine (and detector threads) run
        while True:
            # Add a check here to see if engine threads are alive? Optional.
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping engine...")
    except Exception as main_e:
        print(f"\nERROR in main test loop: {main_e}")
        traceback.print_exc()
    finally:
        # Ensure engine and its detector threads are stopped cleanly on exit
        engine.stop()
        print("--- Engine Test Finished ---")  