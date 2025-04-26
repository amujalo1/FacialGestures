# ai_commander.py - Standalone UI Voice Control via PyAutoGUI
import speech_recognition as sr
import threading
import time
import pyautogui # To simulate clicks
import traceback
import os # Needed for getenv

# --- Configuration ---
ACTIVATION_PHRASE = "computer" # Or "okay app", "hey accessicommand" etc.
# Whisper settings
WHISPER_MODEL_SIZE = "tiny.en"
PAUSE_THRESHOLD = 0.7 # Allow slightly longer pauses for commands
ENERGY_THRESHOLD = 400 # Adjust based on your mic sensitivity
PHRASE_TIME_LIMIT = 6 # Max length of command phrase

# --- Target Window/Element Info (CRITICAL - MUST BE ACCURATE) ---
APP_WINDOW_TITLE = "AccessiCommand" # Exact title of your main Tkinter window
# --- Use pyautogui.displayMousePosition() to find these coordinates! ---
# These are EXAMPLES - replace with YOUR coordinates for YOUR screen/window size/position
BUTTON_COORDS = {
    "start button": (236, 217),    # Center of START ENGINE button
    "stop button": (405, 217),     # Center of STOP ENGINE button
    "config button": (314, 276)    # Center of CONFIGURE BINDINGS button
    # Add coordinates for Config Dialog buttons if needed (e.g., "save", "cancel")
    # "save config": (X, Y),
    # "cancel config": (X, Y)
}
# --- Make sure coordinates are correct ---

class UICommander:
    def __init__(self):
        print("[UIC LOG] Initializing UICommander...") # Added Init Log
        self.recognizer = sr.Recognizer()
        # Attempt to initialize microphone
        self.microphone = None
        mic_init_success = False
        try:
            print("[UIC LOG] Attempting to initialize microphone...") # Added Mic Init Log Start
            self.microphone = sr.Microphone() # Use default device
            mic_init_success = True
            print("[UIC LOG] Microphone initialized successfully.") # Added Mic Init Log Success
            # Adjust for ambient noise right away
            print("[UIC LOG] Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"[UIC LOG] Ready. Energy threshold: {self.recognizer.energy_threshold:.2f}")
        except AttributeError as e:
             print(f"ERROR [UIC]: PyAudio missing? {e}") # Changed log prefix
             print("       Voice command execution will likely fail.")
        except Exception as e:
            print(f"ERROR [UIC]: Failed to initialize microphone: {e}") # Changed log prefix
            print("       Voice command execution will likely fail.")
            # No SystemExit here, allow script to run but log the error

        self.running = False
        self.thread = None
        self._is_listening = False

        # Configure recognizer (even if mic failed, set attributes)
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True

        print("--- UI Commander Initializing ---") # Kept this section separator
        print(f"Activation Phrase: '{ACTIVATION_PHRASE}'")
        print(f"Recognizer settings: energy={self.recognizer.energy_threshold}, pause={self.recognizer.pause_threshold}")
        print("------------------------------------")


    def _execute_action_by_keyword(self, command_text):
        """ Determines action based on simple keywords and executes via pyautogui. """
        print(f"[UIC LOG] Parsing command: '{command_text}'") # Added Log
        command = command_text.lower().strip()
        action_executed = False
        target_coord = None
        action_description = "Unknown Action" # For logging

        # Simple keyword matching -> coordinate lookup
        if ("start" in command or "run" in command or "activate" in command) and "engine" in command:
            target_coord = BUTTON_COORDS.get("start button")
            action_description = "Start Engine Button"
            print(f"[UIC LOG] Matched intent: Start Engine -> Coords: {target_coord}") # Added Log
        elif ("stop" in command or "pause" in command or "halt" in command) and "engine" in command:
            target_coord = BUTTON_COORDS.get("stop button")
            action_description = "Stop Engine Button"
            print(f"[UIC LOG] Matched intent: Stop Engine -> Coords: {target_coord}") # Added Log
        elif ("config" in command or "setting" in command or "binding" in command or "open" in command):
            target_coord = BUTTON_COORDS.get("config button")
            action_description = "Config Button"
            print(f"[UIC LOG] Matched intent: Open Config -> Coords: {target_coord}") # Added Log
        # Add more commands here...

        # Execute click if coordinates found
        if target_coord:
            print(f"[UIC LOG] Attempting to click {action_description} at {target_coord}") # Added Log
            try:
                # Optional: Bring window to front first
                windows = pyautogui.getWindowsWithTitle(APP_WINDOW_TITLE)
                if windows:
                    print(f"[UIC LOG] Found window '{APP_WINDOW_TITLE}', attempting activation...") # Added Log
                    try:
                        windows[0].activate()
                        time.sleep(0.2) # Small delay after activation
                        print("[UIC LOG] Window activated.") # Added Log
                    except Exception as act_e:
                         print(f"WARN [UIC]: Could not activate window '{APP_WINDOW_TITLE}': {act_e}")

                pyautogui.click(target_coord[0], target_coord[1])
                action_executed = True
                print("[UIC LOG] Click simulated successfully.") # Added Log
            except Exception as click_e:
                print(f"ERROR [UIC]: PyAutoGUI click failed for {action_description}: {click_e}") # Changed log prefix
                traceback.print_exc()
        else:
            # Only warn if command was not empty but no coords found
            if command:
                print(f"[UIC LOG] No action/coordinates mapped for command: '{command}'") # Added Log

        return action_executed


    def _listen_loop(self):
        """ Listens for activation phrase and processes commands. """
        print("[UIC LOG] Background listening thread started.") # Added Log
        if self.microphone is None:
            print("ERROR [UIC]: Mic unusable. Listener thread exiting.") # Changed log prefix
            self.running = False
            return

        while self.running:
            if not self.running:
                print("[UIC LOG] Running flag false, exiting loop.") # Added Log
                break
            audio_data = None
            # print(f"[UIC LOG] Listening for activation '{ACTIVATION_PHRASE}'...") # Noisy - keep commented
            self._is_listening = True
            try:
                # Listen for speech
                # print("[UIC LOG] Entering microphone context...") # Debug if needed
                with self.microphone as source:
                    # print("[UIC LOG] Calling recognizer.listen...") # Debug if needed
                    audio_data = self.recognizer.listen(
                        source, timeout=None, phrase_time_limit=PHRASE_TIME_LIMIT)
                self._is_listening = False
                # print("[UIC LOG] Exited microphone context.") # Debug if needed

                if audio_data and self.running: # Check running flag after blocking listen
                    # Transcribe
                    # print("[UIC LOG] Audio received, transcribing...") # Noisy
                    try:
                        recognized_text = self.recognizer.recognize_whisper(
                            audio_data, model=WHISPER_MODEL_SIZE, language="english").lower()
                        cleaned_text = recognized_text.strip(" .,!?\"'\n\t")
                        print(f"[UIC LOG] Heard: '{cleaned_text}'") # Log transcription

                        # Check for activation phrase
                        if cleaned_text.startswith(ACTIVATION_PHRASE):
                            command_text = cleaned_text[len(ACTIVATION_PHRASE):].strip()
                            if command_text:
                                print(f"[UIC LOG] Activation detected, processing command: '{command_text}'") # Added Log
                                # Directly execute based on keywords
                                self._execute_action_by_keyword(command_text)
                            else:
                                print("[UIC LOG] Activation phrase heard, no command.")
                        # else: print("[UIC LOG] Activation phrase not detected.") # Noisy

                    except sr.UnknownValueError: print("[UIC LOG] Whisper couldn't understand audio.") # Changed log prefix
                    except sr.RequestError as e: print(f"ERROR [UIC]: Whisper RequestError: {e}") # Changed log prefix
                    except Exception as e: print(f"ERROR [UIC]: Transcription/Processing error: {e}"); traceback.print_exc() # Changed log prefix
                # elif not audio_data:
                    # print("[UIC LOG] No audio data returned from listen.") # Noisy

            except sr.WaitTimeoutError: self._is_listening = False; print("[UIC LOG] WaitTimeoutError occurred."); continue # Added Log
            except OSError as e: print(f"ERROR [UIC]: Mic OS Error: {e}"); time.sleep(2); self._is_listening = False # Changed log prefix
            except Exception as e: print(f"ERROR [UIC]: Listen loop error: {e}"); traceback.print_exc(); time.sleep(1); self._is_listening = False # Changed log prefix

        print("[UIC LOG] Background listening thread stopped.") # Added Log


    def start(self):
        if self.running: print("[UIC LOG] Already running."); return # Changed log prefix
        # Check microphone again before starting thread
        if self.microphone is None: print("ERROR [UIC]: Cannot start - mic failed previously."); return # Changed log prefix
        print("--- UI Commander Starting ---")
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        if self.thread.is_alive(): print("[UIC LOG] Started successfully.") # Changed log prefix
        else: print("ERROR [UIC]: Failed thread start."); self.running = False # Changed log prefix

    def stop(self):
        if not self.running: print("[UIC LOG] Already stopped."); return # Changed log prefix
        print("--- UI Commander Stopping ---")
        self.running = False
        if self.thread and self.thread.is_alive():
            print("[UIC LOG] Waiting for listener thread..."); self.thread.join(timeout=1.0) # Short wait ok
            if self.thread.is_alive(): print("WARN [UIC]: Thread join timeout.") # Changed log prefix
        self.thread = None
        print("[UIC LOG] Stopped.") # Changed log prefix

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting UI Commander Standalone ---") # Changed title
    commander = None # Define before try block
    try:
        commander = UICommander()
        # Start only if microphone was initialized successfully
        if commander.microphone:
            commander.start()
            print(f"\nUI Commander active. Say '{ACTIVATION_PHRASE}' followed by a UI command.") # Changed title
            print("Examples: 'Computer, start engine', 'Computer, open config', 'Computer, stop engine'")
            print("Press Ctrl+C to stop.")
            while True: time.sleep(1) # Keep main thread alive while listener runs
        else:
            print("\nERROR: UI Commander could not start due to microphone initialization failure.") # Changed title
            print("       Please ensure PyAudio is installed correctly and a microphone is available.")

    except KeyboardInterrupt: print("\nCtrl+C detected. Stopping commander...")
    except SystemExit as e: print(f"Exiting due to error: {e}") # Catch SystemExit from init fail
    except Exception as main_e: print(f"\nERROR in main loop: {main_e}"); traceback.print_exc()
    finally:
        if commander: commander.stop() # Ensure stop is called if commander exists
        print("--- UI Commander Finished ---") # Changed title