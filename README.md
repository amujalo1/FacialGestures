# AccessiCommand: Control Your Computer with Voice and Gestures

## Project Description

AccessiCommand is a hackathon project aimed at creating an alternative and intuitive way for users to interact with their computers using facial gestures, hand gestures, and voice commands. Traditional input methods (keyboard/mouse) can be limiting for various reasons, including accessibility needs or simply providing a different kind of user experience (like gaming). AccessiCommand provides a flexible framework to map these alternative inputs to standard system actions.

## Features

*   **Multimodal Input:** Supports control via:
    *   Voice Commands (Configurable keywords)
    *   Facial Gestures (Blinks, eyebrow raise, mouth open, head tilt)
    *   Hand Gestures (Open palm, fist, thumbs up, pointing index, victory sign)
*   **Configurable Bindings:** Easily map specific voice commands, facial gestures, or hand gestures to system actions (like pressing keyboard keys, mouse clicks, or executing shortcuts).
*   **Intuitive GUI:** Simple graphical interface to start/stop the input detection engine and configure custom bindings.
*   **Separate UI Voice Control:** Use voice commands (e.g., "Computer, start engine", "Computer, open configuration") to control the application's GUI itself, even before the main detection engine is running.
*   **Combined Visual Detection:** Process face and hand tracking from a single camera feed simultaneously.
*   **Modular Architecture:** Designed with separate components for detection (Voice, Face, Hand), core logic (Engine), actions (Registry, System Actions), and configuration (ConfigManager, config file), making it extensible.
*   **Offline Voice Transcription:** Uses Whisper via `speech_recognition` for fast, local speech-to-text.

## Architecture Overview

AccessiCommand follows a modular architecture:

1.  **Detectors (`detectors/`):** Independent modules (`voice_detector.py`, `facial_detector.py`, `hand_detector.py`) responsible *only* for detecting specific events from input streams (microphone, camera). They do *not* know what action should result. When an event is detected (e.g., "go" heard, "left eyebrow raised start"), they report it by calling the `Engine`'s `handle_event` method.
2.  **Engine (`core/engine.py`):** The central coordinator.
    *   Initializes based on configuration loaded by the `ConfigManager`.
    *   Creates instances of the configured Detectors, providing them its `handle_event` method as a callback.
    *   Manages camera capture for visual detectors and distributes frames.
    *   When `handle_event` is called by a detector, the Engine looks up the incoming event in its loaded `bindings`.
    *   If a matching binding is found, it uses the `ActionRegistry` to find the corresponding action function and executes it.
    *   Runs visual processing in a background thread.
3.  **Configuration (`config/`):**
    *   `config.json`: The primary file (in the project root) storing user-defined bindings and settings.
    *   `manager.py`: Contains the `ConfigManager` class responsible for reading and writing `config.json`.
4.  **Actions (`actions/`):**
    *   `registry.py`: Maps Action IDs (like "PRESS_SPACE", "MOUSE_CLICK_LEFT") defined in `config.json` to actual Python functions.
    *   `system_actions.py`: Contains the low-level Python functions (using `pyautogui`) that perform operations on the computer.
5.  **User Interface (`ui/`):**
    *   `main.py`: The application entry point, initializes the main GUI window and the core `Engine`.
    *   `main_window.py`: Defines the main application window with Start/Stop and Configure buttons. Interacts with the `Engine`.
    *   `config_dialog.py`: Defines the dialog window for adding, deleting, and saving bindings to `config.json`. Interacts with `ConfigManager` and `Engine` (to trigger reload).
6.  **UI Voice Commander (`ui_commander.py`):** A **separate** script/process that runs alongside the main app. It listens for specific UI command phrases (like "start engine") and uses `pyautogui` to simulate clicks on the main application window buttons.

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/YourRepo/FacialGestures.git # Replace with your actual repo URL
    cd FacialGestures
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    ```
3.  **Activate the Virtual Environment:**
    *   On Windows Command Prompt or PowerShell:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On Linux or macOS (or Git Bash on Windows):
        ```bash
        source .venv/bin/activate
        ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   *(If you don't have `requirements.txt`, manually install the key packages: `pip install mediapipe opencv-python pyautogui speechrecognition pyaudio openai-whisper`)*
    *   **PyAudio Note for Windows:** `pip install PyAudio` can sometimes fail on Windows. If it does, you'll need to download a pre-compiled `.whl` file from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) that matches your Python version (`cp311` for 3.11) and architecture (`win_amd64`) and install it manually (e.g., `pip install "PyAudio‑0.2.11‑cp311‑cp311‑win_amd64.whl"`).
    *   **Whisper Model:** The `openai-whisper` library will automatically download the necessary model file (e.g., `tiny.en.pt`) on its first use, which might take a moment.
5.  **Ensure `__init__.py` Files Exist:** Verify that you have an empty file named `__init__.py` inside the `accessicommand` directory and all its subdirectories (`actions`, `config`, `core`, `detectors`, `ui`). Python needs these to recognize directories as packages.
6.  **Create Default Configuration:** Create a file named `config.json` in the **root directory** (`FacialGestures/`) with your desired initial bindings and settings. See the example in the Architecture section above or create a basic one:
    ```json
    {
        "bindings": [
            { "trigger_type": "voice", "trigger_event": "go", "action_id": "PRESS_RIGHT" },
            { "trigger_type": "face", "trigger_event": "MOUTH_OPEN_START", "action_id": "PRESS_K_DOWN" },
            { "trigger_type": "face", "trigger_event": "MOUTH_OPEN_STOP", "action_id": "PRESS_K_UP" }
            // Add more bindings here via the GUI later
        ],
        "settings": {
            "voice_detector": {}, // Use defaults
            "facial_detector": {"camera_index": 0}, // Use camera 0 for face
            "hand_detector": {"camera_index": 0} // Use camera 0 for hand (can be same or different)
        }
    }
    ```
7.  **Find PyAutoGUI Coordinates (for `ui_commander.py`):**
    *   Run the main application: `python accessicommand/main.py`
    *   Position the window where you want it.
    *   Open a **separate terminal** (with venv active) and run:
        ```bash
        python -c "import pyautogui, time; print('Hover mouse over target in 5s...'); time.sleep(5); print(f'Coords: {pyautogui.position()}')"
        ```
    *   Quickly move your mouse over the center of the "START ENGINE", "STOP ENGINE", and "CONFIGURE BINDINGS" buttons and note the coordinates.
    *   Open `ai_commander.py` and **replace the placeholder coordinates** in the `BUTTON_COORDS` dictionary with the coordinates you found.

## How to Run and Use

1.  **Start the Main Application:**
    *   Open your terminal in the `FacialGestures/` root directory.
    *   Activate your virtual environment (`.\.venv\Scripts\activate` or `source .venv/bin/activate`).
    *   Run the main application script:
        ```bash
        python accessicommand/main.py
        ```
    *   This will open the GUI window and initialize the core engine components (but they won't be actively detecting yet).

2.  **Start the UI Voice Commander (Optional):**
    *   Open a **separate terminal window** (or use a split terminal).
    *   Navigate to the `FacialGestures/` root directory.
    *   Activate your virtual environment.
    *   Run the standalone UI commander script:
        ```bash
        python ai_commander.py
        ```
    *   This script will start listening for UI command phrases immediately.

3.  **Control the GUI (Using `ui_commander.py`):**
    *   With both windows/scripts running, activate the terminal where `ai_commander.py` is running.
    *   Speak clearly: Say the activation phrase (**"computer"**) followed by a command.
    *   Examples:
        *   "Computer, **start engine**"
        *   "Computer, **open configuration**"
        *   "Computer, **stop engine**" (Only works if engine is running)
        *   "Computer, **close application**" (Will attempt to close the main window)
    *   Watch the console output of `ai_commander.py` to see the transcription and the simulated click action. Watch the main GUI to see if the corresponding button is clicked.

4.  **Control the Computer (Using Configured Bindings):**
    *   In the main GUI window, click the **"START ENGINE"** button (or use the voice command "Computer, start engine").
    *   The GUI status should change to "Engine Running". The console will show the engine and detectors starting (including camera initialization).
    *   Now, perform facial or hand gestures, or speak system trigger words defined in your `config.json` (e.g., "go", "back").
    *   The corresponding actions (like pressing keys via `pyautogui`) should be triggered.

5.  **Configure Bindings:**
    *   Click the **"CONFIGURE BINDINGS"** button (or use the voice command "Computer, open configuration").
    *   A new configuration window will open.
    *   Add new bindings by selecting Type (voice, face, hand), Event (keyword or gesture event), and Action (from the list).
    *   Select existing bindings and click "Delete Selected" to remove them.
    *   Click **"Save & Close"**.
    *   Console output will show the configuration being saved. The engine will be stopped (if running). The GUI buttons will reset to the "Stopped" state.
    *   To use the *new* bindings, click **"START ENGINE"** again (or use the voice command) to restart the engine with the updated configuration.

6.  **Stopping:**
    *   To stop the AI Commander, press `Ctrl+C` in the terminal where you ran `ai_commander.py`.
    *   To stop the main application, close the GUI window or press `Ctrl+C` in the terminal where you ran `python accessicommand/main.py`.

## Technologies Used

*   Python
*   Tkinter / ttk (for GUI)
*   MediaPipe (for Face and Hand tracking)
*   OpenCV (`cv2`) (for camera access and image manipulation)
*   SpeechRecognition (for voice input and Whisper transcription)
*   PyAudio (for microphone access)
*   PyAutoGUI (for simulating system input)
*   threading (for running detection loops in background)
