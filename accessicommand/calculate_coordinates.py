# helper_get_coords.py
import pyautogui
import time

# --- Instructions ---
print("Move your mouse cursor over the desired target (e.g., the center of a button)")
print("in the AccessiCommand window in the next 5 seconds.")
print("The coordinates will be printed here.")
# ------------------

time.sleep(5) # Gives you 5 seconds to move your mouse

try:
    x, y = pyautogui.position() # Get the current mouse position
    print(f"\n--- Found Mouse Position ---")
    print(f"X={x} Y={y}")
    print("----------------------------")
    print("Copy these coordinates and paste them into the BUTTON_COORDS dictionary in your ui_commander.py file.")

except Exception as e:
    print(f"ERROR: Could not get mouse position: {e}")
    print("Please ensure pyautogui is installed correctly.")