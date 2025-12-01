import json
import os

HISTORY_FILE = "conversation_history.json"

def load_conversation_history():
    """Loading conversation history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, ValueError):
            # File exists but is empty or corrupted — reset to empty history
            return []
    return []

def save_conversation_history(history):
    """Saving conversation history to JSON file."""
    #Load existing history
    current_history = load_conversation_history()
    # duplicate check (case-insensitive)
    existing_entries = {movie.lower() for movie in current_history if isinstance(movie, str)}
    updated = False

    # Accept both single string or list of strings
    new_entries = history if isinstance(history, list) else [history]

    for entry in new_entries:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip()
        if cleaned and cleaned.lower() not in existing_entries:
            current_history.append(cleaned)
            existing_entries.add(cleaned.lower())
            updated = True

    if updated:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as file:
                json.dump(current_history, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    return current_history

def clear_conversation_history():
    """Clearing the conversation history file."""
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            print("Conversation history cleared.")
        except Exception as e:
            print(f"Error clearing conversation history: {e}")
    else:
        print("No conversation history to clear.")
                  
