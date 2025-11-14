import io
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import app

class AppGUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Movie Indexing Application")
        self.root.geometry("600x400")
        
        # variables
        self.query_text = tk.StringVar()
        self.custom_text = tk.StringVar()
        self.use_custom_text = tk.BooleanVar(value=False)

        
        self.setup_ui()
        
        
    def setup_ui(self):
        #main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        #configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Give resizing weight to column 1 (the entry box)
        main_frame.columnconfigure(1, weight=1) 
        
        # Give the resizing weight to row 4 (the output area)
        main_frame.rowconfigure(4, weight=1) 
        
        #title label
        title_label = ttk.Label(main_frame, text="Movie Indexing Application", font=("Helvetica", 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0,10))    
        
        
        #qury label and entry
        query_label = ttk.Label(main_frame, text="Enter Query:", font=("Arial", 10, "bold"))
        
        # Stick the label to the East (right) of its cell.
        query_label.grid(row=1, column=0, sticky=tk.E, padx=5)
        
        query_entry = ttk.Entry(
                        main_frame,
                        textvariable=self.query_text,
                        width=50
                        )
        query_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        
        # Output area
        ttk.Label(main_frame, text="Results:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=(tk.W, tk.N), pady=5, padx=5
        )
        
        self.output_text = scrolledtext.ScrolledText(
            main_frame,
            width=80,
            height=20,
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        # Use columnspan=2 to make the text area span both columns
        self.output_text.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
         # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.run_button = ttk.Button(
            button_frame,
            text="Search",
            command=self.run_app,
            width=20
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Output",
            command=self.clear_output,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Exit",
            command=self.root.quit,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Initialize custom text widget (will be set in on_lab_change)
        self.custom_text_widget = None
    
    
    
    def run_app(self):
        """Run the selected lab in a separate thread."""
        # Disable run button
        self.run_button.config(state="disabled")
        
        # Clear output
        self.clear_output()
        
        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._run_lab_thread)
        thread.daemon = True
        thread.start()
    
    def _run_lab_thread(self):
        """Thread target for running the lab."""
        try:
            
            query = self.query_text.get().strip()
            if not query:
                self.write_output("Error: Please enter movie/mvoies comma-seperated.\n")
                self.root.after(0, lambda: self.run_button.config(state="normal"))
                return
            
            
            # Redirect stdout to capture prints
            original_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            
            try:
                app.run_app_file(query)
                
                # get captured output
                output = captured_output.getvalue()
                
                #write to gui
                self.root.after(0, lambda: self.write_output(output))
            
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
                
        except Exception as e: # pylint: disable=broad-except
            error_msg = f"Error: {str(e)}\n"
            self.root.after(0, lambda: self.write_output(error_msg))
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.config(state="normal"))
                
                

    
    def write_output(self, message):
        """Write message to the output text widget."""
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)  # Auto-scroll to the end
    
    def clear_output(self):
        """Clear the output text widget."""
        self.output_text.delete(1.0, tk.END)
    
def main():
    root = tk.Tk()
    app_gui = AppGUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()