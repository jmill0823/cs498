import subprocess

# Main file to run all of the files consecutively

# Run the scripts
def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

# Main
if __name__ == "__main__":
    # List of scripts to run
    scripts = ["visualize_data.py", "preprocessor.py", "model_train.py", "model_predict.py"]

    # Loop through the scripts and run each one
    for script in scripts:
        run_script(script)
        print(f"-> {script} finished!")
    print("Program complete!")
