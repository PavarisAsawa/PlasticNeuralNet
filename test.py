import os

script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
usd_path = os.path.join(script_dir, "slalom_fixedbody_16dof.usd")  # Build the full path
print(script_dir)