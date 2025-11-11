from omni.isaac.kit import SimulationApp
import omni.isaac.urdf
import omni.usd

# Initialize Isaac Sim
sim_app = SimulationApp()

# Path to the URDF file you want to convert
urdf_file_path = "/home/user/rh/src/rh_p12_rn_a_description/urdf/rh_p12_rn_a.urdf"
# Path where you want to save the resulting USD file
usd_file_path = "/home/user/rh/src/rh_p12_rn_a_description/usd/rh_p12_rn_a.usd"

# Create an empty stage
stage = omni.usd.get_context().get_stage()

# Load the URDF file into the USD stage
omni.isaac.urdf.load(urdf_file_path, stage)

# Save the USD file
stage.GetRootLayer().Export(usd_file_path)

print(f"URDF file converted to USD and saved at {usd_file_path}")

