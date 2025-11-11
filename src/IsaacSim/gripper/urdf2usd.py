from omni.isaac.kit import SimulationApp
import omni.usd
import omni.isaac.urdf

# Start the simulation app
sim_app = SimulationApp()

# Path to your generated URDF (after processing the .xacro)
#urdf_file_path = "/path/to/your/generated_robot.urdf"
urdf_file_path = "/home/user/rh/src/rh_p12_rn_a_description/urdf/rh_p12_rn_a.urdf"


# Create an empty USD stage
stage = omni.usd.get_context().get_stage()

# Load the URDF model into the USD stage
omni.isaac.urdf.load(urdf_file_path, stage)

# Export the stage to a USD file
#usd_file_path = "/path/to/save/robot_model.usd"
usd_file_path = "/home/user/rh/src/rh_p12_rn_a_description/usd/rh_p12_rn_a.usd"
 
stage.GetRootLayer().Export(usd_file_path)

print(f"URDF successfully converted to USD and saved at {usd_file_path}")

