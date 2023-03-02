import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Open the Simulink model
model = 'my_simulink_model'
eng.eval('open_system(model)')

# Get the handle to the External Input block
block_handle = eng.get_param('my_simulink_model/External Input', 'handle')

# Send data to the External Input block
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3x3 array
eng.set_param(block_handle, 'RuntimeObject', data)

# Run the simulation
eng.set_param(model, 'SimulationCommand', 'start')

# Close the Simulink model and MATLAB engine
eng.eval('close_system(model)')
eng.quit()
