import sys, struct
import socket
import struct

# Set the IP address and port of the Simulink block
IP_ADDRESS = '127.0.0.1' # Change to the IP address of your Simulink block
PORT_Send = 1234 # Change to the port number of your Simulink block

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send data to the Simulink block
data = b'A'
try:
    while True:
        sock.sendto(data, (IP_ADDRESS, PORT_Send))

        # Receive data from the Simulink block
        received_data, address = sock.recvfrom(1024)

        # Convert the received data into a numpy array
        # array = struct.unpack('f'*3, received_data)


        # Print the received array
        # print(array)
except KeyboardInterrupt:
    
    # Close the socket
    sock.close()
