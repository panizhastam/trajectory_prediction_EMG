import socket
import sys

# if len(sys.argv) == 3:
#     # Get "IP address of Server" and also the "port number" from argument 1 and argument 2
#     ip = sys.argv[1]
#     port = int(sys.argv[2])
# else:
#     print("Run like : python3 server.py <arg1:server ip:this system IP 192.168.1.6> <arg2:server port:4444 >")
#     exit(1)

ip = '127.0.0.1'
port_send = 1234

# Create a UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
print("Do Ctrl+c to exit the program !!")
s.connect((ip, port_send))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
port_rec = 1236
ip_rec = '127.0.0.2'
server_address = (ip_rec, port_rec)
sock.bind(server_address)
try:
    while True:
        send_data = "2"
        s.sendto(send_data.encode('utf-8'), (ip, port_send))
        print("\n\n 1. Client Sent : ", send_data, "\n\n")
        # from_server = s.recv(4096)
        received_data, address = sock.recvfrom(4096)
        print("\n\n 2. Client received : ", received_data.decode('utf-8'), "\n\n")
        
except KeyboardInterrupt:
    
    # Close the socket
    s.close()
