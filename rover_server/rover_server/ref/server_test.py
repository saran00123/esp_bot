import socket

HOST = '192.168.1.20'
PORT = 80

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    message = "Hello from Python"
    s.sendall(message.encode())
    print("Sent:", message)
