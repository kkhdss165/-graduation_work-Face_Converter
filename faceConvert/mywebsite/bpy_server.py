import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import socket
from create_face import createFaceMesh

HOST = "127.0.0.1"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))

while True:
    server_socket.listen()
    client_socket, addr = server_socket.accept()

    ##받을때까지 대기
    data = client_socket.recv(1024)

    file_name = str(data.decode())
    print("file_name", file_name)
    list = createFaceMesh(file_name).createMesh()

    send_data = ''

    for j in list:
        for i in j:
            send_data += str(i)
            send_data += '/'

    send_data = send_data[:-1]

    client_socket.sendall(send_data.encode())

    client_socket.close()

server_socket.close()