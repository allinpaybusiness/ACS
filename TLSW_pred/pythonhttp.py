# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:34:07 2017

@author: s
"""

import socket

import re

 
HOST, PORT = '', 8888
 
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(10)
print("Serving HTTP on port %s ..." % PORT)
while True:
      client_connection, client_address = listen_socket.accept()
      request = client_connection.recv(1024)
      
    #  request.split("\r\n")

      st = re.split(" /|,|\r\n".encode('utf-8'),request)
      print(st[1])
   #   request.split("\r\n".encode('utf-8')), request)
     
      http_response = """\
      HTTP/1.1 200 OK
     
      Hello, World!
      """
     
      client_connection.sendall(http_response.encode('utf-8'))
      client_connection.close()