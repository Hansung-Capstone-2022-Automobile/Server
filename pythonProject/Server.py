import time
import zmq
import cv2

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:12345")

while True:
    #  Wait for next request from client
    message = socket.recv()

    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")