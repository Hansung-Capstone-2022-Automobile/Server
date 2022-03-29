import zmq
import cv2
import base64
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:12345")


print("Start Server")
message = socket.recv() # recieve
# 디코딩
message2 = message.decode('utf-8')
img = base64.b64decode(message2)
color2 = np.asarray(bytearray(img), dtype="uint8")
testImage = cv2.imdecode(color2, 1)

#  이미지 표시
cv2.imshow('test', testImage)
cv2.waitKey()
cv2.destroyAllWindows()
print("Server close")

    #  Send reply back to client