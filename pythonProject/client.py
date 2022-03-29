import zmq
import cv2
import base64

# zmq 초기화
context = zmq.Context()
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
# 서버와 연결
socket.connect("tcp://localhost:12345")

# 이미지 인코딩
color_img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
color = cv2.imencode('.jpg', color_img)
a = base64.b64encode(color[1]).decode('utf-8').encode('utf-8')

# 이미지 전송
while True:
    socket.send(a)
    message = socket.recv()
    print("Received reply [ %s ]" % (message.decode('utf-8')))
    if message.decode('utf-8') == "Exit":
        break

# 종료 대사 출력
print("종료")
socket.disconnect("tcp://localhost:12345")
