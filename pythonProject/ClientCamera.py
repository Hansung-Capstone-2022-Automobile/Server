import zmq
import cv2
import base64

# 카메라 설정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('[!] video open failed!')

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# zmq 초기화
context = zmq.Context()
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
# 서버와 연결
socket.connect("tcp://localhost:12346")

# 이미지 인코딩
# color_img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

# 이미지 전송
while True:
    ret, frame = cap.read()
    flag, image = cv2.imencode('.jpg', frame)
    encodingImage = base64.b64encode(image).decode('utf-8').encode('utf-8')
    socket.send(encodingImage)
    message = socket.recv()
    print("Received reply [ %s ]" % (message.decode('utf-8')))
    if message.decode('utf-8') == "Exit":
        break

# 종료 대사 출력
print("종료")
socket.disconnect("tcp://localhost:12346")
