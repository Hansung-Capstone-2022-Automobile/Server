# 필요한 패키지 import
# Camera
import socket  # 소켓 프로그래밍에 필요한 API를 제공하는 모듈
import struct  # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle  # 바이트(bytes) 형식의 데이터 변환 모듈
import cv2  # OpenCV(실시간 이미지 프로세싱) 모듈
# MQTT
import numpy as np
import onnxruntime
import paho.mqtt.client as mqtt
import torchvision.transforms as transforms
import math

# import json

'''
MQTT SERVER
'''


def on_connect(client_mqtt, userdata, flags, rc):
    # 연결이 성공적으로 된다면 완료 메세지 출력
    if rc == 0:
        print("completely connected")
    else:
        print("Bad connection Returned code=", rc)


# 연결이 끊기면 출력
def on_disconnect(client_mqtt, userdata, flags, rc=0):
    print(str(rc))


# def on_publish(client_mqtt, userdata, mid):
#     print("In on_pub callback mid= ", mid)

def calc_length(h, w):
    return math.sqrt((h - 191) ** 2 + (w - 127) ** 2)
# 새로운 클라이언트 생성
client_mqtt = mqtt.Client()
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_publish(메세지 발행)
client_mqtt.on_connect = on_connect
client_mqtt.on_disconnect = on_disconnect
# client_mqtt.on_publish = on_publish
# 로컬 아닌, 원격 mqtt broker에 연결
# address : broker.hivemq.com
# port: 1883 에 연결
client_mqtt.connect('broker.hivemq.com', 1883)
client_mqtt.loop_start()

'''
Camera SERVER
'''
# 서버 ip 주소 및 port 번호
ip = '192.168.144.125'
port = 50001

# 소켓 객체 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 소켓 주소 정보 할당
server_socket.bind((ip, port))

# 연결 리스닝(동시 접속) 수 설정
server_socket.listen(10)

print('클라이언트 연결 대기')

# 연결 수락(클라이언트 (소켓, 주소 정보) 반환)
client_socket, address = server_socket.accept()
print('클라이언트 ip 주소 :', address[0])

# 수신한 데이터를 넣을 버퍼(바이트 객체)
data_buffer = b""

# calcsize : 데이터의 크기(byte)
# - L : 부호없는 긴 정수(unsigned long) 4 bytes
data_size = struct.calcsize("L")
# count = 0

model = "./Duckie.onnx"
# model = "./DuckieWithoutCamera.onnx"
sess = onnxruntime.InferenceSession(model)
to_tensor = transforms.ToTensor()

'''
이미지 라인 디텍션
'''
width = 256
half_width = 128
height = 192

resize = transforms.Resize([height, width])

lineV = np.ones((1, half_width)) # (1,320) 즉, 180도를 측정할 직선 성분
lineH = np.ones((height, 1)) # (480,1) 즉, 90도를 측정할 직선 성분
dim1 = np.zeros((height-1, half_width))
dim2 = np.zeros((height, half_width)) # 우측 빈 절반
dim3 = np.block([[dim1], [lineV]]) # (479,320) (1,320) -> (480,320) 직선
dim4 = np.eye(half_width) # (320,320) 대각 선분만 1
dim5 = np.rot90(dim4)
dim6 = np.zeros((height-half_width, half_width)) # dim3 윗 부분을 채워 줄 부분
dim7 = np.zeros((height, half_width)) # 우측 빈 절반
dim8 = np.zeros((height, half_width-1)) # 중앙 직선을 그리기 위해 -1
dim9 = np.vstack([dim6,dim4]) # 135도 직선 그림
dim10 = np.vstack([dim6, dim5]) # 135도 직선 그림

mask1 = np.block([dim3, dim2]) # (480,640) -좌측:180도 -우측:아무것도 없는 배경
mask2 = np.block([dim9, dim2]) # (480,640) -좌측:135도 -우측:아무것도 없는 배경
mask3 = np.block([dim8, lineH, dim7]) # (480,640) -중앙 직선
mask4 = np.block([dim7, dim10]) # (480,640) -좌측:아무것도 없는 배경 -우측:45도
mask5 = np.block([dim2, dim3])

# count = 0

while True:
    # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
    while len(data_buffer) < data_size:
        # 데이터 수신
        data_buffer += client_socket.recv(4096)
    # if count < 100000:
    #     count += 1
    #     print(count)
    #     continue
    # count = 0

    # 버퍼의 저장된 데이터 분할
    packed_data_size = data_buffer[:data_size]
    data_buffer = data_buffer[data_size:]

    # struct.unpack : 변환된 바이트 객체를 원래의 데이터로 변환
    # - > : 빅 엔디안(big endian)
    #   - 엔디안(endian) : 컴퓨터의 메모리와 같은 1차원의 공간에 여러 개의 연속된 대상을 배열하는 방법
    #   - 빅 엔디안(big endian) : 최상위 바이트부터 차례대로 저장
    # - L : 부호없는 긴 정수(unsigned long) 4 bytes
    frame_size = struct.unpack(">L", packed_data_size)[0]

    # 프레임 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
    while len(data_buffer) < frame_size:
        # 데이터 수신
        data_buffer += client_socket.recv(4096)

    # 프레임 데이터 분할
    frame_data = data_buffer[:frame_size]
    data_buffer = data_buffer[frame_size:]

    # print("수신 프레임 크기 : {} bytes".format(frame_size))

    # loads : 직렬화된 데이터를 역직렬화
    # - 역직렬화(de-serialization) : 직렬화된 파일이나 바이트 객체를 원래의 데이터로 복원하는 것
    frame = pickle.loads(frame_data)

    # imdecode : 이미지(프레임) 디코딩
    # 1) 인코딩된 이미지 배열
    # 2) 이미지 파일을 읽을 때의 옵션
    #    - IMREAD_COLOR : 이미지를 COLOR로 읽음

    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # 프레임 출력
    # cv2.imshow('Frame', frame)
    img_resize = cv2.resize(frame, (width, height))
    canny = cv2.Canny(cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY), 30, 70)
    frame = cv2.resize(frame, (width, height))
    img = to_tensor(frame)
    Input = np.expand_dims(img, axis=0)

    len1 = 0
    len2 = 0
    len3 = 0
    len4 = 0
    len5 = 0

    if np.transpose(np.nonzero(mask1 * canny)).size > 0:
        len1 = calc_length(np.transpose(np.nonzero(mask1 * canny))[-1][0],
                           np.transpose(np.nonzero(mask1 * canny))[-1][1]) / (calc_length(191, 0))

    if np.transpose(np.nonzero(mask2 * canny)).size > 0:
        len2 = calc_length(np.transpose(np.nonzero(mask2 * canny))[-1][0],
                           np.transpose(np.nonzero(mask2 * canny))[-1][1]) / (
                   calc_length(191 - 127, 0))

    if np.transpose(np.nonzero(mask3 * canny)).size > 0:
        len3 = calc_length(np.transpose(np.nonzero(mask3 * canny))[-1][0],
                           np.transpose(np.nonzero(mask3 * canny))[-1][1]) / (
                   calc_length(0, 127))

    if np.transpose(np.nonzero(mask4 * canny)).size > 0:
        len4 = calc_length(np.transpose(np.nonzero(mask4 * canny))[-1][0],
                           np.transpose(np.nonzero(mask4 * canny))[-1][1]) / (
                   calc_length(191-127, 255))

    if np.transpose(np.nonzero(mask5 * canny)).size > 0:
        len5 = calc_length(np.transpose(np.nonzero(mask5 * canny))[0][0],
                           np.transpose(np.nonzero(mask5 * canny))[0][1]) / (
                   calc_length(191, 255))

    obs_1 = np.array([[len1, len2, len3, len4, len5]]).astype(np.float32)

    output = sess.run(["discrete_actions"],
                      {"obs_0": Input, "obs_1": obs_1,
                       "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})
    print(obs_1)
    # output = sess.run(["discrete_actions"],
    #                   {"obs_0": obs_1,
    #                    "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})

    message = "stop"
    if output[0][0][0] == 0:
        message = "stop"
    elif output[0][0][0] == 1:
        message = "go"
    elif output[0][0][0] == 2:
        message = "back"
    elif output[0][0][0] == 3:
        message = "left"
    elif output[0][0][0] == 4:
        message = "right"

    cv2.imshow('canny', canny)
    cv2.imshow('img_resize', img_resize)

    '''
    MQTT SERVER
    '''
    # 'test/hello' 라는 topic 으로 메세지 발행
    # client_mqtt.loop_stop()

    # # 'q' 키를 입력하면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        message = "exit"
        client_mqtt.publish('test/hello', message, 1)
        break

    client_mqtt.publish('test/hello', message, 1)

# 소켓 닫기
client_socket.close()
server_socket.close()
# MQTT 닫기
client_mqtt.disconnect()
print('연결 종료')

# 모든 창 닫기
cv2.destroyAllWindows()