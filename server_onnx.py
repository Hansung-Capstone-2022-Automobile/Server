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


def on_publish(client_mqtt, userdata, mid):
    print("In on_pub callback mid= ", mid)


# 새로운 클라이언트 생성
client_mqtt = mqtt.Client()
# 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_publish(메세지 발행)
client_mqtt.on_connect = on_connect
client_mqtt.on_disconnect = on_disconnect
client_mqtt.on_publish = on_publish
# 로컬 아닌, 원격 mqtt broker에 연결
# address : broker.hivemq.com
# port: 1883 에 연결
client_mqtt.connect('broker.hivemq.com', 1883)
client_mqtt.loop_start()

'''
Camera SERVER
'''
# 서버 ip 주소 및 port 번호
ip = '192.168.86.69'
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

model = "../Duckie.onnx"
sess = onnxruntime.InferenceSession(model)
to_tensor = transforms.ToTensor()
while True:
    # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
    while len(data_buffer) < data_size:
        # 데이터 수신
        data_buffer += client_socket.recv(4096)
    # if count < 10:
    #     count += 1
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
    frame = cv2.resize(frame, (256, 192))
    img = to_tensor(frame)
    Input = np.expand_dims(img, axis=0)
    output = sess.run(["discrete_actions"], {"obs_0": Input, "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})
    # print(output[0][0][0])
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

    cv2.imshow('Frame', frame)
    '''
    MQTT SERVER
    '''
    # 'test/hello' 라는 topic 으로 메세지 발행
    # client_mqtt.loop_stop()

    # # 'q' 키를 입력하면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        message = "exit"
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
