import cv2

def loadImage(file_path):
    # 이미지 읽기
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # 이미지 인코딩
    # retval : 압축 결과(True/False), buf :인코딩된 이미지
    retval, buf = cv2.imencode('.webp',
                               src,
                               [cv2.IMWRITE_WEBP_QUALITY, 100])
    return src, retval, buf

src, retval, buf = loadImage('test.jpg')

# 이미지 디코딩
webp_img = cv2.imdecode(buf, 1)

# 원본, 인코딩, 디코딩 이미지 화면 출력
cv2.imshow('src', src)
cv2.imshow('encoding', buf)
cv2.imshow('decoding', webp_img)

# 화면 출력창 대기/닫기
cv2.waitKey()
cv2.destroyAllWindows()