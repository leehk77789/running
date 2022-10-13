import torch
import cv2
#모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#이미지파일 읽고 result에 저장
img = cv2.imread('Untitled.jpeg')
results = model(img)
results.save()

#pandas 를 numpy배열로 바꾼다.
result = results.pandas().xyxy[0].to_numpy()
#6번째 항목의 값이 person인것만 저장
result = [item for item in result if item[6]=='person']
#새로운 이미지를 저장
tmp_img = cv2.imread('Untitled.jpeg')
print(tmp_img.shape)
#cropped 해당 범위만 가져가겠다.
cropped = tmp_img[int(result[0][1]):int(result[0][3]), int(result[0][0]):int(result[0][2])]
print(cropped.shape)
#cropped를 저장
cv2.imwrite('result2.png', cropped)
cv2.rectangle(tmp_img, (int(results.xyxy[0][0][0].item()), int(results.xyxy[0][0][1].item())), (int(results.xyxy[0][0][2].item()), int(results.xyxy[0][0][3].item())), (255,255,255))
cv2.imwrite('result.png', tmp_img)