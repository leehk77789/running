import torch
import cv2
#모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#이미지파일 읽고 result에 저장
img = cv2.imread('Untitled.jpeg')
print(img.shape)
results = model(img)

print(results.xyxy[0], results.xyxy[0][0][0].item())  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)

tmp_img = cv2.imread('Untitled.jpeg') #이미지 읽어와서 tmp에저장
cv2.rectangle(tmp_img, (int(results.xyxy[0][0][0].item()), int(results.xyxy[0][0][1].item())), (int(results.xyxy[0][0][2].item()), int(results.xyxy[0][0][3].item())), (255,255,255))
#tmp에 네모를 그린다.
cv2.imwrite('result.png', tmp_img)
#네모를 그린 tmp이미지를 저장한다.