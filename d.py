import torch
import cv2
#욜로에는 80개의 모델들이 이미 학습되어있음.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://teamsparta.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F396579ac-a81f-4073-9409-248ba51bc5ea%2FUntitled.jpeg?table=block&id=e70a6c23-18cd-4bc6-8c46-b585967dfd56&spaceId=83c75a39-3aba-4ba4-a792-7aefe4b07895&width=2000&userId=&cache=v2']  # batch of images

results = model(imgs)

results.save() #or.show()