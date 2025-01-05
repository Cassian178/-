from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('F:\\人像定位与性别识别\\pythonProject\\runs\\detect\\train3\\weights\\last.pt')
    results=model.train(data='datasets_for_yolo/data.yaml', seed=42, epochs=100, imgsz = 640, device = 0, batch = 4)
