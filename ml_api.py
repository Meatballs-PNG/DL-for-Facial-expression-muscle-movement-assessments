from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Dropout, Input, Multiply, Reshape, MaxPooling2D, Conv2D, Add, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File
import base64
import random
import json

# json檔路徑
json_path = '/home/wei_jai/test.json'

# 模型自訂層
# XceptionLayer 
class XceptionLayer(Layer):
    def __init__(self, **kwargs):
        super(XceptionLayer, self).__init__(**kwargs)
        self.xception = Xception(weights='imagenet', include_top=False)

    def call(self, inputs, input_shape=(100, 100, 3)):
        return self.xception(inputs)

    def get_config(self):
        config = super().get_config()  
        return config

# CBAMLayer 
class CBAMLayer(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # 通道注意力
        self.channel_avg_pool = GlobalAveragePooling2D()
        self.channel_max_pool = MaxPooling2D()
        self.channel_dense_1 = Dense(input_shape[-1] // self.reduction_ratio, activation='relu')
        self.channel_dense_2 = Dense(input_shape[-1], activation='sigmoid')

        # 空间注意力
        self.spatial_conv = Conv2D(1, kernel_size=(7, 7), padding='same', activation='sigmoid')

    def call(self, inputs):
        # 通道注意力
        channel_avg_pool = self.channel_avg_pool(inputs)
        channel_max_pool = self.channel_max_pool(inputs)
        channel_avg_pool = Reshape((1, 1, -1))(channel_avg_pool)
        channel_max_pool = Reshape((1, 1, -1))(channel_max_pool)
        channel_avg_pool = self.channel_dense_1(channel_avg_pool)
        channel_max_pool = self.channel_dense_1(channel_max_pool)
        channel_avg_pool = self.channel_dense_2(channel_avg_pool)
        channel_max_pool = self.channel_dense_2(channel_max_pool)
        channel_attention = Add()([channel_avg_pool, channel_max_pool])
        channel_attention = Multiply()([inputs, channel_attention])

        # 空间注意力
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_attention)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_attention)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.spatial_conv(concat)
        spatial_attention = Multiply()([channel_attention, spatial_attention])

        return spatial_attention

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config
    
app = FastAPI()
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者设置为你允许的域名，如 ["https://yourwebsite.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # 添加GET方法
    allow_headers=["*"],
)

# 载入模型
# 指定模型路徑
# Assuming you have multiple models saved at different paths
MODEL_PATHS = {
    "a": "/home/wei_jai/model.keras",
    "Model 2": "/home/wei_jai/dog_cat.keras",
    "Model 3": "/home/wei_jai/model.keras",
}

# Function to load the model dynamically based on label
def load_model_by_label(label):
    if label in MODEL_PATHS:
        return tf.keras.models.load_model(MODEL_PATHS[label], custom_objects={'XceptionLayer': XceptionLayer, 'CBAMLayer': CBAMLayer})
    else:
        raise ValueError(f"Invalid model label: {label}")

# 初始化MediaPipe的面部檢測器和面部關鍵點檢測器
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# 定義全局變量 original_img
original_img = None

# 初始化MediaPipe的面部檢測器和面部關鍵點檢測器
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


# 定義特徵點連線顏色
def connect_points(image, coordinates, color):
    for i in range(len(coordinates) - 1):
        cv2.line(image, coordinates[i], coordinates[i + 1], color, 2)

# 定義肌肉部位座標計算函數
def detect_face_landmarks(image, results, feature_points_keys):
    # 從JSON檔載入自訂特徵點資訊和所占比例
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        if results.multi_face_landmarks:
            for feature_points_key in feature_points_keys:
                # 創建列表儲存計算後的座標
                final_coordinates = []

                for item in data[feature_points_key]:
                    scale_factors = [float(factor) for factor in item["p"].split()]  # 比例因子列表
                    feature_point_indices = [int(idx) for idx in item["v"].split()]  # 特徵點編號列表

                    # 初始化變數
                    summed_x = 0
                    summed_y = 0
                    summed_z = 0

                    # 將指定特徵點之相對座標乘以比例因子後計算出所需肌肉部位座標
                    for idx, factor in zip(feature_point_indices, scale_factors):
                        landmark = results.multi_face_landmarks[0].landmark[idx]
                        x, y, z = landmark.x, landmark.y, landmark.z
                        x_image = int(x * image.shape[1] * factor)  # 轉換為圖像座標系統的 x 座標並乘上比例因子
                        y_image = int(y * image.shape[0] * factor)  # 轉換為圖像座標系統的 y 座標並乘上比例因子
                        z_image = int(z * factor)  # z 座標乘上比例因子
                        summed_x += x_image
                        summed_y += y_image
                        summed_z += z_image

                    # 將最終座標儲存到列表中
                    final_coordinates.append((summed_x, summed_y))

                # 使列表中的最後一個特徵點與第一個特徵點重合
                final_coordinates.append(final_coordinates[0])
                # 使用隨機色彩連線
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # 呼叫連線函數
                connect_points(image, final_coordinates, color)

    return image

@app.post("/emotion_recognition")
async def emotion_recognition(file: UploadFile = File(...), model_label: str = Form(...)):
    global json_path  # 使用全局變量 json_path
    # Dynamically load the selected model based on label
    try:
        model = load_model_by_label(model_label)
    except ValueError as e:
        return {"error": str(e)}
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = img.copy()  # 備份原始圖像
    
    # 進行面部網格檢測
    results = face_mesh.process(original_img)
    
    # 如果未偵測到任何人臉，返回訊息
    if not results.multi_face_landmarks:
        return {"result": "未偵測到人臉", "muresult": ""}
    
    # 如果有偵測到人臉，執行表情預測
    else:
        # 調整圖像大小以匹配模型輸入大小
        img = cv2.resize(img, (100, 100))
        img_array = np.expand_dims(img, axis=0)  # 添加批次維度
        img_array = img_array / 255.0  # 正規化像素值
        
        # 進行預測
        predictions = model.predict(img_array)
        
        # 解碼預測結果
        predicted_class = np.argmax(predictions)
        
        # 根據預測結果處理圖像
        if predicted_class == 0:
            # 處理並顯示m1部位
            processed_image = detect_face_landmarks(original_img, results, ['m2','m4','m7','m17','m21'])
            emotion_result = '生氣'
            mu_result = '降眉肌、皺眉肌、眼輪匝肌、口輪匝肌、頦肌'
        elif predicted_class == 1:
            # 處理並顯示m2和m3部位
            processed_image = detect_face_landmarks(original_img, results, ['m7','m12'])
            emotion_result = '快樂'
            mu_result = '眼輪匝肌、顴大肌'
        elif predicted_class == 2:
            processed_image = original_img
            emotion_result = '無表情'
            mu_result = '無'
        # 將處理後的圖像編碼為Base64格式
        _, img_encoded = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(img_encoded).decode()  # 獲取圖像的Base64編碼
        
        # 返回圖像和結果
        return {"image": img_base64, "result": emotion_result, "muresult": mu_result}
@app.get("/web1", response_class=HTMLResponse)
async def web1():
    return """   
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>主頁</title>
  <link rel="icon" href="https://i.postimg.cc/C5JRhJ2Q/image.png" type="image/png">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet"> <!-- 添加 Bootstrap 图标的 CSS 链接 -->
  <style>
   body {
  background-color:#F4F4F4; /* 设置背景颜色为灰色，您可以根据需要修改颜色值 */
}
    .button-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .circle-btn {
      position: relative;
      width: 45px;
      height: 45px;
      border-radius: 50%;
      border: none;
      cursor: pointer;
      transition: background-color 0.3s ease;
      background-color: #ffffff;
    }

    .circle-btn i {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }

    .circle-btn:hover {
      background-color: #f0f0f0;
    }

    .circle-btn:active {
      background-color: #808080;
    }

    #menuBtn i {
      font-size: 24px;
    }

    #settingBtn i {
      font-size: 24px;
    }

    #menu {
  position: fixed; /* 將選單設置為固定位置 */
  top: 0;
  left: -400px;
  width: 240px;
  height: 100vh;
  background-color: #ede9e8;
  transition: left 0.3s ease;
  padding: 20px;
  box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.3);
 z-index: 2;
}

    .menu-button-container {
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .square-btn {
      width: 100%;
      height: 40px;
      border: none;
      border-radius: 10px;
      background-color: #ede9e8;
      color: #9e9092;
      margin-bottom: 10px;
      cursor: pointer;
      transition: background-color 0.3s ease, color 0.3s ease;
      text-align: left;
      font-size: 14px;
    }

    .square-btn:hover {
      background-color: #ffffff;
    }

    .square-btn:active {
      background-color: #808080;
      color: #ffffff;
    }

    #image-container {
      width: calc(100% - 400px); /* 讓圖像框靠右 */
      height: 100%;
      display: flex;
      justify-content: flex-end; /* 靠右對齊 */
      align-items: center;
    }

    #displayed-image {
      max-width: 100%;
      max-height: 100%;
    }
  </style>
</head>
<body>
  <div class="button-container">
    <!-- 左侧按钮 -->
    <button class="circle-btn" id="menuBtn"><i class="bi bi-list"></i></button> <!-- 使用 Bootstrap 图标 bi-list -->
    <!-- 右侧按钮 -->
    <button class="circle-btn" id="settingBtn"><i class="bi bi-gear"></i></button> <!-- 使用 Bootstrap 图标 bi-gear -->
  </div>

  <!-- 隐藏的菜单 -->
  <div id="menu">
    <div class="menu-button-container">
      <!-- 添加四边形按钮 -->
      <button class="square-btn" id="homeBtn">主頁</button>
      <button class="square-btn" id="emotionBtn">表 情 辨 識 & 顯 示 動 作 肌 群</button>
      <button class="square-btn" id="muscleBtn">臉 部 肌 肉 位 置 說 明</button>
      <button class="square-btn" id="aboutBtn">關 於 我 們</button>
    </div>
  </div>

  <!-- 图片显示区域 -->
  <div id="image-container">
    <img src="" alt="" id="displayed-image">
  </div>

  <script>
    document.addEventListener('DOMContentLoaded', function() {
      const menuBtn = document.getElementById('menuBtn');
      const settingBtn = document.getElementById('settingBtn');
      const menu = document.getElementById('menu');
      const homeBtn = document.getElementById('homeBtn');
      const emotionBtn = document.getElementById('emotionBtn');
      const muscleBtn = document.getElementById('muscleBtn');
      const aboutBtn = document.getElementById('aboutBtn');
      const displayedImage = document.getElementById('displayed-image');

      // 点击menu按钮
      menuBtn.addEventListener('click', function(event) {
        menu.style.left = '0'; // 显示菜单
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 点击setting按钮
      settingBtn.addEventListener('click', function(event) {
        // 在这里添加setting按钮的事件处理逻辑
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 点击其他地方隐藏菜单
      document.addEventListener('click', function(event) {
        const target = event.target;
        if (!menu.contains(target) && target !== menuBtn) {
          menu.style.left = '-400px'; // 隐藏菜单
        }
      });

      // 点击"主頁"
      homeBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
      });

      // 点击"表情辨識 & 顯示動作肌群"
      emotionBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web2'; // 替换为相应页面的 URL
      });
   // 点击"臉部肌肉位置說明"
      muscleBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web3'; // 替换为相应页面的 URL
      });
      // 点击"關於我們"
      aboutBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
      });
    });
  </script>
</body>
</html>
"""

@app.get("/web2", response_class=HTMLResponse)
async def web2():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>表情辨識 & 顯示動作肌群</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet"> <!-- 添加 Bootstrap 图标的 CSS 链接 -->
   <style>
  body {
    background-color:#F4F4F4; /* 设置背景颜色为灰色，您可以根据需要修改颜色值 */
  }
    .button-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }
    .circle-btn {
      position: relative;
      width: 45px;
      height: 45px;
      border-radius: 50%;
      border: none;
      cursor: pointer;
      transition: background-color 0.3s ease;
      background-color: #ffffff;
    }

    .circle-btn i {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }

    .circle-btn:hover {
      background-color: #f0f0f0;
    }

    .circle-btn:active {
      background-color: #808080;
    }

    #menuBtn i {
      font-size: 24px;
    }

    #settingBtn i {
      font-size: 24px;
    }

    #menu {
      position: fixed; /* 將選單設置為固定位置 */
      top: 0;
      left: -400px;
      width: 240px;
      height: 100vh;
      background-color: #ede9e8;
      transition: left 0.3s ease;
      padding: 20px;
      box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.3);
      z-index: 2;
    }

    .menu-button-container {
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .square-btn {
      width: 100%;
      height: 40px;
      border: none;
      border-radius: 10px;
      background-color: #ede9e8;
      color: #9e9092;
      margin-bottom: 10px;
      cursor: pointer;
      transition: background-color 0.3s ease, color 0.3s ease;
      text-align: left;
      font-size: 14px;
    }

    .square-btn:hover {
      background-color: #ffffff;
    }

    .square-btn:active {
      background-color: #808080;
      color: #ffffff;
    }

    .container {
      width: 100%;
      text-align: center;
      margin-top: 100px;
    }

    .box-container {
      display: inline-block;
      margin: 40px 100px; /* 添加了边距以分隔两个方框 */
      position: relative;
      vertical-align: top; /* 將元素置頂對齊 */
      font-size: 20px;
      top: -60px; 
      z-index: 1;
    }

    .box {
      background-color: #ffffff; /* 设置背景颜色为灰色，您可以根据需要修改颜色值 */
      width: 500px;
      height: 500px;
      border: 4px solid #000;
      border-radius: 10px;
      box-sizing: border-box;
      margin-bottom: 20px;
      position: relative;
    }

    .box img{
      width: 100%;
      height: 100%;
      object-fit: cover; /* 填满容器并保持纵横比 */
      position: absolute;
      top: 0;
      left: 0;                                                                                                                                             
    }

    .box p {
      position: absolute;
      top: -60px; /* 文字位置调整为负数以确保在框外 */
      left: 50%;
      transform: translateX(-50%);
      background-color: #fff;
      padding: 10px;
      border: 2px solid #000;
      border-bottom: none;
    }

    .button-container2 {
      text-align: center;
      position: absolute;
      bottom: -40px; /* 按钮位置调整为负数以确保在框外 */
      left: 50%;
      transform: translateX(-50%);
    }

    .button-54 {
      font-family: "Open Sans", sans-serif;
      font-size: 16px;
      letter-spacing: 2px;
      text-decoration: none;
      text-transform: uppercase;
      color: #000;
      cursor: pointer;
      border: 3px solid;
      padding: 0.25em 0.5 em;
      box-shadow: 1px 1px 0px 0px, 2px 2px 0px 0px, 3px 3px 0px 0px, 4px 4px 0px 0px, 5px 5px 0px 0px;
      position: relative;
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
    }

    .button-54:active {
      box-shadow: 0px 0px 0px 0px;
      top: 5px;
      left: 5px;
    }

    @media (min-width: 768px) {
      .button-54 {
        padding: 0.25em 0.75em;
      }
    }

    #video-container {
      width: 500px;
      display: inline-block;
      height: 500px;
      border: 4px solid #000;
      border-radius: 10px;
      box-sizing: border-box;
      margin-bottom: 20px;
      margin: 0 100px; /* 添加了边距以分隔两个方框 */
      overflow: hidden; /* 确保视频不会溢出方框 */
      position: relative;
    }

    #resultBoximg{
      transform: scaleX(-1); 
      filter: brightness(100%); /* 调整亮度的滤镜，数值可以根据需求调整 */
    }

    #video {
      width: 100%;
      height: 100%;
      transform: scaleX(-1); 
      filter: brightness(100%); /* 调整亮度的滤镜，数值可以根据需求调整 */
      object-fit: cover; /* 填满容器并保持纵横比 */
    }
  </style>
</head>
<body>
  <div class="button-container">
    <!-- 左侧按钮 -->
    <button class="circle-btn" id="menuBtn"><i class="bi bi-list"></i></button> <!-- 使用 Bootstrap 图标 bi-list -->
    <!-- 右侧按钮 -->
    <button class="circle-btn" id="settingBtn"><i class="bi bi-gear"></i></button> <!-- 使用 Bootstrap 图标 bi-gear -->
  </div>

  <!-- 隐藏的菜单 -->
  <div id="menu">
    <div class="menu-button-container">
      <!-- 添加四边形按钮 -->
      <button class="square-btn" id="homeBtn">主頁</button>
      <button class="square-btn" id="emotionBtn">表 情 辨 識 & 顯 示 動 作 肌 群</button>
      <button class="square-btn" id="muscleBtn">臉 部 肌 肉 位 置 說 明</button>
      <button class="square-btn" id="aboutBtn">關 於 我 們</button>
    </div>
  </div>

<div class="container">
  <div class="box-container">
    <p>請將臉部至於框內</p>
    <div class="box"><video id="video" autoplay="true" playsinline></video></div>
    <div class="button-container2">
      <button class="button-54" id="capture-btn" role="button">擷取</button>
    </div>
  </div>

  <div class="box-container">
    <p>辨識結果為：<span id="resultBox"></span></p>
    <div class="box" id="resultBoximg"></div>
    <p>使用到的肌肉：<span id="resultmuBox"></span></p>
  </div>
</div>

 <script>
document.addEventListener('DOMContentLoaded', function() {
    const menuBtn = document.getElementById('menuBtn');
    const settingBtn = document.getElementById('settingBtn');
    const menu = document.getElementById('menu');
    const homeBtn = document.getElementById('homeBtn');
    const emotionBtn = document.getElementById('emotionBtn');
    const muscleBtn = document.getElementById('muscleBtn');
    const aboutBtn = document.getElementById('aboutBtn');
    const displayedImage = document.getElementById('displayed-image');
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('capture-btn');

    // Load overlay image with CORS
    const overlayImage = new Image();
    overlayImage.crossOrigin = "anonymous"; // This tells the browser to use CORS
    overlayImage.src = 'https://i.ibb.co/vvDFc54/image.png';
    let overlayImageLoaded = false;

    overlayImage.onload = () => {
        overlayImageLoaded = true;
        console.log('Overlay image loaded');
    };

    // 获取摄像头视频流并显示在 video 元素中
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            console.error('Error accessing the camera:', err);
        });

    // 捕获图像并发送到服务器
    captureBtn.addEventListener('click', () => {
        if (!overlayImageLoaded) {
            console.error('Overlay image is not loaded yet');
            return;
        }
        const resultBoximg = document.getElementById('resultBoximg');
        resultBoximg.innerHTML = '';
        const canvas = document.createElement('canvas');
        const size = Math.min(video.videoWidth, video.videoHeight); // 截取正方形的尺寸

        canvas.width = size;
        canvas.height = size; // 正方形

        const ctx = canvas.getContext('2d');
        const x = (video.videoWidth - size) / 2; // 计算截取的起点坐标
        const y = (video.videoHeight - size) / 2;

        ctx.drawImage(video, x, y, size, size, 0, 0, size, size); // 绘制截取的部分到正方形 Canvas 上

        // Draw overlay image onto canvas
        ctx.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);

        // 将图像转换为 Blob 对象
        canvas.toBlob(blob => {
            // 创建表单数据对象
            const formData = new FormData();
            formData.append('file', blob, 'captured_image.jpg'); // 图像文件名，修改为 JPEG 格式

            // 发送 POST 请求
            fetch("https://1e1c-210-59-96-137.ngrok-free.app/emotion_recognition", {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // 处理返回的图像数据
                if (data.image) {
                    const imgBase64 = data.image;
                    // 创建一个 img 元素
                    const imgElement = document.createElement('img');
                    // 设置图像的 Base64 编码作为 src 属性
                    imgElement.src = 'data:image/jpeg;base64,' + imgBase64.replace(/"/g, '');

                    // 将图像元素添加到 resultBoximg 中
                    const resultBoximg = document.getElementById('resultBoximg');
                    resultBoximg.innerHTML = '';  // 清空 resultBoximg 中的内容
                    resultBoximg.appendChild(imgElement);  // 将图像元素添加到 resultBoximg 中
                }

                // 处理返回的辨识结果
                const predictionResult = data.result;
                // 将辨识结果显示在页面上
                const resultBoxText = document.getElementById('resultBox');
                resultBoxText.innerHTML = predictionResult;

                // 处理返回的辨识结果
                const munResult = data.muresult;
                // 将辨识结果显示在页面上
                const resultmuBoxText = document.getElementById('resultmuBox');
                resultmuBoxText.innerHTML = munResult;
                
            })
            .catch(error => console.error('Error receiving image:', error));

        }, 'image/jpeg'); // 修改图像格式为 JPEG
    });

    // 点击menu按钮
    menuBtn.addEventListener('click', function(event) {
        menu.style.left = '0'; // 显示菜单
        event.stopPropagation(); // 阻止事件冒泡
    });

    // 点击setting按钮
    settingBtn.addEventListener('click', function(event) {
        // 在这里添加setting按钮的事件处理逻辑
        event.stopPropagation(); // 阻止事件冒泡
    });

    // 点击其他地方隐藏菜单
    document.addEventListener('click', function(event) {
        const target = event.target;
        if (!menu.contains(target) && target !== menuBtn) {
            menu.style.left = '-400px'; // 隐藏菜单
        }
    });

    // 点击"主頁"
    homeBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
    });

    // 点击"表情辨識 & 顯示動作肌群"
    emotionBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web2'; // 替换为相应页面的 URL
    });
    // 点击"臉部肌肉位置說明"
    muscleBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web3'; // 替换为相应页面的 URL
    });
    // 点击"關於我們"
    aboutBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
    });
});
</script>

</body>
</html>

"""
@app.get("/web3", response_class=HTMLResponse)
async def web3():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>臉部肌肉位置說明</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet"> <!-- 添加 Bootstrap 图标的 CSS 链接 -->
  <style>
   body {
  background-color:#F4F4F4; /* 设置背景颜色为灰色，您可以根据需要修改颜色值 */
}
     .button-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }


    .circle-btn {
      position: relative;
      width: 45px;
      height: 45px;
      border-radius: 50%;
      border: none;
      cursor: pointer;
      transition: background-color 0.3s ease;
      background-color: #ffffff;
    }

    .circle-btn i {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }

    .circle-btn:hover {
      background-color: #f0f0f0;
    }

    .circle-btn:active {
      background-color: #808080;
    }

    #menuBtn i {
      font-size: 24px;
    }

    #settingBtn i {
      font-size: 24px;
    }

     #menu {
  position: fixed; /* 將選單設置為固定位置 */
  top: 0;
  left: -400px;
  width: 240px;
  height: 100vh;
  background-color: #ede9e8;
  transition: left 0.3s ease;
  padding: 20px;
  box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.3);
 z-index: 2;
}

    .menu-button-container {
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .square-btn {
      width: 100%;
      height: 40px;
      border: none;
      border-radius: 10px;
      background-color: #ede9e8;
      color: #9e9092;
      margin-bottom: 10px;
      cursor: pointer;
      transition: background-color 0.3s ease, color 0.3s ease;
      text-align: left;
      font-size: 14px;
    }

    .square-btn:hover {
      background-color: #ffffff;
    }

    .square-btn:active {
      background-color: #808080;
      color: #ffffff;
    }

    #image-container {
      width: calc(100% - 400px); /* 讓圖像框靠右 */
      height: 100%;
      display: flex;
      justify-content: flex-end; /* 靠右對齊 */
      align-items: center;
    }
    .container {
      position: relative;
      height: 100%;
      text-align: center;
    }

 .button-row {
      position: absolute;
      display: flex;
      flex-direction: column;
      gap: 60px; /* 按鈕組之間的垂直間距 */

    }

    .button-group {
      display: flex;
     flex-wrap: wrap; /* 在需要時換行 */
      justify-content: center; /* 在中央對齊 */
      gap: 20px; /* 按鈕之間的水平間距 */
  
    }

    .button-54.nav-button {
      font-family: "Open Sans", sans-serif;
      font-size: 16px;
      letter-spacing: 2px;
      text-decoration: none;
      text-transform: uppercase;
      color: #000;
      cursor: pointer;
      border: 3px solid;
      padding: 0.25em 0.5em;
      width: 100px; /* 調整按鈕的寬度 */
      box-shadow: 1px 1px 0px 0px, 2px 2px 0px 0px, 3px 3px 0px 0px, 4px 4px 0px 0px, 5px 5px 0px 0px;
      position: relative;
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
    }

    .button-54.nav-button:active {
      box-shadow: 0px 0px 0px 0px;
      top: 5px;
      left: 5px;
    }

    @media (max-width: 10000px) {
  .button-row {
    /* 保持絕對定位 */
    position: absolute;
    /* 調整 top 的值以適應垂直間距 */
    top: 50px; /* 距離頁面頂部的距離 */
    left: 50%;
    transform: translate(-50%); /* 水平和垂直置中 */
  }
}


    .box-container {
      display: inline-block;
      margin: 0px auto 0; /* 距離頁面頂端100px，水平居中 */
      position: relative;
    
    }

    .box {
      background-color:#ffffff; /* 设置背景颜色为灰色，您可以根据需要修改颜色值 */
      width: 500px;
      height: 500px;
      border: 4px solid #000;
      border-radius: 10px;
      box-sizing: border-box;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden; /* 確保圖片不會超出框 */
    }

    .box img {
      width: 100%; /* 使圖片填滿容器 */
      height: auto; /* 根據圖片比例自動調整高度 */
      object-fit: cover; /* 填滿容器並保持圖片比例 */
      position: absolute; /* 調整圖片位置 */
      top: 50%; /* 將圖片向下移動一半框高度 */
      left: 50%; /* 將圖片向右移動一半框寬度 */
      transform: translate(-50%, -50%); /* 使圖片居中 */
    }

    .box p {
      position: absolute;
      top: -100px; /* 調整文字位置為框的上方 */
      left: 50%;
      transform: translateX(-50%);
      background-color: #fff;
      padding: 10px;
      border: 2px solid #000;
      border-bottom: none;
    }

    img {
      width: 300px;
      height: 300px;
      transition: opacity 0.5s ease-in-out;
      opacity: 1;
      margin-top: 50px; /* 調整圖片與按鈕之間的垂直距離 */
    }

    #text {
      position: absolute;
      top: 770px; /* 調整文字與圖片之間的垂直距離 */
      left: 50%;
      transform: translateX(-50%);
      width: 100%;
      text-align: center;
      font-size: 20px;
    }
  </style>
</head>
<body>
 <div class="button-container">
    <!-- 左侧按钮 -->
    <button class="circle-btn" id="menuBtn"><i class="bi bi-list"></i></button> <!-- 使用 Bootstrap 图标 bi-list -->
    <!-- 右侧按钮 -->
    <button class="circle-btn" id="settingBtn"><i class="bi bi-gear"></i></button> <!-- 使用 Bootstrap 图标 bi-gear -->
  </div>

  <!-- 隐藏的菜单 -->
  <div id="menu">
    <div class="menu-button-container">
      <!-- 添加四边形按钮 -->
      <button class="square-btn" id="homeBtn">主頁</button>
      <button class="square-btn" id="emotionBtn">表 情 辨 識 & 顯 示 動 作 肌 群</button>
      <button class="square-btn" id="muscleBtn">臉 部 肌 肉 位 置 說 明</button>
      <button class="square-btn" id="aboutBtn">關 於 我 們</button>
    </div>
  </div>

  <div class="container">
    <div class="button-row">
      <div class="button-group">
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/SR6dVtHH/image.png', '這是皺眉肌')">皺眉肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/v8qpV1Gy/image.png', '這是降眉肌')">降眉肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/c4JdqyLx/image.png', '這是顴大肌')">顴大肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/1541CGv4/image.png', '這是顴小肌')">顴小肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/hGYKPJJ2/image.png', '這是咬肌')">咬肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/tJpCnsbW/image.png', '這是額肌')">額肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/zBRRFb4Q/image.png', '這是眼輪匝肌')">眼輪匝肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/QCswXsq6/image.png', '這是口輪匝肌')">口輪匝肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/QtqSJ7pb/image.png', '這是降下唇肌')">降下唇肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/L56Ky19S/image.png', '這是頰肌')">頰肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/65Z4sXPs/image.png', '這是笑肌')">笑肌</button>
        <button class="button-54 nav-button" onclick="showImage('https://i.postimg.cc/NGzRqbfX/image.png', '這是頦肌')">頦肌</button>
      </div>
    

   <div class="box-container">

   
      <div class="box" id="Boximg"><img src="https://i.postimg.cc/GpRsw906/aaa.png" alt=""></div>
    </div>
  </div>
  <div id="text">這是臉部肌肉解剖圖</div>

</div>
<script>
document.addEventListener('DOMContentLoaded', function() {
      const menuBtn = document.getElementById('menuBtn');
      const settingBtn = document.getElementById('settingBtn');
      const menu = document.getElementById('menu');
      const homeBtn = document.getElementById('homeBtn');
      const emotionBtn = document.getElementById('emotionBtn');
      const muscleBtn = document.getElementById('muscleBtn');
      const aboutBtn = document.getElementById('aboutBtn');
      const displayedImage = document.getElementById('displayed-image');

      // 点击menu按钮
      menuBtn.addEventListener('click', function(event) {
        menu.style.left = '0'; // 显示菜单
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 点击setting按钮
      settingBtn.addEventListener('click', function(event) {
        // 在这里添加setting按钮的事件处理逻辑
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 点击其他地方隐藏菜单
      document.addEventListener('click', function(event) {
        const target = event.target;
        if (!menu.contains(target) && target !== menuBtn) {
          menu.style.left = '-400px'; // 隐藏菜单
        }
      });

       // 点击"主頁"
      homeBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
      });

      // 点击"表情辨識 & 顯示動作肌群"
      emotionBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web2'; // 替换为相应页面的 URL
      });
   // 点击"臉部肌肉位置說明"
      muscleBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web3'; // 替换为相应页面的 URL
      });
      // 点击"關於我們"
      aboutBtn.addEventListener('click', function() {
        window.location.href = 'https://1e1c-210-59-96-137.ngrok-free.app/web1'; // 替换为相应页面的 URL
      });
    });
  // JavaScript代碼
  var isExecuting = false; // 標誌變量，用於檢查函數是否正在執行

  function showImage(imageUrl, textContent) {
    if (isExecuting) return; // 如果函數正在執行，則退出
    
    isExecuting = true; // 將標誌設置為 true，表示函數正在執行

    var imageElement = document.querySelector("#Boximg img");
    var textElement = document.getElementById("text");
    var buttons = document.querySelectorAll(".button-54.nav-button"); // 只選擇帶有.nav-button類的按鈕

    // 清空並設置文字
    textElement.innerHTML = '';

    // 禁用所有按鈕
    buttons.forEach(function(button) {
      button.disabled = true;
    });

    // 將圖片的透明度設置為 0
    imageElement.style.opacity = 0;

    // 過渡效果顯示圖片
    setTimeout(function() {
      var index = 0;
      var interval = setInterval(function() {
        if (index < textContent.length) {
          textElement.innerHTML += textContent[index++];
        }
        else {
          clearInterval(interval);
          
          // 啟用所有按鈕
          buttons.forEach(function(button) {
            button.disabled = false;
          });

          // 重置標誌變量，在函數執行結束後
          setTimeout(function() {
            isExecuting = false;
          }, 100); // 留出一些時間避免立即被再次點擊
        }
      }, 100); // 控制每個字之間的間隔時間
      imageElement.src = imageUrl; // 改變圖片
      imageElement.style.opacity = 1; // 將透明度設置為 1，顯示圖片
    }, 500); // 這裡的500是和圖片過渡時間相對應的
  }
</script>
</body>
</html>

"""
