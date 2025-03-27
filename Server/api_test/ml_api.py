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

#ngrok重啟時請先"ctrl+f"選取下列網址，並點選"全部取代"為新網址
ngrok = 'https://8638-210-59-96-137.ngrok-free.app'

# 载入定義的肌肉檔案
# 指定json檔路徑
json_path = '/home/wei-jie/test.json'

# 载入模型
# 指定模型路徑
MODEL_PATHS = {
    "Model_1": "/home/wei-jie/3class.keras",
    "Model_2": "/home/wei-jie/model.keras",
    "Model_3": "/home/wei-jie/model.keras",
}

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

# 處理從網頁收到的模型類別資訊
def load_model_by_label(label):
    if label in MODEL_PATHS:
        return tf.keras.models.load_model(MODEL_PATHS[label], custom_objects={'XceptionLayer': XceptionLayer, 'CBAMLayer': CBAMLayer})
    else:
        raise ValueError(f"Invalid model label: {label}")

# 初始化MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# 定義全局變量 original_img (網頁傳送的原始影像)
original_img = None

# 定義特徵點連線顏色
def connect_points(image, coordinates, color):
    for i in range(len(coordinates) - 1):
        cv2.line(image, coordinates[i], coordinates[i + 1], color, 2)

def detect_face_landmarks(image, results, feature_points_keys):
    # 從JSON檔讀取自訂肌肉範圍及占比
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 從JSON檔讀取肌肉視覺化標記顏色
    mu_color_mapping = {mu["mu_no"]: mu["mu_color"] for mu in data["mu_to_na"]}

    if results.multi_face_landmarks:
        for feature_points_key in feature_points_keys:
            # 創建列表用於儲存計算後座標
            final_coordinates = []

            # 检查是否存在特徵點資訊
            if feature_points_key not in data:
                print(f"No feature points found for key: {feature_points_key}")
                continue  # 如果没有特徵點，則跳過

            for item in data[feature_points_key]:
                # 确保包含比例和特徵點編號
                if "p" not in item or "v" not in item:
                    print(f"Missing data for item: {item}")
                    continue  # 如果數據不完整，則跳過

                scale_factors = [float(factor) for factor in item["p"].split()]  # 比例列表
                feature_point_indices = [int(idx) for idx in item["v"].split()]  # 特徵點編號列表

                # 初始化變數
                summed_x = 0
                summed_y = 0
                summed_z = 0

                # 指定的特徵點之座標*比例後計算出所需座標
                for idx, factor in zip(feature_point_indices, scale_factors):
                    # 确保索引在有效範圍
                    if idx < len(results.multi_face_landmarks[0].landmark):
                        landmark = results.multi_face_landmarks[0].landmark[idx]
                        x, y, z = landmark.x, landmark.y, landmark.z
                        x_image = int(x * image.shape[1] * factor)  #  x * 比例
                        y_image = int(y * image.shape[0] * factor)  #  y * 比例
                        z_image = int(z * factor)  # z * 比例
                        summed_x += x_image
                        summed_y += y_image
                        summed_z += z_image

                # 儲存計算後座標
                if summed_x != 0 and summed_y != 0:  # 确保坐标有效
                    final_coordinates.append((summed_x, summed_y))

            # 使列表中的最后特徵點與第一特徵點連接
            if final_coordinates:
                final_coordinates.append(final_coordinates[0])

                # 根據json檔獲取肌肉標記色彩，否則默認標記白色
                color = mu_color_mapping.get(feature_points_key, "#ffffff") 

                # 转换颜色格式（B, G, R）
                color_tuple = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1]

                # 调用连线函数
                connect_points(image, final_coordinates, color_tuple)

    return image

# 計算多邊形面積
def polygon_area(points):
    if len(points) < 3:
        return 0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# 計算左右面積差異
def calculate_area_difference(img, face_landmarks, json_data):
    # 指定正規化範圍
    min_area = 0  # 最小值
    max_area = 10  # 最大值

    # 面積正規化函數
    def normalize_area(area):
        return (area - min_area) / (max_area - min_area) if max_area > min_area else 0

    image_height, image_width = img.shape[:2]
    left_coords, right_coords = [], []

    for region, coords in zip(['area_l', 'area_r'], [left_coords, right_coords]):
        for item in json_data[region]:
            if "p" not in item or "v" not in item:
                continue

            scale_factors = [float(factor) for factor in item["p"].split()]
            feature_point_indices = [int(idx) for idx in item["v"].split()]

            # 計算實際座標
            summed_x, summed_y = 0, 0
            for idx, factor in zip(feature_point_indices, scale_factors):
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x, y = landmark.x * image_width * factor, landmark.y * image_height * factor
                    summed_x += x
                    summed_y += y
            if summed_x != 0 and summed_y != 0:
                coords.append((summed_x, summed_y))

        if coords:
            coords.append(coords[0]) 

    # 計算指定部位面積
    left_area = polygon_area(left_coords)
    right_area = polygon_area(right_coords)

    # 計算面積差，保留小數精度
    area_difference = round(abs(left_area - right_area), 4)

    # 面積差正規化
    normalized_difference = normalize_area(area_difference)
    
    # 若正規化後的數值超過 300，進行調整 (作為bug的暫時處理，有時計算出的面積會突然很大，可能是輸入的圖象尺寸過大)
    if normalized_difference > 300:
        print(f"正規化面積超過 300，調整前: {normalized_difference}")
        normalized_difference /= 10
        print(f"調整後的正規化面積: {normalized_difference}")
    
    normalized_difference = round(normalized_difference, 4)

    # 標記指定部位範圍
    cv2.polylines(img, [np.array(left_coords, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, [np.array(right_coords, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # 正確顯示正規化數值
    print(f"面積差（未正規化）: {area_difference}")
    print(f"面積差（正規化）: {normalized_difference}")

    # 設定等級和對應結果
    if normalized_difference < 135:
        level = "0"
        result = "正常"
    elif 135 <= normalized_difference < 150:
        level = "1"
        result = "輕度不協調"
    elif 150 <= normalized_difference < 160:
        level = "2"
        result = "中度不協調"
    else:  # normalized_difference >= 150
        level = "3"
        result = "重度不協調"

    return img, result, level, area_difference, normalized_difference


#收到圖像後的處理
@app.post("/emotion_recognition")
async def emotion_recognition(file: UploadFile = File(...), model_label: str = Form(...)):
    global json_path  # 使用全局變量 json_path
    try:
        model = load_model_by_label(model_label)
    except ValueError as e:
        return {"error": str(e)}
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = img.copy()  # 備份原始圖像
    
    # 複製原始圖像供面積計算使用（不經過肌肉標記處理）
    area_image_copy = original_img.copy()

    # 進行面部網格檢測
    results = face_mesh.process(original_img)
    
    # 如果未偵測到任何人臉，返回訊息
    if not results.multi_face_landmarks:
        return {"result": "未偵測到人臉", "muresult": "", "area_result": ""}

    # 進行表情預測
    img = cv2.resize(img, (100, 100))
    img_array = np.expand_dims(img, axis=0)  # 添加批次維度
    img_array = img_array / 255.0  # 正規化像素值
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    # 讀取 JSON 資料
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 查詢表情對應的AU和MU 
    exp_data = None
    for exp in data['exp_to_au']:
        if str(predicted_class) == str(exp.get('exp_num')):  # 對應表情代碼
            exp_data = exp
            break
    
    if exp_data:
        emotion_result = exp_data.get('exp', '未知表情')
        au_list = exp_data.get('au_no', '').split()
        
        mu_list = []
        mu_names = []
        mu_colors = []
        for au in au_list:
            for au_data in data['au_to_mu']:
                if au_data['au_no'] == au:
                    mus = au_data['mu_no'].split()
                    mu_list.extend(mus)
                    for mu in mus:
                        for mu_data in data['mu_to_na']:
                            if mu_data['mu_no'] == mu:
                                mu_names.append(mu_data['mu_na'])
                                mu_colors.append(mu_data['mu_color'])
                                break
        
        # 使用查詢到的MU執行檢測
        processed_image_for_muscles = detect_face_landmarks(original_img, results, mu_list)
        mu_result = ', '.join(mu_names)
        mu_color_result = ', '.join(mu_colors)
    else:
        emotion_result = "未知表情"
        processed_image_for_muscles = original_img
        mu_result = ""
        mu_color_result = ""

    # 使用先前複製的原始圖像進行面積計算
    area_image, area_result, level, area_difference,normalized_difference = calculate_area_difference(area_image_copy, results.multi_face_landmarks[0], data)

    # 水平翻轉圖像
    processed_image_for_muscles = cv2.flip(processed_image_for_muscles, 1)  # 水平翻轉肌肉標記圖像
    area_image = cv2.flip(area_image, 1)  # 水平翻轉面積計算圖像

    # 把圖片編碼為Base64格式
    _, encoded_image = cv2.imencode('.jpg', processed_image_for_muscles)
    muscle_image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    _, encoded_area_image = cv2.imencode('.jpg', area_image)
    area_image_base64 = base64.b64encode(encoded_area_image).decode('utf-8')

    # 返回最終結果
    return {
        "muscle_image": muscle_image_base64,  # 臉部肌肉位置的圖像
        "emotion_result": emotion_result, 
        "muresult": mu_result, 
        "mu_colors": mu_color_result,
        "area_image": area_image_base64,  # 面積計算後的圖像
        "area_result": area_result,
        "level": level  # 包含等級
    }



#主頁        
@app.get("/web1", response_class=HTMLResponse)
async def web1():
    return """   
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>主頁</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet"> <!-- Bootstrap 圖標CSS連結 -->
  <style>
    body {
      margin: 0; /* 去掉页面默认的外边距 */
      padding: 0; /* 去掉页面默认的内边距 */
      background-color: #f0f0f0;

    }

    /* 容器_導覽列樣式 */
    .navbar-container {
      display: flex;
      position: fixed;
      align-items: center; /* 水平置中容器內元素 */
      justify-content: space-between; /* 元素分別靠最左和最右端對齊 */
      width: 100%;
      height: 50px;
      border-bottom: 1px solid #ccc; /* 底線 */
      padding: 0; /* 容器內元素與容器間的邊距 */    
      background-color: #f0f0f0;
      z-index: 5;  /* 确保导航栏在页面其他元素的上方 */
    }
    
     /* 導覽列樣式 */
    .navbar-btn {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 45px;
      height: 45px;
      border-radius: 50%;
      border: none;
      background-color: transparent;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }

    /* 導覽列-按鈕觸碰變色 */
    .navbar-btn:hover {
      background-color: #e7e6e6;
    }

    /* 導覽列-按鈕點擊變色 */
    .navbar-btn:active {
      background-color: #a8a7a7;
    }

    /* 左側選單樣式 */
    #sidebar {
      position: fixed; /* 將選單設置為固定位置 */
      top: 0;
      left: -400px; /* 將選單放置頁面外 */
      width: 200px;
      height: 100vh;
      background-color: #f0f0f0;
      transition: left 0.3s ease; /* 實現選單滑入效果 */
      padding: 20px;
      box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.3);
      z-index: 6;
    }

    /* 左側選單-選項容器樣式 */
    .sidebar-selection-container {
      border-bottom: 1px solid #ccc;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    /* 左側選單-選項文字樣式 */
    .sidebar-btn-text {
      text-align: left;
      font-size: 15px;
    }

    /* 左側選單-選項按鈕樣式 */
    .sidebar-btn {
      display: flex;
      flex-direction: column;
      align-items: center;
      flex-direction: row;
      width: 100%;
      height: 40px;
      border: none;
      border-radius: 12px;
      background-color: #f0f0f0;
      color: #ac9599;
      gap: 5px;
      padding: 20px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: background-color 0.3s ease, color 0.3s ease;     
    }

    /* 左側選單-選項按鈕觸碰變色 */
    .sidebar-btn:hover {
      background-color: #ffffff;
    }

    /* 左側選單-選項按鈕點擊變色 */
    .sidebar-btn:active {
      background-color: #d19494;
      color: #ffffff;
    }

    }
  </style>
</head>

<body>
  <!-- 導覽列 -->
  <div class="navbar-container">
    <div class="navbar-container">
    <!-- 按鈕_顯示左側選單 -->
    <button class="navbar-btn" id="sidebarBtn"><i class="bi bi-list" style="transform: scale(1.5); color: #45485f;"></i></button>     
    <!-- 按鈕_顯示設定選單 -->
    <button class="navbar-btn" id="settingBtn"><i class="bi bi-gear" style="transform: scale(1.5); color: #45485f;"></i></button>   
  </div>

  <!-- 左側選單 -->
  <div id="sidebar">
    <div class="sidebar-selection-container">
      <!-- 左側選單選項 -->
      <button class="sidebar-btn" id="homeBtn"><i class="bi bi-house" style="transform: scale(1.5)"></i>
        <div class="sidebar-btn-text">主頁</div>
      </button>
      <button class="sidebar-btn" id="emotionBtn"><i class="bi bi-clipboard-data-fill" style="transform: scale(1.5)"></i>
        <div class="sidebar-btn-text">臉部肌肉分析</div>
      </button> 
    </div>
  </div>

  <script>  
    document.addEventListener('DOMContentLoaded', function() {
      const sidebarBtn = document.getElementById('sidebarBtn');
      const settingBtn = document.getElementById('settingBtn');
      const sidebar = document.getElementById('sidebar');
      const homeBtn = document.getElementById('homeBtn');
      const emotionBtn = document.getElementById('emotionBtn');
      
      // 當點擊sidebar按钮
      sidebarBtn.addEventListener('click', function(event) {
        sidebar.style.left = '0'; // 顯示左側選單
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 當點擊setting按钮
      settingBtn.addEventListener('click', function(event) {
        event.stopPropagation(); // 阻止事件冒泡
      });

      // 當點擊空白處
      document.addEventListener('click', function(event) {
        const target = event.target;
        if (!sidebar.contains(target) && target !== sidebarBtn) {
          sidebar.style.left = '-400px'; // 隱藏左側選單
        }
      });

      // 點擊"主頁"
      homeBtn.addEventListener('click', function() {
        window.location.href = 'https://8638-210-59-96-137.ngrok-free.app/web1'; 
      });

      // 點擊"表情辨識 & 顯示動作肌群"
      emotionBtn.addEventListener('click', function() {
        window.location.href = 'https://8638-210-59-96-137.ngrok-free.app/web2'; 
      });
    });
  </script>
</body>
</html>
"""

#肌肉評估網頁
@app.get("/web2", response_class=HTMLResponse)
async def web2():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="https://i.imgur.com/SzojS5O.png" sizes="32x32">
    <title>表情辨識與肌肉分析</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.11.3/font/bootstrap-icons.min.css"
        rel="stylesheet">
    <style>
        body {
            margin: 0;
            /* 去掉页面默认的外边距 */
            padding: 0;
            /* 去掉页面默认的内边距 */
            background-color: #f0f0f0;
            overflow: auto;
        }

        /* 頁面尺寸變換時,改變排版 */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                gap: 30px;
                overflow-y: auto;
            }
        }

        /* 容器放置分析和回傳類功能 */
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background-color: #f0f0f0;
            overflow: auto;
        }

        .text {
            margin-bottom: 5px;
            font-size: 20px;
        }

        /* 容器_導覽列樣式 */
        .navbar-container {
            display: flex;
            position: fixed;
            align-items: center;
            /* 水平置中容器內元素 */
            justify-content: space-between;
            /* 元素分別靠最左和最右端對齊 */
            width: 100%;
            height: 50px;
            border-bottom: 1px solid #ccc;
            /* 底線 */
            padding: 0;
            /* 容器內元素與容器間的邊距 */
            background-color: #f0f0f0;
            z-index: 5;
            /* 确保导航栏在页面其他元素的上方 */
        }

        /* 導覽列樣式 */
        .navbar-btn {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            border: none;
            background-color: transparent;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        /* 導覽列-按鈕觸碰變色 */
        .navbar-btn:hover {
            background-color: #e7e6e6;
        }

        /* 導覽列-按鈕點擊變色 */
        .navbar-btn:active {
            background-color: #a8a7a7;
        }

        /* 左側選單樣式 */
        #sidebar {
            position: fixed;
            /* 將選單設置為固定位置 */
            top: 0;
            left: -400px;
            /* 將選單放置頁面外 */
            width: 200px;
            height: 100vh;
            background-color: #f0f0f0;
            transition: left 0.3s ease;
            /* 實現選單滑入效果 */
            padding: 20px;
            box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.3);
            z-index: 6;
        }

        /* 左側選單-選項容器樣式 */
        .sidebar-selection-container {
            border-bottom: 1px solid #ccc;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        /* 左側選單-選項文字樣式 */
        .sidebar-btn-text {
            text-align: left;
            font-size: 15px;
        }

        /* 左側選單-選項按鈕樣式 */
        .sidebar-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-direction: row;
            width: 100%;
            height: 40px;
            border: none;
            border-radius: 12px;
            background-color: #f0f0f0;
            color: #ac9599;
            gap: 5px;
            padding: 20px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        /* 左側選單-選項按鈕觸碰變色 */
        .sidebar-btn:hover {
            background-color: #ffffff;
        }

        /* 左側選單-選項按鈕點擊變色 */
        .sidebar-btn:active {
            background-color: #d19494;
            color: #ffffff;
        }

        /* 容器_一般 */
        .norm-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-start;
            max-width: 500px;
            width: 100%;
            gap: 15px;
            border: 2px solid #aaaaaa;
            font-size: 20px;
            font-weight: bold;
            color: #7F7F7F;
            background-color: white;
            padding: 15px;
            position: relative;
            margin-top: 80px;
            box-sizing: border-box;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }

        /* 容器_添加樣本按鈕 */
        .sample-btn-container {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
            align-items: flex-start;
            max-width: 500px;
            width: 100%;
            gap: 15px;
            box-sizing: border-box;
        }

        /* 容器_添加樣本 */
        .add-sample-container {
            display: none;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 10px;
            width: 100%;
            max-width: 700px;
            padding-top: 20px;
            padding-bottom: 20px;
            box-sizing: border-box;
        }

        /* 添加樣本按鈕 */
        .add-sample-btn {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 5px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #9DB6CF;
            color: white;
            border: none;
            border-radius: 5px;
        }

        /* 空方框 */
        .box {
            position: relative;
            max-width: 400px;
            width: 100%;
            aspect-ratio: 1 / 1;
            background-color: #FFFFFF;
            border: 3px solid #999a9b;
            box-sizing: border-box;
            border-radius: 5px;
            overflow: hidden;
            /* 確保內容不會超出容器 */
        }

        /* 空方框圖像設定 */
        .box img {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            /* 填满容器并保持纵横比 */
        }

        /* 視訊框內的圓框濾鏡 */
        .facefilter {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            height: 95%;
            background-color: transparent;
            border: 5px solid #999a9b;
            border-radius: 50%;
        }

        /* 按鈕普通 */
        .norm-button {
            align-items: center;
            appearance: none;
            background-color: #8db18e;
            border-radius: 4px;
            border-width: 0;
            box-shadow: rgba(45, 35, 66, 0.4) 0 2px 4px, rgba(45, 35, 66, 0.3) 0 7px 13px -3px, #D6D6E7 0 -3px 0 inset;
            box-sizing: border-box;
            color: #ffffff;
            cursor: pointer;
            display: inline-flex;
            font-family: "JetBrains Mono", monospace;
            min-width: 111px;
            height: 48px;
            justify-content: center;
            line-height: 1;
            list-style: none;
            overflow: hidden;
            padding-left: 16px;
            padding-right: 16px;
            position: relative;
            text-align: left;
            text-decoration: none;
            transition: box-shadow .15s, transform .15s;
            user-select: none;
            -webkit-user-select: none;
            touch-action: manipulation;
            white-space: nowrap;
            will-change: box-shadow, transform;
            font-size: 16px;
        }

        .norm-button:focus {
            box-shadow: #D6D6E7 0 0 0 1.5px inset, rgba(45, 35, 66, 0.4) 0 2px 3px, rgba(45, 35, 66, 0.3) 0 7px 13px -3px, #D6D6E7 0 -3px 0 inset;
        }

        .norm-button:hover {
            box-shadow: rgba(45, 35, 66, 0.4) 0 4px 8px, rgba(45, 35, 66, 0.3) 0 7px 13px -3px, #D6D6E7 0 -3px 0 inset;
            transform: translateY(-2px);
        }

        .norm-button:active {
            box-shadow: #D6D6E7 0 3px 7px inset;
            transform: translateY(2px);
        }

        /* 按鈕-關閉OR返回 */
        .close-button {
            display: flex;
            position: absolute;
            top: 90px;
            right: 20px;
            background-color: transparent;
            border: none;
            cursor: pointer;
        }

        /* 模型設定按鈕 */
        .model-settings-btn {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background-color: transparent;
            border: none;
            cursor: pointer;
            font-size: 24px;
            display: none;
        }

        /* 容器-模型設定 */
        .model-settings-container {
            width: 70%;
        }

        /* 模型設定-內容 */
        .model-settings-menu {
            display: none;
            position: absolute;
            bottom: -80px;
            left: 250px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 200px;
            padding: 10px;
            box-sizing: border-box;
            z-index: 2;
        }

        /* 模型設定內容-顯示 */
        .model-settings-menu.show {
            display: flex;
            flex-direction: row;
        }

        /* 模型設定-內容-標題 */
        .model-settings-menu .menu-header {
            font-size: 15px;
            margin-bottom: 10px;
            font-weight: bold;
        }

        /* 模型設定-內容-選擇項目 */
        .model-settings-menu select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }

        /* 預留按鈕 */
        .reserved-menu-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: transparent;
            border: none;
            cursor: pointer;
            font-size: 24px;
        }

        /* 預留按鈕-內容 */
        .reserved-menu {
            display: none;
            position: absolute;
            top: 40px;
            left: 310px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 150px;
            height: auto;
            padding: 0;
            box-sizing: border-box;
            overflow: hidden;
            flex-direction: column;
            z-index: 2;
        }

        /* 預留按鈕-顯示 */
        .reserved-menu.show {
            display: flex;
        }

        /* 預留按鈕-內容-選項 */
        .reserved-menu button {
            flex: 1;
            background-color: transparent;
            border: none;
            font-weight: bold;
            color: #7F7F7F;
            padding: 8px;
            margin: 0;
            text-align: left;
            cursor: pointer;
        }

        /* 預留按鈕-內容-選項觸碰變色 */
        .reserved-menu button:hover {
            background-color: #f0f0f0;
        }

        /* 圖標-內容提示 */
        .tip-icon {
            display: flex;
            background-color: transparent;
            color: #6e6e6e;
            cursor: pointer;
            margin-top: 20px;
            margin-left: 25px;
        }

        /* 背景-內容提示 */
        .tip {
            position: absolute;
            background-color: #333;
            color: #fff;
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            width: 150px;
            left: -20px;
            top: -50px;
            font-size: 18px;
            display: none;
            word-wrap: break-word;
            white-space: normal;
            word-break: keep-all;
            z-index: 1;
        }

        /* 攝像頭影像 */
        #video {
            width: 100%;
            height: 100%;
            transform: scaleX(-1);
            filter: brightness(100%);
            /* 调整亮度的滤镜，数值可以根据需求调整 */
            object-fit: cover;
            /* 填满容器并保持纵横比 */
        }

        /* 攝像頭頁面-按鈕-容器 */
        .Camera-Btn-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 50px;
            margin-top: 10px;
        }


        /* 上傳圖像頁面-按鈕-容器 */
        .Upload-Btn-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 50px;
            margin-top: 10px;
        }

        /* 圖像上傳預覽 */
        #UploadSamplePreview img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            /* 確保圖片自動縮放填滿容器，且不會變形 */
        }

        /* 容器_回傳結果 */
        .result-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 20px;
            width: 100%;
            max-width: 700px;
            padding-top: 10px;
            padding-bottom: 10px;
            box-sizing: border-box;
        }

        /* 回傳結果-圖像 */
        #result-img {
            transform: scaleX(-1);
            filter: brightness(100%);
            /* 调整亮度的滤镜，数值可以根据需求调整 */
        }

        /* 容器-回傳結果-情緒 */
        #emotion-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            /* 允許換行 */
            gap: 15px;
            width: 300px;
            /* 控制容器寬度 */
        }

        /* 回傳結果-情緒 */
        .expmessage {
            display: flex;
            align-items: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(100%);
            transition: all 1s ease-out;
            /* 縮短過渡動畫至 0.2 秒 */
        }

        /* 回傳結果-情緒-訊息滑入效果 */
        .expmessage.visible {
            opacity: 1;
            transform: translateY(0);
        }

        /* 回傳結果-情緒-圖標 */
        .expicon {
            margin-right: 10px;
        }

        /* 容器-回傳結果-肌肉 */
        #mu-container {
            display: flex;
            flex-wrap: wrap;
            /* 允許換行 */
            gap: 15px;
            width: 300px;
            /* 控制容器寬度 */
        }

        /* 回傳結果-肌肉 */
        .mumessage {
            display: flex;
            align-items: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(100%);
            transition: all 1s ease-out;
            /* 縮短過渡動畫至 0.2 秒 */
            width: calc(40%);
            /* 每條訊息寬度為 25% 減去間隔，以便顯示四條訊息一行 */
        }

        /* 回傳結果-肌肉-訊息滑入效果 */
        .mumessage.visible {
            opacity: 1;
            transform: translateY(0);
        }

        /* 回傳結果-肌肉-圖標 */
        .muicon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }

        /* 容器-回傳結果-情緒 */
        #level-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            /* 允許換行 */
            gap: 15px;
            width: 300px;
            /* 控制容器寬度 */
        }

        /* 回傳結果-臉部面積 */
        .levelmessage {
            display: flex;
            align-items: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(100%);
            transition: all 1s ease-out;
            /* 縮短過渡動畫至 0.2 秒 */
        }

        /* 回傳結果-臉部面積-訊息滑入效果 */
        .levelmessage.visible {
            opacity: 1;
            transform: translateY(0);
        }

        /* 回傳結果-臉部面積-圖標 */
        .levelicon {
            margin-right: 10px;
        }

        /* 容器-回傳結果-中風判斷 */
        #area-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            /* 允許換行 */
            gap: 15px;
            width: 300px;
            /* 控制容器寬度 */
        }

        /* 回傳結果-中風判斷 */
        .areamessage {
            display: flex;
            align-items: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(100%);
            transition: all 1s ease-out;
            /* 縮短過渡動畫至 0.2 秒 */
        }

        /* 回傳結果-中風判斷-訊息滑入效果 */
        .areamessage.visible {
            opacity: 1;
            transform: translateY(0);
        }

        /* 回傳結果-中風判斷-圖標 */
        .areaicon {
            margin-right: 10px;
        }

        /* 容器_載入中濾鏡 */
        #loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.8);
            /* 半透明背景 */
            padding: 20px;
            border-radius: 10px;
        }

        /* 載入中濾鏡-圓圈 */
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #ccc;
            border-top: 5px solid #333;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 10px;

        }

        /* 載入中濾鏡-圓圈自轉動畫 */
        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }
    </style>

</head>

<body>
    <!-- 導覽列 -->
    <div class="navbar-container">
        <!-- 按鈕_顯示左側選單 -->
        <button class="navbar-btn" id="sidebarBtn"><i class="bi bi-list"
                style="transform: scale(1.5); color: #45485f;"></i></button>
        <!-- 按鈕_顯示設定選單 -->
        <button class="navbar-btn" id="settingBtn"><i class="bi bi-gear"
                style="transform: scale(1.5); color: #45485f;"></i></button>
    </div>

    <!-- 左側選單 -->
    <div id="sidebar">
        <div class="sidebar-selection-container">
            <!-- 左側選單選項 -->
            <button class="sidebar-btn" id="homeBtn"><i class="bi bi-house" style="transform: scale(1.5)"></i>
                <div class="sidebar-btn-text">主頁</div>
            </button>
            <button class="sidebar-btn" id="emotionBtn"><i class="bi bi-clipboard-data-fill"
                    style="transform: scale(1.5)"></i>
                <div class="sidebar-btn-text">臉部肌肉分析</div>
            </button>
        </div>
    </div>

    <!-- 分析和回傳類功能容器 -->
    <div class="container" id="container">
        <!-- 新增圖片樣本容器 -->
        <div class="norm-container" id="sample-container">
            新增圖片樣本:
            <div style="border-bottom: 1px solid #aaaaaa; width: 100%;"></div>

            <!-- 預留按鈕和選單 -->
            <button class="reserved-menu-btn" id="reserved-menu-btn">
                <i class="bi bi-three-dots-vertical" style="color: #7a8b9c;"></i>
            </button>
            <div class="reserved-menu" id="reserved-menu">
                <button>選項 1</button>
                <button>選項 2</button>
                <button>選項 3</button>
                <button>選項 4</button>
                <button>選項 5</button>
            </div>

            <!-- 模型設定按鈕和選單 -->
            <button class="model-settings-btn" id="model-settings-btn">
                <i class="bi bi-gear" style="color: #7a8b9c;"></i>
            </button>
            <div class="model-settings-menu" id="model-settings-menu">
                <div class="model-settings-container">
                    <div class="menu-header">選擇模型</div>
                    <select id="modelSelect">
                        <option>Model_1</option>
                        <option>Model_2</option>
                        <option>Model_3</option>
                    </select>
                </div>
                <!-- 模型設定說明 -->
                <div class="tip-icon" id="model-settings-tip-icon">
                    <i class="bi bi-question-diamond"></i>
                    <div class="tip" id="model-settings-tip">
                        可於選單中切換用於辨識的模型種類。
                    </div>
                </div>
            </div>

            <!-- 新增圖片樣本選項-攝像頭和上傳 -->
            <div class="sample-btn-container" id="sample-btn-container">
                <button class="add-sample-btn" id="add-sample-btn-camera">
                    <i class="bi bi-camera-video" style="transform: scale(1.5);"></i>攝像頭
                </button>
                <button class="add-sample-btn" id="add-sample-btn-upload">
                    <i class="bi bi-cloud-upload" style="transform: scale(1.5);"></i>上傳
                </button>
            </div>

            <!-- 攝像頭功能頁 -->
            <div class="add-sample-container" id="add-sample-camera">
                <button class="close-button" id="close-btn-camera"><i class="bi bi-x-circle-fill"
                        style="transform: scale(1.5); color: red;"></i></button>
                <div class="text">請將頭部對齊圓圈</div>
                <div class="box"> <video id="video" autoplay="true" playsinline></video>
                    <div class="facefilter"></div>
                </div>
                <div class="Camera-Btn-container" id="Camera-Btn-container">
                    <button class="norm-button" id="analyzeBtn-camera">開始分析</button>
                </div>
            </div>

            <!-- 上傳圖像功能頁 -->
            <div class="add-sample-container" id="add-sample-upload">
                <button class="close-button" id="close-btn-upload"><i class="bi bi-x-circle-fill"
                        style="transform: scale(1.5); color: red;"></i></button>
                <!-- 隱藏的 input file 按鈕 -->
                <input type="file" id="hiddenFileInput" style="display: none;" accept="image/png, image/jpeg">
                <div class="text">預覽圖像</div>
                <div class="box" id="UploadSamplePreview"></div>
                <div class="Upload-Btn-container" id="Upload-Btn-container">
                    <button class="norm-button" id="upload-btn">從裝置上傳</button>
                    <button class="norm-button" id="analyzeBtn-upload">開始分析</button>
                </div>
            </div>
        </div>

        <!-- 伺服器回傳的結果頁面-處理後肌肉視覺化圖像 & 表情 & 使用到肌肉 -->
        <div class="norm-container" id="containerd">
            <div class="result-container" id="result-exp-mu">
                <div class="text">伺服器回傳的結果</div>
                <div class="box" id="result-muimg"></div>
                <div class="text">情緒</div>
                <div id="loading" style="display:none;">
                    <div class="spinner"></div>
                    <p>Loading...</p>
                </div>
                <div id="emotion-container"></div>
                <div class="text">使用到的肌肉部位</div>
                <div id="mu-container"></div>
            </div>
        </div>

        <!-- 伺服器回傳的結果頁面-處理後臉部比例圖像 & 臉部比例 -->
        <div class="norm-container" id="containerF">
            <!-- 容器 伺服器回傳的結果 -->
            <div class="result-container" id="result-face-area">
                <div class="text">伺服器回傳的結果</div>
                <div class="box" id="result-areaimg"></div>
                <div class="text">評估結果</div>
                <div id="level-container"></div>
                <div id="area-container"></div>

            </div>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const sidebarBtn = document.getElementById('sidebarBtn');
            const settingBtn = document.getElementById('settingBtn');
            const sidebar = document.getElementById('sidebar');
            const homeBtn = document.getElementById('homeBtn');
            const emotionBtn = document.getElementById('emotionBtn');

            // 當點擊sidebar按钮
            sidebarBtn.addEventListener('click', function (event) {
                sidebar.style.left = '0'; // 顯示左側選單
                event.stopPropagation(); // 阻止事件冒泡
            });

            // 當點擊setting按钮
            settingBtn.addEventListener('click', function (event) {
                event.stopPropagation(); // 阻止事件冒泡
            });

            // 當點擊空白處
            document.addEventListener('click', function (event) {
                const target = event.target;
                if (!sidebar.contains(target) && target !== sidebarBtn) {
                    sidebar.style.left = '-400px'; // 隱藏左側選單
                }
            });

            // 點擊"主頁"
            homeBtn.addEventListener('click', function () {
                window.location.href = 'https://8638-210-59-96-137.ngrok-free.app/web1';
            });

            // 點擊"表情辨識 & 顯示動作肌群"
            emotionBtn.addEventListener('click', function () {
                window.location.href = 'https://8638-210-59-96-137.ngrok-free.app/web2';
            });
        });

        // 處理情緒結果函數 
        function createEmotionMessage(emotion) {
            const expmessage = document.createElement('div');
            expmessage.classList.add('expmessage');

            const expicon = document.createElement('div');
            expicon.classList.add('expicon');

            // 根据情绪類别設置圖標樣式
            let expiconHTML = ''; // 初始化圖標樣式

            switch (emotion) {
                case 'happy':
                    expiconHTML = '<i class="bi bi-emoji-smile-fill" style="color: #ff893a;"></i>'; // 情緒快樂使用該圖標
                    break;
                case 'angry':
                    expiconHTML = '<i class="bi bi-emoji-angry-fill" style="color: #e24343;"></i>'; // 情緒生氣使用該圖標
                    break;
                default:
                    expiconHTML = '<i class="bi bi-emoji-neutral-fill" style="color: gray;"></i>'; // 情緒無表情使用該圖標
                    break;
            }

            // 将圖標添加到icon元素中
            expicon.innerHTML = expiconHTML;

            const exptext = document.createElement('div');
            exptext.textContent = emotion; // 設定情绪類別文字

            expmessage.appendChild(expicon);
            expmessage.appendChild(exptext);

            return expmessage;
        }

        // 顯示情緒結果函數
        function addEmotionMessages(emotion) {
            const emotionContainer = document.getElementById('emotion-container');
            emotionContainer.innerHTML = ''; // 清空容器中的旧消息

            const emotionMessage = createEmotionMessage(emotion); // 创建带有情绪的消息
            emotionContainer.appendChild(emotionMessage); // 将情绪消息添加到情绪容器中

            // 添加滑入效果
            setTimeout(() => {
                emotionMessage.classList.add('visible');
            }, 200);
        }

        // 處理每個肌肉的名稱和顏色函數
        function createmuMessage(muscle) {
            const mumessage = document.createElement('div');
            mumessage.classList.add('mumessage');

            const muicon = document.createElement('div');
            muicon.classList.add('muicon');
            muicon.style.backgroundColor = muscle.color;  // 使用返回的肌肉色碼

            const mutext = document.createElement('div');
            mutext.textContent = muscle.name;  // 使用返回的肌肉名稱

            mumessage.appendChild(muicon);
            mumessage.appendChild(mutext);

            return mumessage;
        }

        // 顯示肌肉使用部位與對應顏色函數
        function addmuMessages(muscles) {
            const mucontainer = document.getElementById('mu-container');
            mucontainer.innerHTML = '';  // 清空容器中的舊消息

            muscles.forEach((muscle, index) => {
                const mumessage = createmuMessage(muscle);  // 創建帶有名稱和顏色的消息
                mucontainer.appendChild(mumessage);

                // 使用 setTimeout 增加滑入效果
                setTimeout(() => {
                    mumessage.classList.add('visible');
                }, index * 200);
            });
        }

        // 處理臉部面積結果函數 
        function createlevelMessage(level) {
            const levelmessage = document.createElement('div');
            levelmessage.classList.add('levelmessage');

            const levelicon = document.createElement('div');
            levelicon.classList.add('levelicon');

            // 設置圖標和文字樣式
            let leveliconHTML, levelTextColor;
            if (level === "等級：0") {
                leveliconHTML = '<i class="bi bi-bar-chart-line-fill" style="color: #8db18e;"></i>';
                levelTextColor = "#8db18e";
            } else if (level === "等級：1") {
                leveliconHTML = '<i class="bi bi-bar-chart-line-fill" style="color: #e9e04d;"></i>';
                levelTextColor = "#e9e04d";
            } else if (level === "等級：2") {
                leveliconHTML = '<i class="bi bi-bar-chart-line-fill" style="color: #e28c4d;"></i>';
                levelTextColor = "#e28c4d";
            } else if (level === "等級：3") {
                leveliconHTML = '<i class="bi bi-bar-chart-line-fill" style="color: #de6161;"></i>';
                levelTextColor = "#de6161";
            }

            // 將圖標添加到 icon 元素中
            levelicon.innerHTML = leveliconHTML;

            const leveltext = document.createElement('div');
            leveltext.textContent = level; // 設定情緒類別文字
            leveltext.style.color = levelTextColor; // 設置文字顏色

            levelmessage.appendChild(levelicon);
            levelmessage.appendChild(leveltext);

            return levelmessage;
        }


        // 顯示評估結果函數
        function addlevelMessages(level) {
            const levelContainer = document.getElementById('level-container');
            levelContainer.innerHTML = ''; // 清空容器中的旧消息

            const levelMessage = createlevelMessage("等級：" + level); // 创建带有情绪的消息
            levelContainer.appendChild(levelMessage); // 将情绪消息添加到情绪容器中

            // 添加滑入效果
            setTimeout(() => {
                levelMessage.classList.add('visible');
            }, 200);
        }


        // 顯示評估結果函數
        function addareaMessages(area) {
            const areaContainer = document.getElementById('area-container');
            areaContainer.innerHTML = ''; // 清空容器中的旧消息

            const areaMessage = createareaMessage(area); // 创建带有情绪的消息
            areaContainer.appendChild(areaMessage); // 将情绪消息添加到情绪容器中

            // 添加滑入效果
            setTimeout(() => {
                areaMessage.classList.add('visible');
            }, 200);
        }

        // 處理評估結果函數
        function createareaMessage(area) {
            const areamessage = document.createElement('div');
            areamessage.classList.add('areamessage');

            const areaicon = document.createElement('div');
            areaicon.classList.add('areaicon');

            // 設置圖標和文字樣式
            let areaiconHTML, areaTextColor;
            if (area === '重度不協調') {
                areaiconHTML = '<i class="bi bi-exclamation-diamond-fill" style="color: #de6161;"></i>';
                areaTextColor = "#de6161";
            } else if (area === '中度不協調') {
                areaiconHTML = '<i class="bi bi-exclamation-diamond-fill" style="color: #e28c4d;"></i>';
                areaTextColor = "#e28c4d";
            } else if (area === '輕度不協調') {
                areaiconHTML = '<i class="bi bi-exclamation-diamond-fill" style="color: #e9e04d;"></i>';
                areaTextColor = "#e9e04d";
            } else if (area === '正常') {
                areaiconHTML = '<i class="bi bi-exclamation-diamond-fill" style="color: #8db18e;"></i>';
                areaTextColor = "#8db18e";
            }

            // 將圖標添加到 icon 元素中
            areaicon.innerHTML = areaiconHTML;

            const areatext = document.createElement('div');
            areatext.textContent = area; // 設定評估結果文字
            areatext.style.color = areaTextColor; // 設置文字顏色

            // 將圖標和文字添加到訊息容器
            areamessage.appendChild(areaicon);
            areamessage.appendChild(areatext);

            return areamessage;
        }


        // 顯示評估結果函數
        function addareaMessages(area) {
            const areaContainer = document.getElementById('area-container');
            areaContainer.innerHTML = ''; // 清空容器中的旧消息

            const areaMessage = createareaMessage(area); // 创建带有情绪的消息
            areaContainer.appendChild(areaMessage); // 将情绪消息添加到情绪容器中

            // 添加滑入效果
            setTimeout(() => {
                areaMessage.classList.add('visible');
            }, 200);
        }

        const AddSampleBtnCamera = document.getElementById("add-sample-btn-camera");
        const AddSampleBtnUpload = document.getElementById("add-sample-btn-upload");
        const CloseBtnCamera = document.getElementById("close-btn-camera");
        const CloseBtnUpload = document.getElementById("close-btn-upload");
        const SampleBtnContainer = document.getElementById("sample-btn-container");
        const AddSampleCamera = document.getElementById("add-sample-camera");
        const AddSampleUpload = document.getElementById("add-sample-upload");
        const ReservedMenuBtn = document.getElementById("reserved-menu-btn");
        const dropdownMenu = document.getElementById("reserved-menu");
        const ModelSettingsBtn = document.getElementById("model-settings-btn");
        const ModelSettingsMenu = document.getElementById("model-settings-menu");
        const ModelSettingsTipIcon = document.getElementById('model-settings-tip-icon');
        const ModelSettingsTip = document.getElementById('model-settings-tip');
        const UploadBtn = document.getElementById('upload-btn');
        const fileInput = document.getElementById('fileInput');

        // 變數來追蹤攝像頭是否連接成功
        let isCameraConnected = false;

        // 獲取攝像頭影像並顯示在 video 元素中
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                const video = document.getElementById('video'); // 確保 video 元素存在
                video.srcObject = stream;
                video.play();
                isCameraConnected = true; // 攝像頭連接成功
            })
            .catch(err => {
                console.error('Error accessing the camera:', err);
                isCameraConnected = false; // 攝像頭連接失敗
            });

        // 監聽攝像頭頁面 "分析" 按鈕的點擊事件
        document.getElementById('analyzeBtn-camera').addEventListener('click', () => {
            if (!isCameraConnected) {
                alert('請先連接攝像頭！'); // 攝像頭未連接，顯示提示訊息
                return; // 阻止事件繼續觸發
            }
            captureAndSendImage(); // 攝像頭已連接，繼續進行分析
        });

        // 將攝像頭影像發送到伺服器
        function captureAndSendImage() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const loadingElement = document.getElementById('loading'); // 取得 loading 元素
            const size = Math.min(video.videoWidth, video.videoHeight); // 截取video的尺寸
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');
            const x = (video.videoWidth - size) / 2; // 計算截取的起點坐標
            const y = (video.videoHeight - size) / 2;

            // 顯示 loading 動畫
            loadingElement.style.display = 'flex';

            ctx.drawImage(video, x, y, size, size, 0, 0, size, size); // 繪製截取的部分到video

            // 獲取選擇的模型標籤
            const selectedModelLabel = document.getElementById('modelSelect').value;

            // 將圖像轉換為 Blob 物件
            canvas.toBlob(blob => {
                // 創建表單數據物件
                const formData = new FormData();
                formData.append('file', blob, 'captured_image.jpg'); // 圖像文件名，修改為 JPEG 格式
                formData.append('model_label', selectedModelLabel);  // 發送模型標籤

                // 發送 POST 請求
                fetch("https://8638-210-59-96-137.ngrok-free.app/emotion_recognition", {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        const ResultMuImg = document.getElementById('result-muimg');
                        ResultMuImg.innerHTML = '';

                        const ResultAreaImg = document.getElementById('result-areaimg');
                        ResultAreaImg.innerHTML = '';

                        // 顯示臉部肌肉圖像
                        if (data.muscle_image) {
                            const muscleImgBase64 = data.muscle_image;
                            const muscleImgElement = document.createElement('img');
                            muscleImgElement.src = 'data:image/jpeg;base64,' + muscleImgBase64.replace(/"/g, '');
                            ResultMuImg.appendChild(muscleImgElement);
                        }

                        // 顯示臉部區域面積圖像
                        if (data.area_image) {
                            const areaImgBase64 = data.area_image;
                            const areaImgElement = document.createElement('img');
                            areaImgElement.src = 'data:image/jpeg;base64,' + areaImgBase64.replace(/"/g, '');
                            ResultAreaImg.appendChild(areaImgElement);
                        }

                        // 處理返回的辨識結果                    
                        const emotion = data.emotion_result; // 獲取情緒類別的地方
                        addEmotionMessages(emotion); // 調用 addEmotionMessages 函數

                        const level = data.level;
                        addlevelMessages(level);

                        const area = data.area_result;
                        addareaMessages(area);

                        // 處理返回的肌肉名稱和顏色
                        const muNames = data.muresult ? data.muresult.split(', ') : []; // 檢查 muresult 是否存在
                        const muColors = data.mu_colors ? data.mu_colors.split(', ') : []; // 檢查 mu_colors 是否存在

                        // 將名稱和顏色組合為對象，確保長度匹配
                        const muscles = muNames.map((name, index) => ({
                            name: name,
                            color: muColors[index] || "#000000" // 如果顏色未定義，使用預設顏色
                        }));

                        // 傳遞肌肉數據到 addmuMessages 函數
                        addmuMessages(muscles);

                        // 隱藏 loading 動畫
                        loadingElement.style.display = 'none';
                    })
                    .catch(error => {
                        console.error('Error receiving image:', error);
                        loadingElement.style.display = 'none'; // 發生錯誤時隱藏 loading 動畫
                    });

            }, 'image/jpeg'); // 修改圖像格式為 JPEG
        }

        // 監聽上傳頁面 "分析" 按鈕的點擊事件
        document.getElementById('analyzeBtn-upload').addEventListener('click', () => {
            const UploadSamplePreviewBox = document.getElementById('UploadSamplePreview');
            const hiddenFileInput = document.getElementById('hiddenFileInput');
            const uploadedImage = hiddenFileInput.files[0];
            const loadingElement = document.getElementById('loading'); // 取得 loading 元素
            UploadSamplePreviewBox.innerHTML = '';
            // 獲取選擇的模型標籤
            const selectedModelLabel = document.getElementById('modelSelect').value;

            if (uploadedImage) {
                loadingElement.style.display = 'flex';  // 顯示 loading 動畫

                const reader = new FileReader();
                reader.onload = function (event) {
                    const img = new Image();
                    img.src = event.target.result;
                    img.onload = function () {
                        const canvas = document.createElement('canvas');
                        const size = Math.min(img.width, img.height); // 截取預覽圖像區域
                        canvas.width = size;
                        canvas.height = size; // 正方形
                        const ctx = canvas.getContext('2d');
                        const x = (img.width - size) / 2; // 計算截取的起點坐標
                        const y = (img.height - size) / 2;

                        ctx.save();
                        ctx.translate(size, 0);
                        ctx.scale(-1, 1);
                        ctx.drawImage(img, x, y, size, size, 0, 0, size, size);
                        ctx.restore();

                        canvas.toBlob(blob => {
                            const formData = new FormData();
                            formData.append('file', blob, 'uploaded_image.jpg');
                            formData.append('model_label', selectedModelLabel);

                            fetch("https://8638-210-59-96-137.ngrok-free.app/emotion_recognition", {
                                method: 'POST',
                                body: formData
                            })
                                .then(response => response.json())
                                .then(data => {
                                    const ResultMuImg = document.getElementById('result-muimg');
                                    ResultMuImg.innerHTML = '';

                                    const ResultAreaImg = document.getElementById('result-areaimg');
                                    ResultAreaImg.innerHTML = '';

                                    // 顯示臉部肌肉圖像
                                    if (data.muscle_image) {
                                        const muscleImgBase64 = data.muscle_image;
                                        const muscleImgElement = document.createElement('img');
                                        muscleImgElement.src = 'data:image/jpeg;base64,' + muscleImgBase64.replace(/"/g, '');
                                        ResultMuImg.appendChild(muscleImgElement);
                                    }

                                    // 顯示臉部區域面積圖像
                                    if (data.area_image) {
                                        const areaImgBase64 = data.area_image;
                                        const areaImgElement = document.createElement('img');
                                        areaImgElement.src = 'data:image/jpeg;base64,' + areaImgBase64.replace(/"/g, '');
                                        ResultAreaImg.appendChild(areaImgElement);
                                    }


                                    const emotion = data.emotion_result;
                                    addEmotionMessages(emotion);

                                    const level = data.level;
                                    addlevelMessages(level);

                                    const area = data.area_result;
                                    addareaMessages(area);

                                    const muNames = data.muresult ? data.muresult.split(', ') : [];
                                    const muColors = data.mu_colors ? data.mu_colors.split(', ') : [];

                                    const muscles = muNames.map((name, index) => ({
                                        name: name,
                                        color: muColors[index] || "#000000"
                                    }));

                                    addmuMessages(muscles);

                                    loadingElement.style.display = 'none'; // 隱藏 loading 動畫
                                })
                                .catch(error => {
                                    console.error('Error receiving image:', error);
                                    loadingElement.style.display = 'none'; // 隱藏 loading 動畫
                                });

                        }, 'image/jpeg');
                    };
                };
                reader.readAsDataURL(uploadedImage);

                hiddenFileInput.value = '';
            } else {
                alert('請先上傳圖像！');
            }
        });


        // 當按下攝像頭添加樣本，顯示功能頁面
        AddSampleBtnCamera.addEventListener("click", () => {
            ModelSettingsBtn.style.display = "block";
            SampleBtnContainer.style.display = "none";  // 隱藏添加樣本頁面
            AddSampleUpload.style.display = "none";  // 隱藏上傳添加樣本頁面		
            AddSampleCamera.style.display = "flex";  // 顯示攝像頭添加頁面
            dropdownMenu.classList.remove("show"); // 隱藏預留按鈕
        });

        // 當按下上傳添加樣本，顯示功能頁面
        AddSampleBtnUpload.addEventListener("click", () => {
            ModelSettingsBtn.style.display = "block";
            SampleBtnContainer.style.display = "none";  // 隱藏添加樣本頁面
            AddSampleCamera.style.display = "none";  // 隱藏攝像頭添加頁面		
            AddSampleUpload.style.display = "flex";  // 顯示上傳添加樣本頁面
            dropdownMenu.classList.remove("show"); // 隱藏預留按鈕
        });


        // 當按下關閉按鈕時，隱藏模型設定按鈕 & 上傳添加樣本 & 攝像頭添加頁面
        const handleCloseButtonClick = () => {
            ModelSettingsBtn.style.display = "none";  // 隱藏模型設定按鈕
            AddSampleCamera.style.display = "none";  // 隱藏攝像頭添加頁面
            AddSampleUpload.style.display = "none";  // 隱藏上傳添加樣本頁面
            SampleBtnContainer.style.display = "flex";  // 顯示添加樣本頁面
            dropdownMenu.classList.remove("show"); // 隱藏預留按鈕
        };

        CloseBtnCamera.addEventListener("click", handleCloseButtonClick);
        CloseBtnUpload.addEventListener("click", handleCloseButtonClick);

        // 當按下預留按鈕時，顯示選單
        ReservedMenuBtn.addEventListener("click", (event) => {
            event.stopPropagation();  // 阻止點擊事件冒泡到body
            dropdownMenu.classList.toggle("show");// 顯示預留按鈕
            ModelSettingsMenu.classList.remove("show"); // 隱藏設定選單
        });

        // 當按下模型設定按鈕時，顯示選單
        ModelSettingsBtn.addEventListener("click", (event) => {
            event.stopPropagation();  // 阻止點擊事件冒泡到body
            ModelSettingsMenu.classList.toggle("show");
            dropdownMenu.classList.remove("show"); // 隱藏預留按鈕
        });

        // 當點擊頁面空白處時，隱藏預留按鈕和模型設定選單
        document.addEventListener("click", (event) => {
            if (!dropdownMenu.contains(event.target) && !ReservedMenuBtn.contains(event.target)) {
                dropdownMenu.classList.remove("show");
            }
            if (!ModelSettingsMenu.contains(event.target) && !ModelSettingsBtn.contains(event.target)) {
                ModelSettingsMenu.classList.remove("show");
            }
        });

        // 監聽鼠標進入事件，顯示提示訊息並追蹤游標位置
        ModelSettingsTipIcon.addEventListener('mouseenter', function (event) {
            ModelSettingsTip.style.display = 'block'; // 顯示提示訊息
        });

        // 監聽鼠標離開事件，隱藏提示訊息
        ModelSettingsTipIcon.addEventListener('mouseleave', function () {
            ModelSettingsTip.style.display = 'none'; // 隱藏提示訊息
        });

        // 點擊自訂按鈕時觸發隱藏的 input file 按鈕
        UploadBtn.addEventListener('click', function () {
            hiddenFileInput.click();
        });

        // 當檔案被選擇後觸發這個事件
        hiddenFileInput.addEventListener('change', function (e) {
            const file = hiddenFileInput.files[0];
            const reader = new FileReader();

            // 確保檔案存在，並且是圖片檔案
            if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
                reader.readAsDataURL(file); // 讀取檔案為 Data URL

                // 當檔案讀取完成後，將其內容顯示為圖片
                reader.onload = function (e) {
                    const img = new Image();
                    img.src = e.target.result;
                    img.style.maxWidth = '100%'; // 控制圖片大小以適應頁面
                    img.style.height = '100%'; // 使圖片高度填滿容器
                    img.style.objectFit = 'contain'; // 確保圖片不會變形
                    UploadSamplePreview.innerHTML = ''; // 清空之前的內容
                    UploadSamplePreview.appendChild(img); // 顯示圖片           	
                }
            } else {
                fileDisplayArea.innerText = '請上傳 PNG 或 JPG 格式的圖片。';
            }
        });

    </script>
</body>

</html>
"""


