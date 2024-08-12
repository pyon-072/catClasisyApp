from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import pickle
from models.net import Net
from pydantic import BaseModel

import supervision as sv
from inference_sdk import InferenceHTTPClient

app = FastAPI()

# 追加コード: 画像変換の定義
transform = transforms.Compose([
    transforms.Resize(256),     # 256サイズに変換
    transforms.CenterCrop(224), # 中心部を224サイズに切り抜き
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 学習済みのモデルの読み込み
#model = pickle.load(open('models/model_catClasify', 'rb'))

#ネットワークの準備(推論モードでインスタンス化)
net = Net().cpu().eval()

# 重みの読み込み(cpuでの読込)
net.load_state_dict(torch.load('models/catClassify.pt', map_location=torch.device('cpu')))

#推論用の関数
def predictImage(model,image):
    with torch.no_grad():
        y_pred = model(image)

    return y_pred

# roboflowの猫の顔画像検知用の学習モデルにアクセスするURLとAPIキーの設定
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="w4HneTYrMLJmsGFVejM9"
)

#roboflowのAPIの結果から信頼スコアが最も高い猫の顔検知の座標を取得する
def get_bbox(detections):

    #値を格納する変数を初期化
    bbox = []
    confidence = 0

    #値の確認
    for i in range(0,len(detections)):
        #class_nameがcat_faceの時のみ処理対象
        if detections.data['class_name'][i] == 'catFace':
            #信頼スコア最大のもののbbox座標を取得
            if confidence < detections.confidence[i]:
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]

    # 猫の顔の座標を取得出来ている時は変換処理を実施
    if len(bbox) != 0:
        # 座標データを取得
        x_min, y_min, x_max, y_max = map(int, bbox)

        #　正方形になるように座標を変換
        x_new_min, y_new_min, x_new_max, y_new_max = make_square(x_min, y_min, x_max, y_max)

        return x_new_min, y_new_min, x_new_max, y_new_max

    else:
        #猫の顔の座標を取得できていないときは全て1を返す
        return 0,0,0,0


#画像を正方形で抽出できるように座標を変換する関数
def make_square(x_min, y_min, x_max, y_max):
    # 幅と高さを計算
    width = x_max - x_min
    height = y_max - y_min

    # 正方形の一辺の長さを計算（幅と高さの大きい方に合わせる）
    side_length = max(width, height)

    # 中心を計算
    center_x = x_min + width // 2
    center_y = y_min + height // 2

    # 新しい正方形のバウンディングボックスの座標を計算
    new_x_min = max(center_x - side_length // 2, 0)  # 画像の境界を超えないように調整
    new_y_min = max(center_y - side_length // 2, 0)
    new_x_max = new_x_min + side_length
    new_y_max = new_y_min + side_length

    return new_x_min, new_y_min, new_x_max, new_y_max

# ルートディレクトリへの GET で 接続確認結果の表示
@app.get("/")
def root():
    return  {"message": "Connected"}

@app.post("/upload-image") #このURLはStreamlitと一致させること
async def upload_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        # PILを使って画像データを読み込む
        image = Image.open(BytesIO(image_data))

        # 画像データを変換する
        transformed_image = transform(image)

        #モデルは入力データを4次元としているので4次元に変換
        tranformed_4d = torch.reshape(transformed_image,[1,3,224,224])

        #推論の実施
        y_pred = predictImage(net,tranformed_4d)

        #推論結果をソフトマックス
        softmax_tensor = torch.softmax(y_pred, dim=1)

        #ndarray形式に変換
        prediction_list = softmax_tensor.numpy()

        #list形式にして受け渡す
        return {"prediction": prediction_list.tolist()}

    except Exception as e:
        # エラー発生時にHTTP 500エラーを返す
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/detect_catFace") #このURLはStreamlitと一致させること
async def detect_catFace(file: UploadFile = File(...)):
    try:
        image_data_original = await file.read()

        # PILを使って画像データを読み込む
        image_original = Image.open(BytesIO(image_data_original))

        #roboflowの学習済APIにアクセスして猫の顔検知を実施
        result = CLIENT.infer(image_original, model_id="detect_catface/4")
        detections = sv.Detections.from_inference(result)

        #猫の顔の座標を取得する(猫の顔が検知できていない時は0が返る)
        x_min, y_min, x_max, y_max = get_bbox(detections)

        #座標をリスト形式に変換
        result_list = [x_min, y_min, x_max, y_max]

        #list形式にして受け渡す
        return {"detection": result_list}

    except Exception as e:
        # エラー発生時にHTTP 500エラーを返す
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=" Roboflow API Error : cannot detect catFace")

