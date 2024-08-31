import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as image
import torch
import japanize_matplotlib
import seaborn as sns
import requests
from io import BytesIO


#画像サンプル
ok_img = image.open('images/OKsample.jpg')
ok_img_multi = image.open('images/OKsampleMulti.jpg')
ng_img = image.open('images/NGsample.jpg')

not_smailing = image.open('images/猫NO笑顔イラスト.jpg')
little_smailing = image.open('images/猫笑顔イラスト_小笑い.jpg')
big_smailing = image.open('images/猫笑顔イラスト_大笑い.jpg')
ng_catFace = image.open('images/猫NGイラスト.jpg')

# モデルの予測結果によって表示内容を変更する
def custom(num):
    if num == 3:
        word = '## 小笑い'
        image = little_smailing
    elif num == 4:
        word = '## 大笑い'
        image = big_smailing
    else:
        word = '## Not笑顔'
        image = not_smailing
    return word,image

#笑顔率を計算する値を編集する 笑顔以外の判定の時は合算、笑顔の時は値をそのまま利用
#予測時の画像は１件のみなので戻り値は1行X列
def calc_smile_percent(pred,max_index):
    if max_index == 3 or max_index == 4:
        result = pred[0][max_index]
    else:
        result = pred[0][0] + pred[0][1] + pred[0][2]
    
    return result

#猫の顔の座標をもとに顔部分を切り取る
def crop_image(detection_result):
    # アップロードされた画像を開く
    img_original = image.open(uploaded_file)

    #取得した座標を変数に振り分け
    x_min, y_min, x_max, y_max = map(int, detection_result)

    #座標をもとに猫の顔画像を切り抜き
    cropped_img = img_original.crop((x_min, y_min, x_max, y_max))

    # 切り抜いた画像をBytesIOに保存
    cropped_img_bytes = BytesIO()
    cropped_img.save(cropped_img_bytes, format='JPEG')
    cropped_img_bytes.seek(0)

    return cropped_img,cropped_img_bytes

#----画面コントロール用のステータスの設定----------------------------------------------------------------------

#初期設定
if 'is_uploaded' not in st.session_state:
    st.session_state.is_uploaded = False
if 'is_started' not in st.session_state:
    st.session_state.is_started = False
if 'previous_file' not in st.session_state:
    st.session_state.previous_file = None

#アップロードファイルが変更される際にステータスをFalseに設定する
def reset_is_uploaded():
    st.session_state.is_uploaded = False

#評価ボタン押下時にステータスをTrueに設定する
def change_is_started():
    st.session_state.is_started = True

#----サイドバーの設定----------------------------------------------------------------------

st.sidebar.header('画像アップロード')

#画像入力例:OKの表示
st.sidebar.write('画像入力例：OK(猫の顔が正面を向いている)')
st.sidebar.image(ok_img,caption = '画像入力例：OK' , use_column_width = True) #use_column_width:アプリのレイアウトに合わせて画像サイズを自動調整


#画像入力例:OK(複数匹)の表示
st.sidebar.write('画像入力例：OK(複数匹でも可。但し、判別する顔は自動で1つ選択されます。)')
st.sidebar.image(ok_img_multi,caption = '画像入力例：OK' , use_column_width = True) #use_column_width:アプリのレイアウトに合わせて画像サイズを自動調整

#画像入力例:NGの表示
st.sidebar.write('画像入力例：NG(猫の顔が正面を向いてない)')
st.sidebar.image(ng_img,caption = '画像入力例：NG' , use_column_width = True) #use_column_width:アプリのレイアウトに合わせて画像サイズを自動調整


#画像入力例:NGの表示
# st.sidebar.write('画像入力例：NG(猫の顔が分からない、顔以外の情報が含まれる')
# st.sidebar.image(ng_img,caption = '画像入力例：NG' , use_column_width = True) #use_column_width:アプリのレイアウトに合わせて画像サイズを自動調整

#アプロードされた猫画像を変数に格納
uploaded_file = st.sidebar.file_uploader('猫画像アップロード',type='jpg',on_change = reset_is_uploaded)


#----メインページ --------------------------------------------------------------------------------------------------------------------

# ①メインページで利用する変数の設定

#st_result = None

# ②メインページの設定

st.title('Is your cat smiling ??')
st.header('猫の顔画像をアップロードすることでどれくらい笑顔なのかを評価できます！')


#コンテナを２つ設定(1つ目は画像評価前の領域、2つは画像評価結果の領域)
cont1 = st.container()
cont2 = st.container()

#コンテナ2を2分割する設定
col1,col2 = cont2.columns(2)

#画像がアップロードされた際の処理

if uploaded_file is not None and st.session_state.is_uploaded == False:

    #アップロード状態をTrueに設定 ←　本処理は画像ファイルを再アップロードした時のみ処理が実行されるようにする為
    st.session_state.is_uploaded = True

    #---------------------------------------------------------------------------------
    #① FastAPIへの画像データの受渡と猫の顔のboundingboxの座標の取得
    files = {"file" : uploaded_file.getvalue()}
    try:
        response = requests.post("http://localhost:8000/detect_catFace", files=files)
        response.raise_for_status()  # HTTPステータスコードが4xx/5xxのとき例外を発生させる
        detection_result = response.json()["detection"]
    except requests.exceptions.Timeout:
        st.error("The request timed out")
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")  # HTTPエラーが発生した場合の処理
    except requests.exceptions.RequestException as err:
        st.error(f"An error occurred: {err}")

    #--------------------------------------------------------------------------------
    #② 戻り値の座標をもとに猫の顔を切り取って表示する
    # 猫の顔の座標が正しく取得できている場合
    if any(detection_result) :
        #猫の顔の座標をもとに画像を切り抜き
        cropped_img,cropped_file = crop_image(detection_result)

        cont1.write('この画像で評価しますか？')
        cont1.image(cropped_file,use_column_width = True)

        #評価ボタン押下処理で利用するように検知した画像はセッション変数に格納
        st.session_state.cropped_file = cropped_file
        st_result = cont1.button('評価スタート',type='primary',on_click=change_is_started)

    # 猫の顔の座標が取得できなかった場合(座標が全て0で返ってきた時)
    else:
        cont1.write('### 猫の顔を検知できませんでした。')
        cont1.write('### 別の画像にてお試し下さい')
        cont1.image(ng_catFace,use_column_width = True)

#評価ボタンが押下された時の処理
if st.session_state.is_started:

    #評価対象の画像を表示しておく為の処理
    cont1.write('この画像で評価しますか？')
    cont1.image(st.session_state.cropped_file,use_column_width = True)

    #---------------------------------------------------------------------------------
    #① FastAPIへの画像データの受渡と予測結果の取得
    files = {"file" : st.session_state.cropped_file.getvalue()}
    try:
        response = requests.post("http://localhost:8000/upload-image", files=files)
        response.raise_for_status()  # HTTPステータスコードが4xx/5xxのとき例外を発生させる
        prediction_result = response.json()["prediction"]
    except requests.exceptions.Timeout:
        st.error("The request timed out")
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")  # HTTPエラーが発生した場合の処理
    except requests.exceptions.RequestException as err:
        st.error(f"An error occurred: {err}")
    finally:
        # 各種ステータスはエラーになった時含めて必ずリセット
        st.session_state.is_uploaded = False
        st.session_state.is_started = False

   #---------------------------------------------------------------------------------
   #② FastAPIからの予測結果を画面表示用に編集

    # 各行に対して最大値を持つ列のインデックスを取得
    # 予測結果が同じ値の時、インデックス番号が後ろのものから取得
    reversed_index = np.argmax(prediction_result[0][::-1])
    max_indices = len(prediction_result[0]) - 1 - reversed_index

    #行の最大値のインデックスから笑顔率の数字を取得する
    #予測時の画像は１件のみなので戻り値は1行X列想定
    smile_percent = calc_smile_percent(prediction_result,max_indices)

    #%表示対応にした上で、文字型に変換
    smaile_percent_mod = str(int(round(smile_percent,2) * 100))

    #グラフの作成
    #データフレームに変換
    df = pd.DataFrame(prediction_result,columns = ['真顔','怒り','寝顔','小笑い','大笑い'])

    #棒グラフの作成
    fig,ax=plt.subplots(figsize = (4,5))

    sns.barplot(df,saturation = 0.7,palette = 'Set2',ax=ax)

    #データフレームの列名をもとに、ラベルをグラフ上部に設定
    for p, label in zip(ax.patches, df.columns):
        ax.annotate(label,
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                color = 'red',
                fontsize=12)

    plt.axis('off')

    #------------------------------------------------------------------------------
    #③画面の表示内容の設定

    #画面２分割の左側(笑顔率のグラフ表示)
    col1.write('## 評価結果')

    #グラフの表示
    col1.pyplot(fig)

    #画面2分割の右側(笑顔率判定結果の表示)
    smaile_specify,smaile_image = custom(max_indices)

    col2.write(smaile_specify + ':' +  smaile_percent_mod  + '%')
    col2.image(smaile_image,use_column_width = True)

    
