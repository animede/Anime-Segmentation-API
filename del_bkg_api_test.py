import os
import argparse
import cv2
from PIL import Image
import glob
import pickle
from datetime import datetime
import requests
import numpy as np

def main():
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   type=str, default='../../dataset/anime-seg/test2',  help='input data dir')
    parser.add_argument("--host",   type=str,  default="0.0.0.0",  help="サービスを提供するip アドレスを指定。")
    parser.add_argument("--port",   type=str,  default="8007",  help="サービスを提供するポートを指定。")
    parser.add_argument("--test",   type=int,  default=1,   help="TESTモード　1 : OpenCV  2  : pillow")
    parser.add_argument("--out",   type=str,  default="result/",   help="out")
    opt = parser.parse_args()

    url="http://"+opt.host+":"+opt.port+"/del_bkg/"
    url="http://0.0.0.0:8007/del_bkg/" 

# ***********************    テストプログラム　**********************  
    if opt.test==1:  # in/out =  pillow imaeg test
        if not os.path.exists(opt.out):
            os.mkdir(opt.out)
        print("start_time=",datetime.now())
        for i, path in enumerate(glob.glob(f"{opt.data}/*.*")):
            img = Image.open(path) # pil で 画像を開く
            pil_img  = del_bkg_out(img , "pil" , url )    # <<<<<<<<<<<<<<<<<<<<< del_bkg_out()
            pil_img.save(f'{opt.out}/{i:06d}.png')
        print("end_time=",datetime.now())
        pil_img.show()
    
    if opt.test==2:  # in/out = opeCV  imaeg test
        if not os.path.exists(opt.out):
            os.mkdir(opt.out)
        print("start_time=",datetime.now())
        for i, path in enumerate(glob.glob(f"{opt.data}/*.*")):
            img = cv2.imread(path, cv2.IMREAD_COLOR) #OpenCV で 画像を開く
            cv_img= del_bkg_out(img , "cv" , url )    # <<<<<<<<<<<<<<<<<<<<< del_bkg_out()
            cv2.imwrite(f'{opt.out}/{i:06d}.jpg', cv_img)
        print("Channel=",cv_img.shape[2])#OpenCV形式のチャンネル数を確認
        print("end_time=",datetime.now())
        cv2.imshow("cv_img", cv_img)
        cv2.waitKey()

# *********************** 　汎用背景削除　from del_bkg_api  import del_bkg_out 
def del_bkg_out(img , img_mode, url="http://0.0.0.0:8007/del_bkg/"):
    if img_mode=="pil": #pilの場合はcvに変換
        img= np.array( img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #カラーチャンネル変換
   # 以下cv/pil共通      バイナリデータをPOSTリクエストで送信
    _, img_encoded = cv2.imencode('.jpg', img)
    response = requests.post(url, files={"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg"),"mode":(None,img_mode)})
    all_data =response.content
    frame_data = (pickle.loads(all_data))#元の形式にpickle.loadsで復元 #形式はimg_mode指定の通り
    return frame_data #形式はimg_mode指定の通り

if __name__ == "__main__":
    main()
