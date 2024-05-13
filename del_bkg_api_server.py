import os
import argparse
import cv2
from PIL import Image
import torch
import numpy as np
from torch.cuda import amp
import pickle
from datetime import datetime
from train import AnimeSegmentation
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import Response
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='isnet_is', choices=["isnet_is", "isnet", "u2net", "u2netl", "modnet"],  help='net name')
parser.add_argument('--ckpt', type=str, default='isnetis.ckpt', help='model checkpoint path')
parser.add_argument('--img-size', type=int, default=1024, help='hyperparameter, input image size of the net')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')
parser.add_argument("-i","--host",     type=str,  default="0.0.0.0",  help="サービスを提供するip アドレスを指定。")
parser.add_argument("-p","--port",   type=int,  default=50000,    help="サービスを提供するポートを指定。")
opt = parser.parse_args()

# イニシャライズ
img_size=opt.img_size
device = torch.device(opt.device)
model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device)
model.eval()
model.to(device)
        
# =============    FastAPI  ============
app = FastAPI()

@app.post("/del_bkg/")
async  def del_bkg(file: UploadFile = File(...), mode:str  = Form(...)):
       print("mode=",mode)
       if mode=="cv":    #CV形式の時
            file_contents = await file.read()
            nparr = np.frombuffer(file_contents, np.uint8) # バイナリデータをNumPy配列に変換
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)      # OpenCVで画像として読み込む
            #cv2.imwrite(f'{opt.out}/{i:06d}.jpg', imag)
       elif file:    #PIL形式のとき
           image_data = file.file.read()
           img = Image.open(BytesIO(image_data))  # バイナリデータをPIL形式に変換
       out_img , mask = del_bkg_out(img ,mode)
       frame_data = pickle.dumps(out_img, 5)  # tx_dataはpklデータ、イメージのみ返送
       return Response(content=frame_data, media_type="application/octet-stream")

def del_bkg_out(img , img_mode): #del_bkg_out  背景削除     # Input :  img=image , img_mode="pil" or "cv"
        if  img_mode=="pil":
            img= np.array( img, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#カラーチャンネル変換
        mask = get_mask(model, img , s=img_size) # mask
        img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8) # イメージにマスクを追加mask
        pil_img= Image.fromarray(img)
        if  img_mode=="pil":
            return pil_img , mask  #imgはpillow、maskはcv2
        else:
            new_image = np.array(pil_img, dtype=np.uint8)
            img = cv2.cvtColor(new_image , cv2.COLOR_RGBA2BGRA)#opencv形式
            return img , mask  #imgとmaskはcv2
        
#+++++++++++++++++++ infference  ++++++++++++++++++++
def get_mask(model, input_img,  s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
