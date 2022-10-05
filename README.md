#Apex_YOLOv5
---
## Apex AI鎖定敵人輔助工具

* YOLOv5 偵測遊戲畫面 + 滑鼠控制鎖敵
* 輔助控制，依舊要求玩家觀念及滑鼠控制
* 非高階顯卡效果不好(顯卡影響YOLOv5 frame數)
* 可自行訓練權重(此best.pt為YOLOv5s模型權重)
---
### 使用
1. 下載整包github code
2. 安裝anaconda
3. 創建環境
4. 為虛擬環境安裝套件
```
pip install -r requirements.txt
``` 
5. 開啟apex.py設定參數
6. 更改activate.bat中的環境啟動路徑及遊戲資料夾 
```
CALL C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3\envs\APEXTorch38
cd G:\Side_Pro\APEX-yolov5-aim-assist-main
G:
python apex.py
```
7. 點擊activate.bat
8. 待cmd出現「程式已開啟」，按下lock-button中設定的按鍵即可使用

![Uploading file..._wmonob86c]()
[影片demo](https://drive.google.com/file/d/1FpFkIotOpqvh6GaGtuOTnhy4ACnuRaFa/view?usp=sharing)
