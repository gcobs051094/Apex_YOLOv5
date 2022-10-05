from aim.apex_aim import lock
from aim.screen_inf import grab_screen_mss, grab_screen_win32, get_parameters
from aim.cs_model import load_model
import cv2
import win32gui
import win32con
import torch
import numpy as np

from aim.verify_args import verify_args
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import pynput
import argparse
import time
import os
from simple_pid import PID
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='weights/best.pt', help='模型權重')
parser.add_argument('--imgsz', type=int, default=640, help='訓練模型時的影像大小')
parser.add_argument('--conf-thres', type=float, default=0.5, help='信心分數')
parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU分數')
parser.add_argument('--use-cuda', type=bool, default=True, help='啟動cuda') 
parser.add_argument('--show-window', type=bool, default=False, help='是否顯示real time檢測視窗')
parser.add_argument('--top-most', type=bool, default=True, help='是否窗口置頂')
parser.add_argument('--resize-window', type=float, default=1, help='窗口大小調整')
parser.add_argument('--thickness', type=int, default=2, help='偵測BoundingBox粗細')
parser.add_argument('--show-fps', type=bool, default=False, help='是否顯示fps')
parser.add_argument('--show-label', type=bool, default=False, help='是否顯示偵測label')
parser.add_argument('--use_mss', type=str, default=False, help='是否使用mss截屏；为False時使用win32截屏')
parser.add_argument('--region', type=tuple, default=(0.4, 0.4), help='檢測範圍(全螢幕比例)')
parser.add_argument('--hold-lock', type=bool, default=True, help='鎖敵模式；True為按住，False為切換')
parser.add_argument('--lock-sen', type=float, default= 3.0, help='鎖敵幅度系數,遊戲中靈敏度(建議不要調整)')
parser.add_argument('--lock-smooth', type=float, default=6, help='鎖敵平滑系数；越大越smooth的移動到目標(過大會反應慢)')
parser.add_argument('--lock-button', type=str, default='left', help='鎖敵按鍵(滑鼠)；left(滑鼠左鍵)、right(滑鼠右鍵)、x1(側鍵下)、x2(側鍵上)')
parser.add_argument('--head-first', type=bool, default=False, help='是否優先瞄頭')
parser.add_argument('--lock-tag', type=list, default=[0], help='對應標籤；person(若模型不同請自行修改對應標籤)')
parser.add_argument('--lock-choice', type=list, default=[0], help='目標選擇；决定鎖定的目標，從自己的標籤中選')

args = parser.parse_args()

'------------------------------------------------------------------------------------'

verify_args(args)

cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'

args.model_path = cur_dir + args.model_path
args.lock_tag = [str(i) for i in args.lock_tag]
args.lock_choice = [str(i) for i in args.lock_choice]

device = 'cuda' if args.use_cuda else 'cpu'
half = device != 'cpu'
imgsz = args.imgsz

conf_thres = args.conf_thres
iou_thres = args.iou_thres

top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))

monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

model = load_model(args)
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

lock_mode = False
team_mode = True
lock_button = eval('pynput.mouse.Button.' + args.lock_button)

mouse = pynput.mouse.Controller()

#PID係數可調整
#PID(P, I, D)
#P: 加快系統反映。輸出值較快，但越大越不穩定
#I: 積分。用於穩定誤差
#D: 微分。提高系統的動態性能
#以下為個人使用參數可供參考
pidx = PID(1.2, 8, 0.03, setpoint=0, sample_time=0.001,)
pidy = PID(1.22, 3, 0.001, setpoint=0, sample_time=0.001,)
pidx.output_limits = (-4000 ,4000)
pidy.output_limits = (-3000 ,3000)

if args.show_window:
    cv2.namedWindow('aim', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('aim', int(len_x * args.resize_window), int(len_y * args.resize_window))


def on_click(x, y, button, pressed):
    global lock_mode
    if button == lock_button:
        if args.hold_lock:
            if pressed:
                lock_mode = True
                print('鎖定...')
            else:
                lock_mode = False
                print('鎖定模式off')
        else:
            if pressed:
                lock_mode = not lock_mode
                print('鎖定模式', 'on' if lock_mode else 'off')

listener = pynput.mouse.Listener(on_click=on_click)
listener.start()

print('程式已開啟~')
t0 = time.time()
cnt = 0

while True:

    if cnt % 20 == 0:
        top_x, top_y, x, y = get_parameters()
        len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
        top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
        monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}
        cnt = 0

    if args.use_mss:
        img0 = grab_screen_mss(monitor)
        img0 = cv2.resize(img0, (len_x, len_y))
    else:
        img0 = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
        img0 = cv2.resize(img0, (len_x, len_y))

    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    #print(pred[4])  //預測到的信心分數
    aims = []
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # bbox:(tag, x_center, y_center, x_width, y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                aims.append(aim)

        if len(aims):
            if lock_mode:
                lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy)

        if args.show_window:
            for i, det in enumerate(aims):
                tag, x_center, y_center, width, height = det
                x_center, width = len_x * float(x_center), len_x * float(width)
                #print("width:" , width)
                #print("x_center:", x_center)
                y_center, height = len_y * float(y_center), len_y * float(height)
                top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                #print("top_left:", top_left)
                bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                #print("bottom_right:", bottom_right)
                cv2.rectangle(img0, top_left, bottom_right, (0, 255, 0), thickness=args.thickness)
                if args.show_label:
                    cv2.putText(img0, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 0, 0), 4)

    if args.show_window:
        if args.show_fps:
            cv2.putText(img0,"FPS:{:.1f}".format(1. / (time.time() - t0)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 235), 4)
            #cv2.putText(img0, "lock:{:.1f}".format(lock_mode), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 235), 4)
            #cv2.putText(img0, "team:{:.1f}".format(team_mode), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 235), 4)
            print(1. / (time.time() - t0))
            t0 = time.time()

        cv2.imshow('aim', img0)

        if args.top_most:
            hwnd = win32gui.FindWindow(None, 'aim')
            CVRECT = cv2.getWindowImageRect('aim')
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        cv2.waitKey(1)
    pidx(0)
    pidy(0)
    cnt += 1
