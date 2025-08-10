import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from android_tool import AndroidTool
from argparses import args
from dqnAgent import DQNAgent
from getReword import GetRewordUtil
from globalInfo import GlobalInfo

from wzry_env import Environment
from onnxRunner import OnnxRunner

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

rewordUtil = GetRewordUtil()
tool = AndroidTool()
tool.show_scrcpy()
# tool.show_action_log()
env = Environment(tool, rewordUtil)

agent = DQNAgent()

# 控制线程的全局变量
training_running = False
data_collection_running = False
training_thread = None
data_collection_thread = None

def data_collector():
    global data_collection_running
    data_collection_running = True
    print("数据收集线程已启动")
    
    while data_collection_running:
        # 获取当前的图像
        state = tool.screenshot_window()
        # 保证图像能正常获取
        if state is None:
            time.sleep(0.01)
            continue
        # cv2.imwrite('output_image.jpg', state)
        # 初始化对局状态 对局未开始
        globalInfo.set_game_end()
        # 判断对局是否开始
        checkGameStart = start_check.get_max_label(state)

        if checkGameStart == 'started':
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()

            # 对局开始了，进行训练
            while globalInfo.is_start_game() and data_collection_running:
                # 获取预测动作
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)
                print(info, reward)

                # 对局结束
                if done == 1:
                    print("-------------------------------对局结束-----------------------------------")
                    globalInfo.set_game_end()
                    break

                # 追加经验
                globalInfo.store_transition_dqn(state, action, reward, next_state, done)

                state = next_state

        else:
            if data_collection_running:
                print("对局未开始")
            time.sleep(0.1)
    
    print("数据收集线程已停止")


def train_agent():
    global training_running
    training_running = True
    count = 1
    print("训练线程已启动")
    
    while training_running:
        if not globalInfo.is_memory_bigger_batch_size_dqn():
            time.sleep(1)
            continue
        print("training")
        agent.replay()
        if count % args.num_episodes == 0:
            agent.save_model('src/wzry_ai.pt')
        count = count + 1
        if count >= 100000:
            count = 1
    
    print("训练线程已停止")


def start_training():
    """开始训练"""
    global training_thread, data_collection_thread
    
    if training_thread is None or not training_thread.is_alive():
        training_thread = threading.Thread(target=train_agent)
        training_thread.start()
        print("训练已启动")
    else:
        print("训练已经在运行中")
    
    if data_collection_thread is None or not data_collection_thread.is_alive():
        data_collection_thread = threading.Thread(target=data_collector)
        data_collection_thread.start()
        print("数据收集已启动")
    else:
        print("数据收集已经在运行中")

def pause_training():
    """暂停训练"""
    global training_running, data_collection_running
    
    training_running = False
    data_collection_running = False
    print("训练和数据收集已暂停")

def create_gui():
    """创建GUI界面"""
    root = tk.Tk()
    root.title("王者荣耀AI训练控制面板")
    root.geometry("300x200")
    
    # 设置窗口居中
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 300) // 2
    y = (screen_height - 200) // 2
    root.geometry(f"300x200+{x}+{y}")
    
    # 创建标题标签
    title_label = tk.Label(root, text="王者荣耀AI训练控制", font=("Arial", 14, "bold"))
    title_label.pack(pady=20)
    
    # 创建按钮框架
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    # 开始训练按钮
    start_button = tk.Button(button_frame, text="开始训练", command=start_training, 
                           width=12, height=2, bg="#4CAF50", fg="white", font=("Arial", 10))
    start_button.pack(side=tk.LEFT, padx=10)
    
    # 暂停训练按钮
    pause_button = tk.Button(button_frame, text="暂停训练", command=pause_training, 
                           width=12, height=2, bg="#f44336", fg="white", font=("Arial", 10))
    pause_button.pack(side=tk.LEFT, padx=10)
    
    # 状态标签
    status_label = tk.Label(root, text="状态: 等待开始", fg="blue")
    status_label.pack(pady=10)
    
    def update_status():
        """更新状态显示"""
        if training_running:
            status_label.config(text="状态: 训练中...", fg="green")
        else:
            status_label.config(text="状态: 已暂停", fg="red")
        root.after(1000, update_status)
    
    update_status()
    
    # 窗口关闭时的处理
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            pause_training()
            root.quit()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    return root

if __name__ == '__main__':
    # 创建并运行GUI
    gui = create_gui()
    gui.mainloop()
