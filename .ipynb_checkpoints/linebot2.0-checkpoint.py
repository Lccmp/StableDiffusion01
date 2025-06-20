import os
import shutil
import subprocess
import threading
import torch
import io
import json
import time
import logging
import traceback

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage,
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent, FlexSendMessage,
    BubbleContainer, BoxComponent, TextComponent, ButtonComponent, MessageAction, SeparatorComponent
)

from webuiapi import WebUIApi
from deep_translator import GoogleTranslator 

# 匯入 Google Drive 相關模組
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError # 確保引入 HttpError

app = Flask(__name__)

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === LINE Bot 設定 ===
LINE_CHANNEL_ACCESS_TOKEN = 'EkPW2awSR6DF4IbO044ZIlmjFeIqmMLp3vC1kAgmpD5+FssLn5nNnfly37YxeIJpI8klVgipLcdZ8Ey00LLkJTMFUg7GQ3fyJ2uvruGL4SJ27axxJVtdOL0I2YvLoZR6Y1AVvsDslutuUDDde46n7AdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '3bb0908c9e23799bbaba57be1e638e35'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === 資料夾與腳本路徑設定 ===
UPLOAD_FOLDER = '/home/user/lora-train/lora-scripts/linebot_images/20_chien' # 圖片訓練暫存路徑
TAGGER_PYTHON = '/home/user/lora-train/tagger/venv/bin/python3' # 標註腳本的Python解釋器路徑
TAGGING_SCRIPT = '/home/user/lora-train/tagger/phi3_fast.py' # 標註腳本路徑
TRAIN_SCRIPT = '/home/user/lora-train/lora-scripts/train-t1-Copy2.sh' # 訓練腳本路徑
TRAIN_SCRIPT_DIR = '/home/user/lora-train/lora-scripts' # 訓練腳本所在目錄

# === 生圖腳本設定 ===
GENERATE_SCRIPT = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/chien-generate_random_seeds.py'
GENERATED_IMAGES_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/random_output"

# === Stable Diffusion API 設定 ===
SD_API = WebUIApi(host="127.0.0.1", port=7860) # 請確保您的 Stable Diffusion WebUI 已在 7860 埠運行
MODEL_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Stable-diffusion" # 您的 Stable Diffusion 模型資料夾

# --- 新增：模型風格映射 ---
MODEL_STYLES = {
    "anything-v5.safetensors": "動漫風格",
    "chilloutmix_NiPrunedFp32Fix.safetensors": "寫實風格",
    "Deliberate_v6.safetensors":"插畫風格",
    "Classic_disney_Style__Illustrious.safetensors":"迪士尼風格",
    "v1-5-pruned-emaonly.safetensors": "一般風格",
}

# === Google Drive 設定 ===
GOOGLE_DRIVE_GENERATED_IMAGES_FOLDER_ID = "1_9QLHupRK8e-CjfLh70mNdesVIuvemZ1" # 訓練後自動生圖上傳的資料夾 ID
GOOGLE_DRIVE_SD_OUTPUT_FOLDER_ID = "1_9QLHupRK8e-CjfLh70mNdesVIuvemZ1" # 手動生圖上傳的資料夾 ID
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "/home/user/lora-train/elite-mix-441517-k7-618329de7f11.json" # 服務帳戶金鑰路徑

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_IMAGES_OUTPUT_FOLDER, exist_ok=True)
os.makedirs("output", exist_ok=True) # 為 webuiapi 生成的圖片準備


# 全域狀態變數
uploading = False # 用於控制圖片訓練模式下的圖片上傳
user_id_for_process = None # 用於記錄觸發訓練流程的用戶 ID
image_count_lock = threading.Lock()
user_states = {} # 用於儲存每個用戶的狀態


# === Google Drive 認證初始化函數 ===
def authenticate_google_drive():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        logging.info("Google Drive 認證成功！")
        return service
    except Exception as e:
        logging.error(f"Google Drive 認證失敗: {e}")
        return None

# 初始化 Google Drive 服務 (應用程式啟動時認證一次)
drive_service = authenticate_google_drive()
if not drive_service:
    logging.critical("🚨 Google Drive 服務初始化失敗，程式將無法上傳檔案。請檢查服務帳戶金鑰。")
    # 如果 Google Drive 是關鍵功能，這裡可以選擇 abort(500) 或其他處理方式

# --- 上傳圖片到 Google Drive 並獲取公開連結函數 ---
def upload_to_drive_and_get_public_link(service, file_path, folder_id):
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')

        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(fileId=file_id, body=permission, fields='id').execute()

        file_info = service.files().get(fileId=file_id, fields='webContentLink,webViewLink').execute()
        public_link = file_info.get('webContentLink') # 直接下載連結
        if not public_link:
            public_link = file_info.get('webViewLink') # 瀏覽連結
            # 對於 Line bot，有時需要將 "view" 連結轉換為 "uc?export=download" 才能直接顯示
            if public_link and "view" in public_link:
                public_link = public_link.replace("/view", "/uc?export=download")

        if public_link:
            logging.info(f"Uploaded {os.path.basename(file_path)} to Google Drive. Public link: {public_link}")
            return public_link
        else:
            logging.error(f"Failed to get public link for file {file_path}.")
            return None
    except HttpError as error:
        logging.error(f"An error occurred during Google Drive upload: {error}", exc_info=True)
        return None

# === 工具函式 ===
def list_models():
    """列出 Stable Diffusion 模型資料夾中的模型名稱 (只列出 .safetensors 檔案)"""
    try:
        models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".safetensors")]
        # 篩選掉一些非模型檔案或特殊檔案
        models = [m for m in models if "Put Stable Diffusion checkpoints here.txt" not in m]
        return models
    except Exception as e:
        logging.error(f"無法列出 Stable Diffusion 模型: {e}")
        return []

def send_main_menu(user_id):
    """傳送主功能選單給使用者 (Flex Message 版本，只保留生圖和圖片訓練)"""
    flex_message = {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "text",
                    "text": "請選擇功能",
                    "weight": "bold",
                    "size": "xl",
                    "align": "center"
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#FF69B4", # 粉紅色
                    "action": {"type": "message", "label": "生圖功能", "text": "生圖功能"}
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#4A90E2", # 藍色
                    "action": {"type": "message", "label": "圖片訓練功能", "text": "圖片訓練功能"}
                },
            ]
        }
    }
    line_bot_api.push_message(user_id, FlexSendMessage(alt_text="請選擇功能", contents=flex_message))
    logging.info(f"User {user_id} received main menu.")

# === Webhook 接收 ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.error("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    except Exception as e:
        logging.error(f"Line Bot Error: {e}", exc_info=True)
        abort(400)
    return "OK"

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    global uploading, user_id_for_process # 確保能修改全域變數
    user_id = event.source.user_id
    message_text = event.message.text.strip()

    # 初始化用戶狀態，如果不存在
    if user_id not in user_states:
        user_states[user_id] = {
            'mode': 'main_menu', # 預設在主選單模式
            'image_count': 0,
            'prompt': None,
            'lang': 'zh', # 預設中文
            'processed_message_ids': set(),
            'selected_sd_model': None # 新增：記錄用戶選擇的 SD 主模型
        }
        
        
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=initial_welcome_message_zh)) # 預設中文，所以直接發中文提示
        logging.info(f"Sent initial welcome message to user {user_id}.")
        return # 避免在發送歡迎語後立即處理其他指令
    current_state = user_states[user_id]
    current_mode = current_state['mode']
    current_lang = current_state['lang']

    logging.info(f"User {user_id} in mode '{current_mode}' sent text: '{message_text}'")

    # --- 主選單指令 ---
    if message_text == "menu" or message_text == "選單":
        send_main_menu(user_id)
        current_state['mode'] = 'main_menu'
        current_state['prompt'] = None # 清除之前的生圖提示詞
        current_state['selected_sd_model'] = None # 清除已選擇的模型
        current_state['image_count'] = 0 # 清除訓練計數
        uploading = False # 確保退出上傳模式
        return

    # --- 語言切換指令 ---
    if message_text.lower() in ["中文", "chinese"]:
        current_state['lang'] = "zh"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("語言已切換為中文！"))
        return
    elif message_text.lower() in ["英文", "english"]:
        current_state['lang'] = "en"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("Language switched to English!"))
        return

    # --- 進入「圖片訓練功能」模式 ---
    if message_text == "圖片訓練功能":
        if not uploading: # 只有當沒有其他訓練在進行時才啟動
            # 清理舊資料夾 (在啟用模式時清除，而不是每次處理圖片時)
            if os.path.exists(UPLOAD_FOLDER):
                try:
                    shutil.rmtree(UPLOAD_FOLDER)
                    logging.info(f"Cleared old UPLOAD_FOLDER for user {user_id}.")
                except OSError as e:
                    logging.error(f"Error clearing UPLOAD_FOLDER: {e}")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(f"無法清理舊的圖片訓練資料夾，請稍後再試。錯誤: {e}" if current_lang == "zh" else f"Could not clear old image training folder. Error: {e}"))
                    return

            os.makedirs(UPLOAD_FOLDER)

            current_state['mode'] = 'image_training'
            current_state['image_count'] = 0
            uploading = True # 設定全域上傳標誌
            user_id_for_process = user_id # 記錄觸發用戶
            reply_text = "請開始傳送圖片！（共 20 張）" if current_lang == "zh" else "Please start sending images! (20 images in total)"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            logging.info(f"User {user_id} entered image training mode.")
        else:
            reply_text = "目前已有圖片訓練流程正在進行中，請稍後再試。" if current_lang == "zh" else "An image training process is currently in progress, please try again later."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    # --- 進入「生圖功能」模式 ---
    if message_text == "生圖功能":
        current_state['mode'] = 'image_generation'
        current_state['prompt'] = None # 清除舊的提示詞
        current_state['selected_sd_model'] = None # 清除已選擇的模型
        reply_text = "請輸入您的提示詞 (Positive Prompt)！" if current_lang == "zh" else "Please enter your prompt (Positive Prompt)!"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        logging.info(f"User {user_id} entered image generation mode.")
        return

    # --- 處理「圖片訓練功能」模式下的文字訊息 ---
    if current_mode == 'image_training':
        reply_text = f"您目前在圖片訓練模式。請繼續上傳圖片，目前已上傳 {current_state['image_count']} 張。" if current_lang == "zh" else f"You are in image training mode. Please continue uploading images. Current: {current_state['image_count']} images."
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    # --- 處理「生圖功能」模式下的文字訊息 (輸入提示詞) ---
    if current_mode == 'image_generation':
        if current_state['prompt'] is None: # 如果還沒有提示詞，則儲存並要求選擇模型
            current_state['prompt'] = message_text
            
            model_files = list_models() # 獲取所有 Stable Diffusion 模型檔案名稱
            
            # 準備模型選擇按鈕列表
            model_buttons = []
            for model_file in model_files:
                # 使用 MODEL_STYLES 獲取風格名稱，如果沒有則用檔案名稱作為備用
                style_name = MODEL_STYLES.get(model_file, model_file.replace(".safetensors", ""))
                
                model_buttons.append(
                    ButtonComponent(
                        style='primary',
                        color='#66CCFF', # 藍色按鈕
                        action=PostbackAction(label=style_name, data=f"action=select_sd_model&model_name={model_file}") # 實際傳遞檔案名稱
                    )
                )
            
            if not model_buttons:
                reply_text = "目前沒有可用的 Stable Diffusion 模型檔案。請聯繫管理員。" if current_lang == "zh" else "No Stable Diffusion model files found. Please contact the administrator."
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

            # 使用 Flex Message 展示模型選擇按鈕
            flex_message_content = BubbleContainer(
                direction='ltr',
                header=BoxComponent(
                    layout='vertical',
                    contents=[
                        TextComponent(text="請選擇模型風格", weight='bold', size='xl', align='center')
                    ]
                ),
                body=BoxComponent(
                    layout='vertical',
                    spacing='md',
                    contents=model_buttons
                )
            )
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="選擇模型風格", contents=flex_message_content))
            logging.info(f"User {user_id} entered prompt: '{message_text}', awaiting model selection.")
        else: # 如果已經有提示詞，但又輸入文字，則提示選擇模型
            reply_text = "請先選擇模型風格。" if current_lang == "zh" else "Please choose a model style first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    # --- 預設回覆：不在任何特定模式下，提示主選單 ---
    reply_text = "請輸入 'menu' 查看功能選單。" if current_lang == "zh" else "Please type 'menu' to see the function menu."
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# === 處理圖片訊息 (圖片訓練模式專用) ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    global uploading, user_id_for_process
    user_id = event.source.user_id

    # 確保用戶處於圖片訓練模式且允許上傳
    if user_id not in user_states or user_states[user_id]['mode'] != 'image_training' or not uploading:
        logging.info(f"Received image from {user_id} but not in image training mode or uploading state. Ignoring.")
        return

    current_state = user_states[user_id]
    current_lang = current_state['lang']

    # 檢查是否已處理過此訊息ID，避免重複處理
    if event.message.id in current_state['processed_message_ids']:
        logging.info(f"Duplicate message ID {event.message.id} for user {user_id}. Ignoring.")
        return
    current_state['processed_message_ids'].add(event.message.id)

    # === 核心修改：使用鎖來保護計數器 ===
    with image_count_lock:
        current_count_in_state = current_state['image_count']

        if current_count_in_state >= 20:
            logging.info(f"User {user_id} tried to upload more than 20 images. Current (state): {current_count_in_state}. Ignoring.")
            # 不再回覆，因為可能正在訓練或已完成
            return

        new_index = current_count_in_state + 1
        filename = f"{new_index:02d}.png"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            image_content = line_bot_api.get_message_content(event.message.id)
            with open(image_path, 'wb') as f:
                for chunk in image_content.iter_content():
                    f.write(chunk)

            logging.info(f"User {user_id} saved image {filename}. Total (state): {new_index}")
            current_state['image_count'] = new_index

            if new_index == 20:
                logging.info(f"User {user_id} received 20 images, preparing to start tagging, training, and generation.")
                uploading = False # 停止上傳模式
                current_state['mode'] = 'main_menu' # 完成後回到主選單模式
                current_state['processed_message_ids'].clear() # 清空已處理的訊息ID
                
                # 確保只有觸發訓練的用戶才能啟動完整流程
                if user_id_for_process == user_id:
                    threading.Thread(target=run_full_pipeline, args=(user_id,)).start()
                    reply_text = "已收到所有 20 張圖片。開始標註與訓練，這需要一些時間，完成後會通知您。" if current_lang == "zh" else "All 20 images received. Starting tagging and training, this will take some time. You will be notified when complete."
                    line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
                else:
                    logging.warning(f"Unexpected user_id_for_process mismatch. Expected {user_id}, got {user_id_for_process}")
                    line_bot_api.push_message(user_id, TextSendMessage(text="系統內部錯誤：無法啟動訓練流程。" if current_lang == "zh" else "Internal system error: Unable to start training process."))
            # --- 這裡不再有 else 區塊來回覆每張圖片進度 ---
            # 因為 LINE 的 push_message 有限制，頻繁發送可能導致問題

        except Exception as e:
            logging.error(f"Error saving image for user {user_id}: {e}", exc_info=True)
            reply_text = f"圖片儲存失敗：{str(e)}" if current_lang == "zh" else f"Failed to save image: {str(e)}"
            line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))

# === 處理 Postback 事件 ===
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data
    logging.info(f"User {user_id} postback data: {data}")

    if user_id not in user_states:
        user_states[user_id] = {'mode': 'main_menu', 'lang': 'zh', 'prompt': None, 'selected_sd_model': None} # 確保狀態存在
    current_state = user_states[user_id]
    current_lang = current_state['lang']

    # --- 處理 Stable Diffusion 模型選擇邏輯 ---
    if data.startswith("action=select_sd_model&model_name="):
        model_name = data.split("model_name=", 1)[1] # 取得模型檔案名稱
        prompt = current_state.get('prompt') # 從狀態中獲取提示詞

        if not prompt:
            reply_text = "請先輸入提示詞。" if current_lang == "zh" else "Please enter prompt first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

        # 儲存用戶選擇的模型
        current_state['selected_sd_model'] = model_name

        reply_text = "正在生成圖片中，請稍候..." if current_lang == "zh" else "Generating image, please wait..."
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        logging.info(f"User {user_id} selected SD model: {model_name}, prompt: '{prompt}'")

        # 在新的線程中執行生圖
        threading.Thread(target=run_sd_generation, args=(user_id, prompt, model_name, event.reply_token)).start()
        current_state['prompt'] = None # 清除提示詞，準備下一次生圖
        current_state['selected_sd_model'] = None # 清除已選擇的模型
        current_state['mode'] = 'main_menu' # 生圖完成或失敗後回到主選單模式
        return

    logging.warning(f"Unhandled postback event from user {user_id}: {data}")

# --- 圖片訓練完整流程 ---
def run_full_pipeline(user_id):
    global uploading
    drive_service_pipeline = authenticate_google_drive() # 再次認證確保線程獨立性，或使用傳入的drive_service
    if not drive_service_pipeline:
        line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，請檢查服務帳戶金鑰。" if user_states.get(user_id, {}).get('lang', 'zh') == "zh" else "❌ Could not connect to Google Drive service. Please check service account key."))
        return

    current_lang = user_states.get(user_id, {}).get('lang', 'zh') # 獲取用戶語言

    try:
        line_bot_api.push_message(user_id, TextSendMessage(text="圖片上傳完成，開始標註啦!" if current_lang == "zh" else "Image upload complete, starting tagging!"))
        logging.info(f"User {user_id} started tagging process.")

        # === Step 1: 執行標註腳本 ===
        print(f"[{user_id}] 🚀 開始執行標註腳本...")
        tag_result = subprocess.run(
            [
                TAGGER_PYTHON,
                TAGGING_SCRIPT,
                UPLOAD_FOLDER,
                "--character_name", "chien", # 這些參數可能需要根據你的phi3_fast.py進行調整
                "--trigger_word", "masterpiece" # 這些參數可能需要根據你的phi3_fast.py進行調整
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR
        )
        logging.info(f"Tagger stdout for user {user_id}:\n{tag_result.stdout}")
        logging.error(f"Tagger stderr for user {user_id}:\n{tag_result.stderr}") # 即使是空也要記錄
        line_bot_api.push_message(user_id, TextSendMessage(text="標註完成，執行模型訓練，需要大概10-15分鐘。" if current_lang == "zh" else "Tagging complete, starting model training, this will take about 10-15 minutes."))

        # === Step 2: 標註成功後執行訓練 ===
        print(f"[{user_id}] 🚀 開始執行模型訓練腳本...")
        train_result = subprocess.run(
            ['bash', TRAIN_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR,
            timeout=3000 # 增加訓練超時時間，例如 50 分鐘 (3000 秒)
        )
        logging.info(f"Train stdout for user {user_id}:\n{train_result.stdout}")
        logging.error(f"Train stderr for user {user_id}:\n{train_result.stderr}")
        line_bot_api.push_message(user_id, TextSendMessage(text="模型訓練已成功完成！🎉 開始生成預覽圖片..." if current_lang == "zh" else "Model training completed successfully! 🎉 Starting to generate preview images..."))

        # === Step 3: 訓練成功後，執行生圖腳本 chien-generate_random_seeds.py ===
        print(f"[{user_id}] 🚀 開始執行生圖腳本")
        # 清空舊的生成圖片資料夾，確保只處理本次生成的圖片
        if os.path.exists(GENERATED_IMAGES_OUTPUT_FOLDER):
            shutil.rmtree(GENERATED_IMAGES_OUTPUT_FOLDER)
        os.makedirs(GENERATED_IMAGES_OUTPUT_FOLDER)

        generate_result = subprocess.run(
            ['python3', GENERATE_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300 # 生圖時間，例如 5 分鐘
        )
        logging.info(f"Generate stdout for user {user_id}:\n{generate_result.stdout}")
        logging.error(f"Generate stderr for user {user_id}:\n{generate_result.stderr}")
        line_bot_api.push_message(user_id, TextSendMessage(text="預覽圖片生成完成" if current_lang == "zh" else "Preview image generation complete"))

        # === Step 4: 將生成的預覽圖片上傳到 Google Drive 並推播給使用者 ===
        print(f"[{user_id}] 🚀 開始上傳預覽圖片到 Google Drive...")
        image_urls = []
        for filename in os.listdir(GENERATED_IMAGES_OUTPUT_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(GENERATED_IMAGES_OUTPUT_FOLDER, filename)
                public_link = upload_to_drive_and_get_public_link(
                    drive_service_pipeline, img_path, GOOGLE_DRIVE_GENERATED_IMAGES_FOLDER_ID
                )
                if public_link:
                    image_urls.append(public_link)

        if len(image_urls) == 0:
            line_bot_api.push_message(user_id, TextSendMessage(text="⚠️ 未能上傳任何預覽圖片到 Google Drive，請檢查設定或生圖結果。" if current_lang == "zh" else "⚠️ No preview images uploaded to Google Drive. Check settings or generation results."))
        else:
            for url in image_urls:
                try:
                    line_bot_api.push_message(user_id, ImageSendMessage(
                        original_content_url=url,
                        preview_image_url=url
                    ))
                except Exception as send_e:
                    logging.error(f"❌ 無法傳送圖片訊息 (LINE) for user {user_id}: {send_e}", exc_info=True)
            line_bot_api.push_message(user_id, TextSendMessage(text="以下是使用您訓練的模型生成的預覽圖片。" if current_lang == "zh" else "Here are preview images generated using your trained model."))

        # === 移除 Step 5: 關於 LoRA 模型保留的邏輯 ===
        # 因為不再支援用戶保留 LoRA 模型並管理

    except subprocess.CalledProcessError as e:
        error_message = f"程式執行失敗: {e.cmd}\n錯誤輸出:\n{e.stderr}" if current_lang == "zh" else f"Program failed: {e.cmd}\nError output:\n{e.stderr}"
        logging.error(f"子程序錯誤 for user {user_id}: {error_message}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 處理過程中發生錯誤，請聯絡管理員。\n錯誤詳情：{error_message}" if current_lang == "zh" else f"❌ An error occurred during processing. Please contact the administrator.\nError details: {error_message}"))
    except FileNotFoundError as e:
        error_message = f"找不到檔案或程式: {e}" if current_lang == "zh" else f"File or program not found: {e}"
        logging.error(f"檔案未找到錯誤 for user {user_id}: {error_message}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 必要的檔案或程式（如 Python 解釋器、腳本）未找到。請聯絡管理員。\n錯誤詳情：{error_message}" if current_lang == "zh" else f"❌ Required file or program (e.g., Python interpreter, script) not found. Please contact the administrator.\nError details: {error_message}"))
    except subprocess.TimeoutExpired as e:
        error_message = f"處理超時: {e.cmd}" if current_lang == "zh" else f"Processing timed out: {e.cmd}"
        logging.error(f"超時錯誤 for user {user_id}: {error_message}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 處理超時。這可能表示訓練或生圖過程太長。請聯絡管理員。\n錯誤詳情：{error_message}" if current_lang == "zh" else f"❌ Processing timed out. This may indicate the training or generation process is too long. Please contact the administrator.\nError details: {error_message}"))
    except Exception as e:
        logging.error(f"處理過程中發生未知錯誤 for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"❌ 發生未知錯誤。請聯絡管理員。\n錯誤詳情：{e}" if current_lang == "zh" else f"❌ An unknown error occurred. Please contact the administrator.\nError details: {e}"))
    finally:
        # 無論成功失敗都會清理狀態和上傳資料夾
        user_states[user_id]['mode'] = 'main_menu'
        user_states[user_id]['image_count'] = 0
        user_id_for_process = None # 清除觸發用戶ID
        uploading = False # 重置上傳狀態

        if os.path.exists(UPLOAD_FOLDER):
            try:
                shutil.rmtree(UPLOAD_FOLDER)
                logging.info(f"Cleaned up UPLOAD_FOLDER for user {user_id}.")
            except OSError as e:
                logging.error(f"Error cleaning up UPLOAD_FOLDER after pipeline: {e}")


# --- 獨立的生圖流程 (使用 webuiapi) ---
def run_sd_generation(user_id, prompt, model_name, reply_token):
    drive_service_sd = authenticate_google_drive() # 再次認證確保線程獨立性
    if not drive_service_sd:
        line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，請檢查服務帳戶金鑰。" if user_states.get(user_id, {}).get('lang', 'zh') == "zh" else "❌ Could not connect to Google Drive service. Please check service account key."))
        return

    current_lang = user_states.get(user_id, {}).get('lang', 'zh') # 獲取用戶語言

    try:
        # 設定模型
        logging.info(f"Attempting to set model for SD API: {model_name}")
        SD_API.util_set_model(model_name) # 設定 Stable Diffusion WebUI 使用的模型
        logging.info(f"SD API model set to: {model_name}")

        # 翻譯提示詞 (如果需要)
        translated_prompt = prompt
        if current_lang == "zh":
            try:
                # 假設用戶輸入中文，翻譯成英文給 SD
                translated_prompt = GoogleTranslator(source='zh-TW', target='en').translate(prompt)
                logging.info(f"Translated prompt from '{prompt}' to '{translated_prompt}'")
            except Exception as e:
                logging.warning(f"Prompt translation failed: {e}. Using original prompt.")
                translated_prompt = prompt # 翻譯失敗時使用原始提示詞

        # 生成圖片
        logging.info(f"Calling SD API for generation with prompt: '{translated_prompt}'")
        
        # 確保 'output' 資料夾存在，用於存放 webuiapi 生成的圖片
        temp_output_dir = "output"
        os.makedirs(temp_output_dir, exist_ok=True)

        result = SD_API.txt2img(
            prompt=translated_prompt,
            negative_prompt="bad anatomy, low quality, deformed, worst quality, missing fingers, extra fingers, blurry",
            seed=-1, # 隨機種子
            steps=30,
            cfg_scale=7,
            width=512,
            height=768,
            save_images=True, # 讓 webuiapi 保存圖片
            override_settings={"sd_model_checkpoint": model_name} # 再次確保使用指定模型
        )
        
        if result and result.images:
            generated_image_path = os.path.join(temp_output_dir, "generated_image.png")
            # 確保圖片被保存下來
            result.images[0].save(generated_image_path)
            logging.info(f"Generated image saved to: {generated_image_path}")

            # 上傳到 Google Drive
            public_url = upload_to_drive_and_get_public_link(
                drive_service_sd, generated_image_path, GOOGLE_DRIVE_SD_OUTPUT_FOLDER_ID
            )

            if public_url:
                try:
                    line_bot_api.push_message(user_id, ImageSendMessage(
                        original_content_url=public_url,
                        preview_image_url=public_url
                    ))
                    line_bot_api.push_message(user_id, TextSendMessage(text="您的圖片已生成！" if current_lang == "zh" else "Your image has been generated!"))
                except Exception as send_e:
                    logging.error(f"❌ 無法傳送圖片訊息 (LINE) for user {user_id}: {send_e}", exc_info=True)
                    line_bot_api.push_message(user_id, TextSendMessage(text="圖片生成成功，但無法發送給您。請聯繫管理員。" if current_lang == "zh" else "Image generated successfully, but could not be sent to you. Please contact the administrator."))
            else:
                line_bot_api.push_message(user_id, TextSendMessage(text="圖片生成成功，但上傳至 Google Drive 失敗。請聯繫管理員。" if current_lang == "zh" else "Image generated successfully, but failed to upload to Google Drive. Please contact the administrator."))
        else:
            line_bot_api.push_message(user_id, TextSendMessage(text="圖片生成失敗，請檢查提示詞或設定。" if current_lang == "zh" else "Image generation failed. Please check your prompt or settings."))
            logging.error(f"SD API did not return images for user {user_id} with prompt: '{prompt}'")

    except Exception as e:
        logging.error(f"SD Generation error for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"圖片生成時發生錯誤：{str(e)}。請稍後再試。" if current_lang == "zh" else f"An error occurred during image generation: {str(e)}. Please try again later."))
    finally:
        # 清理生成的圖片
        if os.path.exists(temp_output_dir) and os.path.isdir(temp_output_dir):
            try:
                shutil.rmtree(temp_output_dir)
                logging.info(f"Cleaned up temporary output directory: {temp_output_dir}")
            except OSError as e:
                logging.error(f"Error cleaning up temporary output directory: {e}")


# --- 運行 Flask 應用 ---
if __name__ == "__main__":
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(GENERATED_IMAGES_OUTPUT_FOLDER):
        os.makedirs(GENERATED_IMAGES_OUTPUT_FOLDER)
    if not os.path.exists("output"):
        os.makedirs("output")
        
    print(f"已創建模型目錄：{MODEL_FOLDER}，請將 .safetensors 模型文件放入此目錄。")
    
    # 為了方便測試，您可以創建一些假的模型文件，如果它們不存在的話
    for model_name in MODEL_STYLES.keys():
        dummy_model_path = os.path.join(MODEL_FOLDER, model_name)
        if not os.path.exists(dummy_model_path):
            with open(dummy_model_path, 'w') as f:
                f.write(f"This is a dummy file for {model_name}")
            print(f"已創建假模型文件：{dummy_model_path}")

    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)