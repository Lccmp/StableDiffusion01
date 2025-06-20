import os
import shutil
import subprocess
import threading
import torch
import io
import json
import logging
import traceback
import re
import subprocess

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage,
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent, FlexSendMessage,
    BubbleContainer, BoxComponent, TextComponent, ButtonComponent
)

from webuiapi import WebUIApi
from deep_translator import GoogleTranslator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload # Import MediaIoBaseDownload for efficient downloading

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === LINE Bot 設定 ===
LINE_CHANNEL_ACCESS_TOKEN = 'lEjwrG6VNT4HFBelZy8rwFKKV9dOZrn1On5+ZO1TlF4XBS27ixlEJoDISs/lsTK7ttsZ4J/VDr9GyO8yEzufkKnTmMQ+WN5ILcY2C82hPnOxRYn5D5cfV/EJzWpJ8k87CpMniI2Kbq6HLPQvFhkZbgdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '391269aaead5a69836c02fba3fedf3d2'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === 資料夾與腳本路徑設定 ===
UPLOAD_FOLDER = '/home/user/lora-train/lora-scripts/linebot_images/20_chien'
TAGGER_PYTHON = '/home/user/lora-train/tagger/venv/bin/python3'
TAGGING_SCRIPT = '/home/user/lora-train/tagger/phi3_fast.py'
TRAIN_SCRIPT = '/home/user/lora-train/lora-scripts/train-t1-Copy1.sh'
TRAIN_SCRIPT_DIR = '/home/user/lora-train/lora-scripts'

# 你的生圖腳本路徑和輸出資料夾
GENERATE_SCRIPT = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/chien-generate_random_seeds.py'
GENERATED_IMAGES_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/random_output" # 確保你的腳本會將圖片輸出到這裡

# === 生圖腳本設定 ===
# If you have a separate Python script for generation with custom seeds, use it.
# Otherwise, we'll use webuiapi's txt2img directly.
# GENERATE_SCRIPT = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/chien-generate_random_seeds.py'

# === 統一的輸出資料夾，用於所有生成的圖片 ===
CONSOLIDATED_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/generated_images_output"
# 訓練好的 LoRA 模型會暫時放在這裡，直到上傳到 Google Drive
LORA_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Lora"


# === Stable Diffusion API 設定 ===
SD_API = WebUIApi(host="127.0.0.1", port=7860)
MODEL_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Stable-diffusion"
SD_LORA_DOWNLOAD_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Lora"

# === Google Drive 設定 ===
GOOGLE_DRIVE_OUTPUT_FOLDER_ID = "1_9QLHupRK8e-CjfLh70mNdesVIuveZ1"
GOOGLE_DRIVE_LORA_MODELS_FOLDER_ID = "1CXA9TXfEWXHbm0og7Zws1jqkFq2F67B"

SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "/home/user/lora-train/elite-mix-441517-k7-618329de7f11.json"

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONSOLIDATED_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SD_LORA_DOWNLOAD_FOLDER, exist_ok=True)

# 全域狀態變數
uploading = False # 全局变量

def my_function():
    global uploading # 首先声明为全局变量
    print(uploading) # 现在 Python 知道它是全局变量了
    uploading = True
user_id_for_process = None

# 用戶狀態管理
user_states = {}

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

# 初始化 Google Drive 服務
drive_service = authenticate_google_drive()
if not drive_service:
    logging.critical("🚨 Google Drive 服務初始化失敗，程式將無法上傳檔案。請檢查服務帳戶金鑰。")

# --- 上傳圖片到 Google Drive 並獲取公開連結函數 ---
def upload_to_drive_and_get_public_link(drive_service_instance, file_path, folder_id):
    """
    上傳單一檔案到 Google Drive，並設定公開權限，回傳公開連結(URL)。
    適用於任何文件類型，包括圖片和 LoRA 模型。
    """
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service_instance.files().create(body=file_metadata, media_body=media, fields='id, webContentLink, webViewLink').execute()

        # 設定公開權限
        drive_service_instance.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'},
        ).execute()

        # For files like images that can be viewed directly in the browser, prefer webViewLink
        # For downloadable files, webContentLink usually provides a direct download or preview
        # If webContentLink is not suitable for direct image display in LINE, we might need a custom approach or force download link
        if 'webViewLink' in file:
            public_link = file['webViewLink']
        elif 'webContentLink' in file:
            public_link = file['webContentLink']
        else:
            # Fallback for direct download link
            public_link = f"https://drive.google.com/uc?id={file['id']}&export=download"
        
        logging.info(f"Uploaded {os.path.basename(file_path)} to Google Drive. Public link: {public_link}")
        return public_link
    except Exception as e:
        logging.error(f"上傳 Google Drive 失敗: {e}")
        traceback.print_exc()
        return None

# --- 從 Google Drive 下載檔案到本地 ---
def download_file_from_drive(drive_service_instance, file_id, destination_path):
    """
    從 Google Drive 下載檔案。
    """
    try:
        request = drive_service_instance.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f"Download progress: {int(status.progress() * 100)}%")
        logging.info(f"Downloaded file {file_id} to {destination_path}")
        return True
    except HttpError as e:
        logging.error(f"從 Google Drive 下載檔案 {file_id} 時發生 HTTP 錯誤: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"從 Google Drive 下載檔案 {file_id} 時發生錯誤: {e}", exc_info=True)
        return False


# === 工具函式 ===
def list_models():
    """列出 Stable Diffusion 模型資料夾中的模型名稱"""
    try:
        models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".safetensors")]
        models = [m for m in models if "Put Stable Diffusion checkpoints here.txt" not in m]
        return models
    except Exception as e:
        logging.error(f"無法列出 Stable Diffusion 模型: {e}")
        return []

# --- 從 Google Drive 列出 LoRA 模型 ---
def list_lora_models_from_drive(drive_service_instance, folder_id):
    """
    從 Google Drive 的指定資料夾中列出所有 LoRA 模型。
    返回一個列表，每個元素為 {'name': 檔案名稱 (不含副檔名), 'file_id': 檔案ID, 'original_name': 檔案名稱 (不含副檔名)}
    """
    lora_models = []
    try:
        query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service_instance.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])

        for item in items:
            file_name = item['name']
            if file_name.endswith('.safetensors') or file_name.endswith('.pt'):
                lora_tag = file_name.replace('.safetensors', '').replace('.pt', '')
                lora_models.append({
                    'name': lora_tag,
                    'file_id': item['id'],
                    'original_name': lora_tag
                })
        logging.info(f"從 Google Drive 列出 {len(lora_models)} 個 LoRA 模型。")
    except HttpError as e:
        logging.error(f"從 Google Drive 列出 LoRA 模型時發生 HTTP 錯誤: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"從 Google Drive 列出 LoRA 模型時發生未知錯誤: {e}", exc_info=True)
    return lora_models


def send_main_menu(user_id):
    """傳送主功能選單給使用者 (Flex Message 版本)"""
    flex_message = BubbleContainer(
        size="mega",
        body=BoxComponent(
            layout="vertical",
            spacing="md",
            contents=[
                TextComponent(
                    text="請選擇功能",
                    weight="bold",
                    size="xl",
                    align="center"
                ),
                ButtonComponent(
                    style="primary",
                    color="#FAB1A0",
                    action={"type": "message", "label": "生圖功能", "text": "生圖功能"}
                ),
                ButtonComponent(
                    style="primary",
                    color="#4A90E2",
                    action={"type": "message", "label": "圖片訓練功能", "text": "圖片訓練功能"}
                ),
                ButtonComponent(
                    style="primary",
                    color="#6C5CE7",
                    action={"type": "postback", "label": "我的 LoRA 模型", "data": "action=my_models"}
                )
            ]
        )
    )
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
        logging.error(f"Line Bot Error: {e}")
        abort(400)
    return "OK"

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    global uploading, user_id_for_process
    user_id = event.source.user_id
    message_text = event.message.text.strip()

    if user_id not in user_states:
        user_states[user_id] = {
            'mode': 'main_menu',
            'image_count': 0,
            'prompt': None,
            'lang': 'zh',
            'selected_lora_info': None,
            'latest_trained_lora_path': None # Store the path to the newly trained LoRA
        }
    current_state = user_states[user_id]
    current_mode = current_state['mode']
    current_lang = current_state['lang']

    logging.info(f"User {user_id} in mode '{current_mode}' sent text: '{message_text}'")

    if message_text == "menu" or message_text == "選單":
        send_main_menu(user_id)
        current_state['mode'] = 'main_menu'
        current_state['prompt'] = None
        current_state['selected_lora_info'] = None
        current_state['latest_trained_lora_path'] = None # Clear any pending trained LoRA
        current_state['image_count'] = 0
        uploading = False
        return

    if message_text.lower() in ["中文", "chinese"]:
        current_state['lang'] = "zh"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("語言已切換為中文！"))
        return
    elif message_text.lower() in ["英文", "english"]:
        current_state['lang'] = "en"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("Language switched to English!"))
        return

    if message_text == "圖片訓練功能":
        if not uploading:
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
                logging.info(f"Cleared old UPLOAD_FOLDER contents for user {user_id} before new training.")
            os.makedirs(UPLOAD_FOLDER)

            current_state['mode'] = 'image_training'
            current_state['image_count'] = 0
            uploading = True
            user_id_for_process = user_id
            reply_text = "請開始傳送圖片！（共 20 張）" if current_lang == "zh" else "Please start sending images! (20 images in total)"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            logging.info(f"User {user_id} entered image training mode.")
        else:
            reply_text = "目前已有圖片訓練流程正在進行中，請稍後再試。" if current_lang == "zh" else "An image training process is currently in progress, please try again later."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    if message_text == "生圖功能":
        current_state['mode'] = 'image_generation'
        current_state['prompt'] = None
        current_state['selected_lora_info'] = None
        current_state['latest_trained_lora_path'] = None # Clear any pending trained LoRA
        reply_text = "請輸入您的提示詞 (Positive Prompt)！" if current_lang == "zh" else "Please enter your prompt (Positive Prompt)!"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        logging.info(f"User {user_id} entered image generation mode.")
        return

    # Handle text messages in specific states
    if current_mode == 'image_training':
        reply_text = f"您目前在圖片訓練模式。請繼續上傳圖片，目前已上傳 {current_state['image_count']} 張。" if current_lang == "zh" else f"You are in image training mode. Please continue uploading images. Current: {current_state['image_count']} images."
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    if current_mode == 'image_generation':
        if current_state['prompt'] is None:
            current_state['prompt'] = message_text
            # If there's a recently trained LoRA and the user hasn't selected one yet, offer it first
            if current_state['latest_trained_lora_path'] and not current_state['selected_lora_info']:
                lora_filename = os.path.basename(current_state['latest_trained_lora_path'])
                lora_tag = lora_filename.replace('.safetensors', '').replace('.pt', '')
                confirm_message = TextSendMessage(
                    text=f"偵測到您剛訓練完成 LoRA 模型：**{lora_tag}**。\n您想用這個模型來生圖預覽嗎？" if current_lang == "zh" else
                         f"Detected your newly trained LoRA model: **{lora_tag}**.\nWould you like to use this model for image preview?"
                )
                confirm_buttons = TemplateSendMessage(
                    alt_text="使用新模型預覽？",
                    template=ButtonsTemplate(
                        title="使用新模型？" if current_lang == "zh" else "Use New Model?",
                        text=lora_tag,
                        actions=[
                            PostbackAction(
                                label="是，用它預覽" if current_lang == "zh" else "Yes, preview with it",
                                data=f"action=use_new_lora_preview&lora_path={current_state['latest_trained_lora_path']}"
                            ),
                            PostbackAction(
                                label="否，選擇其他" if current_lang == "zh" else "No, select another",
                                data="action=select_base_model" # Go to base model selection
                            )
                        ]
                    )
                )
                line_bot_api.reply_message(event.reply_token, [confirm_message, confirm_buttons])
            else:
                # Proceed to base model selection if no new LoRA or already selected
                send_base_model_selection(user_id, event.reply_token, current_lang)
            return
        else:
            reply_text = "請先選擇模型。" if current_lang == "zh" else "Please choose a model first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    reply_text = "請輸入 'menu' 查看功能選單。" if current_lang == "zh" else "Please type 'menu' to see the function menu."
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

def send_base_model_selection(user_id, reply_token, current_lang):
    model_list = list_models()
    display_models = model_list[:4] if model_list else ["No models found.safetensors"]
    actions = [PostbackAction(label=model[:20], data=f"model={model}") for model in display_models]
    template = ButtonsTemplate(
        title="選擇基礎模型" if current_lang == "zh" else "Choose Base Model",
        text="請選擇模型來生成圖片" if current_lang == "zh" else "Select model to generate image",
        actions=actions
    )
    line_bot_api.reply_message(reply_token, TemplateSendMessage(alt_text="Select model", template=template))
    logging.info(f"User {user_id} awaiting base model selection.")

# === 處理圖片訊息 ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    global uploading, user_id_for_process
    user_id = event.source.user_id

    if user_id not in user_states or user_states[user_id]['mode'] != 'image_training' or not uploading:
        logging.info(f"Received image from {user_id} but not in image training mode or uploading state. Ignoring.")
        return

    current_state = user_states[user_id]
    current_lang = current_state['lang']

    current_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if current_count >= 20:
        logging.info(f"User {user_id} tried to upload more than 20 images. Current: {current_count}. Ignoring.")
        return

    new_index = current_count + 1
    filename = f"{new_index:02d}.png"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image_content = line_bot_api.get_message_content(event.message.id)
        with open(image_path, 'wb') as f:
            for chunk in image_content.iter_content():
                f.write(chunk)

        logging.info(f"User {user_id} saved image {filename}. Total: {new_index}")
        current_state['image_count'] = new_index

        if new_index == 20:
            logging.info(f"User {user_id} received 20 images, preparing to start tagging, training, and generation.")
            uploading = False
            current_state['mode'] = 'main_menu' # Reset mode after image upload completion

            threading.Thread(target=run_full_pipeline, args=(user_id,)).start()
            reply_text = "已收到所有 20 張圖片。開始標註與訓練，這需要一些時間，完成後會通知您。" if current_lang == "zh" else "All 20 images received. Starting tagging and training, this will take some time. You will be notified when complete."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    except Exception as e:
        logging.error(f"Error saving image for user {user_id}: {e}", exc_info=True)
        reply_text = f"圖片儲存失敗：{str(e)}" if current_lang == "zh" else f"Failed to save image: {str(e)}"
        line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))

# === 處理模型按鈕選擇 ===
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data
    logging.info(f"User {user_id} postback data: {data}")

    if user_id not in user_states:
        user_states[user_id] = {
            'mode': 'main_menu',
            'lang': 'zh',
            'prompt': None,
            'selected_lora_info': None,
            'latest_trained_lora_path': None
        }
    current_state = user_states[user_id]
    current_lang = current_state['lang']

    # --- 處理基礎模型選擇 ---
    if data.startswith("model="):
        model_name = data.split("=", 1)[1]
        prompt = current_state.get('prompt')

        if not prompt:
            reply_text = "請先輸入提示詞。" if current_lang == "zh" else "Please enter prompt first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            "正在生成圖片中，請稍候..." if current_lang == "zh" else "Generating image, please wait..."
        ))
        logging.info(f"User {user_id} selected base model: {model_name}, prompt: '{prompt}'")

        threading.Thread(target=run_sd_generation, args=(user_id, prompt, model_name, current_state.get('selected_lora_info'), event.reply_token)).start()
        
        # Clear states after starting generation
        current_state['prompt'] = None
        current_state['selected_lora_info'] = None
        current_state['latest_trained_lora_path'] = None # Ensure this is cleared
        current_state['mode'] = 'main_menu'
        return

    # --- 處理「我的 LoRA 模型」Postback 事件：從 Google Drive 列出 LoRA 模型 ---
    elif data == "action=my_models":
        lora_models_from_drive = list_lora_models_from_drive(drive_service, GOOGLE_DRIVE_LORA_MODELS_FOLDER_ID)
        
        if not lora_models_from_drive:
            reply_text = "Google Drive 上目前沒有找到 LoRA 模型喔！" if current_lang == "zh" else "No LoRA models found on Google Drive yet!"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

        buttons_contents = []
        for model_info in lora_models_from_drive:
            buttons_contents.append(
                ButtonComponent(
                    style="link",
                    height="sm",
                    action=PostbackAction(
                        label=model_info['name'][:20],
                        data=f"action=select_lora&lora_name={model_info['name']}&lora_file_id={model_info['file_id']}&original_name={model_info['original_name']}"
                    )
                )
            )
            if len(buttons_contents) >= 10:
                break
        
        if not buttons_contents:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("沒有可用的 LoRA 模型。" if current_lang == "zh" else "No LoRA models available."))
            return

        flex_message = BubbleContainer(
            size="giga",
            body=BoxComponent(
                layout="vertical",
                spacing="md",
                contents=[
                    TextComponent(
                        text="請選擇您要使用的 LoRA 模型" if current_lang == "zh" else "Please select the LoRA model you want to use",
                        weight="bold",
                        size="xl",
                        align="center"
                    )
                ] + buttons_contents
            )
        )
        line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="選擇我的模型", contents=flex_message))
        logging.info(f"User {user_id} requested my models list from Google Drive.")
        return

    # --- 處理 LoRA 模型選擇 Postback 事件 (從 Drive 或本地) ---
    elif data.startswith("action=select_lora"):
        parts = data.split('&')
        selected_lora_name = None
        selected_lora_file_id = None
        original_lora_name = None

        for part in parts:
            if part.startswith("lora_name="):
                selected_lora_name = part.split("=", 1)[1]
            elif part.startswith("lora_file_id="):
                selected_lora_file_id = part.split("=", 1)[1]
            elif part.startswith("original_name="):
                original_lora_name = part.split("=", 1)[1]

        if not selected_lora_name or not selected_lora_file_id or not original_lora_name:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("錯誤：無法獲取模型資訊。請重新嘗試。" if current_lang == "zh" else "Error: Could not retrieve model information. Please try again."))
            return

        current_state['selected_lora_info'] = {
            'name': selected_lora_name,
            'file_id': selected_lora_file_id,
            'original_name': original_lora_name,
            'is_local': False # Mark as not local, will be downloaded if needed
        }
        current_state['mode'] = 'image_generation'
        current_state['latest_trained_lora_path'] = None # Clear this as user selected from Drive

        reply_text = f"您已選擇 LoRA 模型：**{selected_lora_name}**。\n現在請輸入您的提示詞 (Positive Prompt)！" if current_lang == "zh" else \
                     f"You have selected LoRA model: **{selected_lora_name}**.\nNow please enter your prompt (Positive Prompt)!"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        logging.info(f"User {user_id} selected LoRA from Drive: {selected_lora_name} ({selected_lora_file_id}). Ready for generation.")
        return

    # --- 處理使用新訓練的 LoRA 預覽的 Postback 事件 ---
    elif data.startswith("action=use_new_lora_preview"):
        lora_path = data.split("lora_path=", 1)[1]
        lora_filename = os.path.basename(lora_path)
        lora_tag = lora_filename.replace('.safetensors', '').replace('.pt', '')

        if not os.path.exists(lora_path):
            line_bot_api.reply_message(event.reply_token, TextSendMessage("錯誤：找不到新的 LoRA 模型檔案。請重新訓練或選擇其他模型。" if current_lang == "zh" else "Error: New LoRA model file not found. Please retrain or select another model."))
            return

        # Set the selected_lora_info to the locally trained model
        current_state['selected_lora_info'] = {
            'name': lora_tag,
            'file_id': None, # No file ID for local file
            'original_name': lora_tag,
            'is_local': True, # Mark as local
            'local_path': lora_path # Store local path for direct use
        }
        current_state['latest_trained_lora_path'] = None # Consume the pending LoRA
        current_state['mode'] = 'image_generation' # Ensure mode is correct

        prompt = current_state.get('prompt')
        if not prompt:
            reply_text = "請先輸入提示詞。" if current_lang == "zh" else "Please enter prompt first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            f"正在使用您的新模型 **{lora_tag}** 生成預覽圖，請稍候..." if current_lang == "zh" else f"Generating preview image with your new model **{lora_tag}**, please wait..."
        ))
        logging.info(f"User {user_id} chose to preview with new LoRA: {lora_tag}.")

        # Generate preview images using the *newly trained* LoRA
        threading.Thread(target=generate_preview_and_ask_upload, args=(user_id, prompt, "ChilloutMix-Ni-pruned-fp32.safetensors", current_state['selected_lora_info'])).start()
        return

    # --- 處理不使用新訓練的 LoRA，選擇基礎模型 (從 "否，選擇其他" 按鈕) ---
    elif data == "action=select_base_model":
        current_state['latest_trained_lora_path'] = None # Clear the pending new LoRA
        send_base_model_selection(user_id, event.reply_token, current_lang)
        return

    # --- 處理預覽圖後的上傳確認 ---
    elif data.startswith("action=upload_lora"):
        lora_path_to_upload = data.split("lora_path=", 1)[1]
        if not os.path.exists(lora_path_to_upload):
            line_bot_api.reply_message(event.reply_token, TextSendMessage("錯誤：模型檔案不存在，無法上傳。" if current_lang == "zh" else "Error: Model file not found, cannot upload."))
            return

        line_bot_api.reply_message(event.reply_token, TextSendMessage("好的，正在將您的 LoRA 模型上傳到 Google Drive..." if current_lang == "zh" else "Okay, uploading your LoRA model to Google Drive..."))
        threading.Thread(target=upload_lora_to_drive_thread, args=(user_id, lora_path_to_upload)).start()
        # Reset relevant state after initiating upload
        current_state['latest_trained_lora_path'] = None
        current_state['mode'] = 'main_menu'
        return

    # --- 處理預覽圖後不滿意，不進行上傳 ---
    elif data == "action=skip_upload_lora":
        line_bot_api.reply_message(event.reply_token, TextSendMessage("好的，LoRA 模型將不會上傳到 Google Drive。您可以隨時重新訓練或選擇其他模型。" if current_lang == "zh" else "Okay, the LoRA model will not be uploaded to Google Drive. You can retrain or select other models at any time."))
        # Clear the locally trained LoRA path and reset mode
        current_state['latest_trained_lora_path'] = None
        # Optionally, remove the local LoRA file if it's no longer needed
        # if current_state['selected_lora_info'] and current_state['selected_lora_info']['is_local']:
        #     if os.path.exists(current_state['selected_lora_info']['local_path']):
        #         os.remove(current_state['selected_lora_info']['local_path'])
        #         logging.info(f"Deleted local LoRA file: {current_state['selected_lora_info']['local_path']}")
        current_state['selected_lora_info'] = None
        current_state['mode'] = 'main_menu'
        send_main_menu(user_id) # Send main menu for next action
        return

    logging.warning(f"Unhandled postback event from user {user_id}: {data}")

# --- 執行完整的訓練與生圖流程 ---
def run_full_pipeline(user_id):
    global uploading
    drive_service_pipeline = authenticate_google_drive()
    if not drive_service_pipeline:
        line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，請檢查服務帳戶金鑰。"))
        user_states[user_id]['mode'] = 'main_menu'
        user_states[user_id]['image_count'] = 0
        global uploading
        uploading = False
        return

    current_lang = user_states.get(user_id, {}).get('lang', 'zh')
    current_state = user_states[user_id] # Get current state to store latest_trained_lora_path

    try:
        line_bot_api.push_message(user_id, TextSendMessage(text="圖片上傳完成，開始標註啦!" if current_lang == "zh" else "Image upload complete, starting tagging!"))
        logging.info(f"User {user_id} started tagging process.")

        # === Step 1: 執行標註腳本 ===
        print("🚀 開始執行標註腳本...")
        tag_result = subprocess.run(
            [
                TAGGER_PYTHON,
                TAGGING_SCRIPT,
                UPLOAD_FOLDER,
                "--character_name", "chien",
                "--trigger_word", "masterpiece"
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR
        )
        logging.info(f"Tagger stdout for user {user_id}:\n{tag_result.stdout}")
        logging.error(f"Tagger stderr for user {user_id}:\n{tag_result.stderr}")
        line_bot_api.push_message(user_id, TextSendMessage(text="標註完成，執行模型訓練，需要大概10-15分鐘。" if current_lang == "zh" else "Tagging complete, starting model training, this will take about 10-15 minutes."))

        # === Step 2: 標註成功後執行訓練 ===
        print("🚀 開始執行模型訓練腳本...")
        train_result = subprocess.run(
            ['bash', TRAIN_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR,
            timeout=3000
        )
        logging.info(f"Train stdout for user {user_id}:\n{train_result.stdout}")
        logging.error(f"Train stderr for user {user_id}:\n{train_result.stderr}")
        line_bot_api.push_message(user_id, TextSendMessage(text="模型訓練已成功完成！🎉 開始準備生成預覽圖..." if current_lang == "zh" else "Model training completed successfully! 🎉 Preparing to generate preview images..."))

        # === Step 3: 訓練成功後，找到訓練好的 LoRA 模型檔案 ===
        lora_files = [f for f in os.listdir(LORA_OUTPUT_FOLDER) if f.endswith('.safetensors') or f.endswith('.pt')]
        lora_files.sort(key=lambda x: os.path.getmtime(os.path.join(LORA_OUTPUT_FOLDER, x)), reverse=True)

        if lora_files:
            latest_lora_file = lora_files[0]
            latest_lora_path = os.path.join(LORA_OUTPUT_FOLDER, latest_lora_file)
            current_state['latest_trained_lora_path'] = latest_lora_path # Store the path

            # Prompt user to enter a prompt for preview generation
            reply_text = f"恭喜！您的專屬 LoRA 模型 **{latest_lora_file.replace('.safetensors', '').replace('.pt', '')}** 已訓練完成！\n\n現在請輸入一個提示詞，我們將為您生成幾張預覽圖，讓您評估模型效果。\n（如果您對模型不滿意，可以選擇不將它上傳到 Google Drive。）" if current_lang == "zh" else \
                         f"Congratulations! Your exclusive LoRA model **{latest_lora_file.replace('.safetensors', '').replace('.pt', '')}** has been trained!\n\nPlease enter a prompt now. We will generate some preview images for you to evaluate the model's quality.\n(If you are not satisfied with the model, you can choose not to upload it to Google Drive.)"
            line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
            current_state['mode'] = 'image_generation' # Switch to image generation mode to receive prompt
            current_state['prompt'] = None # Clear prompt for new input

        else:
            line_bot_api.push_message(user_id, TextSendMessage("❌ 未找到訓練完成的 LoRA 模型檔案。請檢查訓練腳本的輸出路徑。" if current_lang == "zh" else "❌ No trained LoRA model file found. Please check the training script's output path."))
            logging.error(f"No trained LoRA model file found in {LORA_OUTPUT_FOLDER} after training.")
            send_main_menu(user_id) # Go back to main menu
            current_state['mode'] = 'main_menu'


    except subprocess.CalledProcessError as e:
        error_message = f"腳本執行失敗。錯誤訊息：{e.stderr}" if current_lang == "zh" else f"Script execution failed. Error: {e.stderr}"
        logging.error(f"Script execution failed for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"圖片標註或訓練失敗：{error_message}"))

    except FileNotFoundError as e:
        error_message = f"找不到必要的檔案：{e.filename}" if current_lang == "zh" else f"Required file not found: {e.filename}"
        logging.error(f"File not found for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"錯誤：{error_message}"))

    except subprocess.TimeoutExpired as e:
        error_message = f"腳本執行超時。命令：{e.cmd}" if current_lang == "zh" else f"Script execution timed out. Command: {e.cmd}"
        logging.error(f"Script timeout for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"錯誤：訓練腳本執行超時，請稍後再試或檢查設定。"))

    except Exception as e:
        error_message = f"執行過程中發生未知錯誤：{str(e)}" if current_lang == "zh" else f"An unknown error occurred during processing: {str(e)}"
        logging.error(f"Unknown error during pipeline for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"處理過程中出錯：{error_message}"))

    finally:
        # Clear UPLOAD_FOLDER only if it's not needed for re-uploading, i.e., after initial processing
        if os.path.exists(UPLOAD_FOLDER) and not uploading: # Only clear if not in an active upload state
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)
            logging.info(f"Cleared UPLOAD_FOLDER contents after pipeline for user {user_id}.")

        user_states[user_id]['image_count'] = 0
        uploading = False 

# --- 生成預覽圖並詢問是否上傳 ---
def generate_preview_and_ask_upload(user_id, prompt, base_model_name, lora_info):
    current_lang = user_states.get(user_id, {}).get('lang', 'zh')
    generated_image_urls = []
    try:
        # Use the provided lora_info, which contains 'local_path' if it's the newly trained one
        lora_tag = lora_info['original_name']
        lora_path_for_sd = lora_info['local_path'] if lora_info['is_local'] else None

        if not lora_path_for_sd or not os.path.exists(lora_path_for_sd):
            line_bot_api.push_message(user_id, TextSendMessage("錯誤：無法找到用於生成預覽圖的 LoRA 模型。" if current_lang == "zh" else "Error: Could not find the LoRA model for preview generation."))
            send_main_menu(user_id)
            return

        # Ensure the base model is set
        model_path = os.path.join(MODEL_FOLDER, base_model_name)
        SD_API.util_set_model(model_path)
        SD_API.util_wait_for_ready()
        logging.info(f"SD_API model set to {base_model_name} for preview generation for user {user_id}.")

        # Add LoRA to prompt for preview
        prompt_en = GoogleTranslator(source='auto', target='en').translate(prompt) if current_lang == "zh" else prompt
        prompt_with_lora = f"<lora:{lora_tag}:0.8>, {prompt_en}"
        negative_prompt = "worst quality, low quality, blurry, nsfw, nude, naked, nipples, sex, bad anatomy, watermark, text"

        line_bot_api.push_message(user_id, TextSendMessage(f"正在使用您的新模型 **{lora_tag}** 生成預覽圖 (2 張)..." if current_lang == "zh" else f"Generating 2 preview images with your new model **{lora_tag}**..."))

        # Generate 2 preview images
        for i in range(2):
            result = SD_API.txt2img(
                prompt=prompt_with_lora,
                negative_prompt=negative_prompt,
                width=512,
                height=768,
                steps=20,
                cfg_scale=7,
                sampler_name="DPM++ 2M Karras",
                n_iter=1,
                seed=-1, # Use random seed for each preview
                save_images=True,
                do_not_save_grid=True,
                override_settings={"sd_model_checkpoint": base_model_name} # Ensure base model is used
            )
            
            if result.images:
                img_byte_arr = io.BytesIO()
                result.images[0].save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0) # Reset stream position

                preview_filename = f"preview_{lora_tag}_{i+1}.png"
                preview_path = os.path.join(CONSOLIDATED_OUTPUT_FOLDER, preview_filename)
                with open(preview_path, 'wb') as f:
                    f.write(img_byte_arr.getvalue())
                
                logging.info(f"Generated preview image saved to {preview_path}")

                public_url = upload_to_drive_and_get_public_link(drive_service, preview_path, GOOGLE_DRIVE_OUTPUT_FOLDER_ID)
                if public_url:
                    generated_image_urls.append(public_url)
                    # Send image to user
                    line_bot_api.push_message(user_id, ImageSendMessage(original_content_url=public_url, preview_image_url=public_url))
                    logging.info(f"Sent preview image {i+1} to user {user_id}: {public_url}")
                else:
                    line_bot_api.push_message(user_id, TextSendMessage("無法上傳預覽圖到 Google Drive。" if current_lang == "zh" else "Could not upload preview image to Google Drive."))
                    logging.error(f"Failed to upload preview image {preview_filename} for user {user_id}.")

            else:
                line_bot_api.push_message(user_id, TextSendMessage(f"預覽圖 {i+1} 生成失敗。" if current_lang == "zh" else f"Preview image {i+1} generation failed."))
                logging.error(f"Failed to generate preview image {i+1} for user {user_id}.")
                break # Stop if one preview fails

        if generated_image_urls:
            # Ask user if they like the model and want to upload it
            lora_path_to_upload = lora_info['local_path'] # Get the actual path for the upload button
            confirm_message = TextSendMessage(
                text="您對這些預覽圖滿意嗎？如果滿意，可以將您的 LoRA 模型上傳到 Google Drive，這樣以後就可以隨時使用它生圖了！" if current_lang == "zh" else
                     "Are you satisfied with these preview images? If so, you can upload your LoRA model to Google Drive, so you can use it anytime for image generation!"
            )
            confirm_buttons = TemplateSendMessage(
                alt_text="上傳 LoRA 模型？",
                template=ButtonsTemplate(
                    title="上傳您的 LoRA 模型？" if current_lang == "zh" else "Upload Your LoRA Model?",
                    text="選擇是否將模型儲存到雲端" if current_lang == "zh" else "Choose whether to save your model to the cloud",
                    actions=[
                        PostbackAction(
                            label="是，上傳！" if current_lang == "zh" else "Yes, upload!",
                            data=f"action=upload_lora&lora_path={lora_path_to_upload}"
                        ),
                        PostbackAction(
                            label="否，跳過上傳" if current_lang == "zh" else "No, skip upload",
                            data="action=skip_upload_lora"
                        )
                    ]
                )
            )
            line_bot_api.push_message(user_id, [confirm_message, confirm_buttons])
        else:
            line_bot_api.push_message(user_id, TextSendMessage("預覽圖生成失敗，LoRA 模型將不會上傳。您可以嘗試重新訓練。" if current_lang == "zh" else "Preview image generation failed, LoRA model will not be uploaded. You can try retraining."))
            # Clean up the local LoRA if preview failed
            if lora_info['is_local'] and os.path.exists(lora_info['local_path']):
                os.remove(lora_info['local_path'])
                logging.info(f"Deleted local LoRA file due to preview failure: {lora_info['local_path']}")
            user_states[user_id]['latest_trained_lora_path'] = None
            user_states[user_id]['selected_lora_info'] = None
            user_states[user_id]['mode'] = 'main_menu'
            send_main_menu(user_id) # Go back to main menu


    except Exception as e:
        logging.error(f"Error during preview generation for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(f"生成預覽圖時發生錯誤：{str(e)}" if current_lang == "zh" else f"An error occurred during preview generation: {str(e)}"))
        # Clean up local LoRA if any error during preview
        if lora_info['is_local'] and os.path.exists(lora_info['local_path']):
            os.remove(lora_info['local_path'])
            logging.info(f"Deleted local LoRA file due to error during preview: {lora_info['local_path']}")
        user_states[user_id]['latest_trained_lora_path'] = None
        user_states[user_id]['selected_lora_info'] = None
        user_states[user_id]['mode'] = 'main_menu'
        send_main_menu(user_id) # Go back to main menu

    finally:
        # Clean up generated preview images from local storage after they are uploaded/handled
        if os.path.exists(CONSOLIDATED_OUTPUT_FOLDER):
            for f in os.listdir(CONSOLIDATED_OUTPUT_FOLDER):
                os.remove(os.path.join(CONSOLIDATED_OUTPUT_FOLDER, f))
            logging.info(f"Cleared CONSOLIDATED_OUTPUT_FOLDER contents after preview for user {user_id}.")


# --- 獨立的 LoRA 上傳到 Google Drive 函數 (for threading) ---
def upload_lora_to_drive_thread(user_id, lora_path_to_upload):
    current_lang = user_states.get(user_id, {}).get('lang', 'zh')
    drive_service_thread = authenticate_google_drive() # Get a new service instance for the thread
    if not drive_service_thread:
        line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，LoRA 模型上傳失敗。"))
        return

    try:
        uploaded_lora_url = upload_to_drive_and_get_public_link(
            drive_service_thread, lora_path_to_upload, GOOGLE_DRIVE_LORA_MODELS_FOLDER_ID
        )

        if uploaded_lora_url:
            lora_filename = os.path.basename(lora_path_to_upload)
            line_bot_api.push_message(user_id, TextSendMessage(f"您的 LoRA 模型 **{lora_filename}** 已成功上傳到 Google Drive！您現在可以在「我的 LoRA 模型」中選擇它來生圖了。\n連結：{uploaded_lora_url}" if current_lang == "zh" else f"Your LoRA model **{lora_filename}** has been successfully uploaded to Google Drive! You can now select it from 'My LoRA Models' to generate images.\nLink: {uploaded_lora_url}"))
            logging.info(f"Uploaded {lora_filename} to Google Drive from preview confirmation. Public link: {uploaded_lora_url}")
        else:
            line_bot_api.push_message(user_id, TextSendMessage("LoRA 模型上傳 Google Drive 失敗。請聯絡管理員。" if current_lang == "zh" else "Failed to upload LoRA model to Google Drive. Please contact the administrator."))

    except Exception as e:
        logging.error(f"Error uploading LoRA model {lora_path_to_upload} to Drive: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(f"上傳 LoRA 模型時發生錯誤：{str(e)}" if current_lang == "zh" else f"An error occurred while uploading LoRA model: {str(e)}"))
    finally:
        # Always clean up the local LoRA model after upload attempt
        if os.path.exists(lora_path_to_upload):
            os.remove(lora_path_to_upload)
            logging.info(f"Deleted local trained LoRA file after upload attempt: {lora_path_to_upload}")
        # Reset the state related to the latest trained LoRA
        user_states[user_id]['latest_trained_lora_path'] = None
        user_states[user_id]['selected_lora_info'] = None
        user_states[user_id]['mode'] = 'main_menu'
        send_main_menu(user_id) # Return to main menu after process


# --- 執行 Stable Diffusion 生圖流程 (考慮 LoRA 下載) ---
def run_sd_generation(user_id, prompt, base_model_name, selected_lora_info, reply_token):
    # ...
    command = [
        'python3',
        GENERATE_SCRIPT,
        '--prompt', prompt,
        '--output_dir', GENERATED_IMAGES_OUTPUT_FOLDER,
        '--model_name', base_model_name
    ]

    if selected_lora_info and (selected_lora_info.get('is_local') or selected_lora_info.get('file_id')):
        # 處理 LoRA 模型路徑，確保是本地檔案路徑
        lora_path_for_generation = None
        if selected_lora_info.get('is_local') and selected_lora_info.get('local_path'):
            lora_path_for_generation = selected_lora_info['local_path']
        elif selected_lora_info.get('file_id'): # LoRA 從 Google Drive 下載而來
            lora_filename_with_ext = selected_lora_info['original_name']
            if not (lora_filename_with_ext.endswith('.safetensors') or lora_filename_with_ext.endswith('.pt')):
                lora_filename_with_ext += '.safetensors' # 確保副檔名
            
            download_path = os.path.join(SD_LORA_DOWNLOAD_FOLDER, lora_filename_with_ext)
            
            if not os.path.exists(download_path):
                # 如果檔案不存在，則從 Google Drive 下載
                line_bot_api.push_message(user_id, TextSendMessage(text=f"正在下載 LoRA 模型 {selected_lora_info['name']}..." if current_lang == "zh" else f"Downloading LoRA model {selected_lora_info['name']}..."))
                if download_file_from_drive(drive_service, selected_lora_info['file_id'], download_path):
                    lora_path_for_generation = download_path
                else:
                    logging.error(f"Failed to download LoRA model {selected_lora_info['name']} (ID: {selected_lora_info['file_id']}).")
                    line_bot_api.push_message(user_id, TextSendMessage(text="下載 LoRA 模型失敗，請重試或選擇其他模型。" if current_lang == "zh" else "Failed to download LoRA model. Please try again or select another model."))
                    return # 終止生成

            else: # 檔案已存在本地
                lora_path_for_generation = download_path
                logging.info(f"LoRA model {selected_lora_info['name']} already exists locally: {download_path}")

        if lora_path_for_generation:
            command.extend(['--lora_path', lora_path_for_generation])
        else:
            logging.warning(f"No valid LoRA path for generation, LoRA will not be applied.")

    # 執行你的生圖程式
    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=TRAIN_SCRIPT_DIR)
    # ... 處理生成的圖片並發送給用戶

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)