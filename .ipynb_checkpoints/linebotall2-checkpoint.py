import os
import shutil
import subprocess
import threading
import torch
import io
import json
import logging
import traceback

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage,
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent, FlexSendMessage
)

from webuiapi import WebUIApi
from deep_translator import GoogleTranslator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === LINE Bot 設定 ===
LINE_CHANNEL_ACCESS_TOKEN = 'DmtvsyIb7ihW+QGFrUopuACakbTX7r2VuVhY0RlAtKygvdt9ZYL1x37NRCYuIjuulnMwDrgrZ0BtlpWh/J35fVEhSaxWTKgf/sMtn9esKl6vQyt64onjEgvGQKuz2dK0fCt9YJoflHcybmfymb+wpAdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '3519b0cce23b332f8e7d0f98b80b4187'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === 資料夾與腳本路徑設定 ===
UPLOAD_FOLDER = '/home/user/lora-train/lora-scripts/linebot_images/20_chien' # 圖片訓練暫存路徑
TAGGER_PYTHON = '/home/user/lora-train/tagger/venv/bin/python3' # 標註腳本的Python解釋器路徑
TAGGING_SCRIPT = '/home/user/lora-train/tagger/phi3_fast.py' # 標註腳本路徑
TRAIN_SCRIPT = '/home/user/lora-train/lora-scripts/train-t1-Copy1.sh' # 訓練腳本路徑
TRAIN_SCRIPT_DIR = '/home/user/lora-train/lora-scripts' # 訓練腳本所在目錄

# === 生圖腳本設定 ===
GENERATE_SCRIPT = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/chien-generate_random_seeds.py'

# === 統一的輸出資料夾，用於所有生成的圖片 ===
CONSOLIDATED_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/generated_images_output"

# === Stable Diffusion API 設定 ===
SD_API = WebUIApi(host="127.0.0.1", port=7860) # 請確保您的 Stable Diffusion WebUI 已在 7860 埠運行
MODEL_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Stable-diffusion" # 您的 Stable Diffusion 模型資料夾

# === Google Drive 設定 ===
GOOGLE_DRIVE_OUTPUT_FOLDER_ID = "1_9QLHupRK8e-CjfLh70mNdesVIuvemZ1" # 使用您提供的 ID
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "/home/user/lora-train/elite-mix-441517-k7-618329de7f11.json" # 服務帳戶金鑰路徑

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONSOLIDATED_OUTPUT_FOLDER, exist_ok=True) # 確保統一的輸出資料夾存在

# 全域狀態變數
uploading = False # 用於控制圖片訓練模式下的圖片上傳
user_id_for_process = None # 用於記錄觸發訓練流程的用戶 ID

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
    上傳單一圖片到 Google Drive，並設定公開權限，回傳公開連結(URL)
    """
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service_instance.files().create(body=file_metadata, media_body=media, fields='id').execute()

        # 設定公開權限
        drive_service_instance.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'},
        ).execute()

        # 取得公開連結
        image_url = f"https://drive.google.com/uc?export=view&id={file['id']}"
        return image_url
    except Exception as e:
        logging.error(f"上傳 Google Drive 失敗: {e}")
        traceback.print_exc()
        return None

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

def send_main_menu(user_id):
    """傳送主功能選單給使用者 (Flex Message 版本)"""
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
                    "color": "#FF69B4",
                    "action": {"type": "message", "label": "生圖功能", "text": "生圖功能"}
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#4A90E2",
                    "action": {"type": "message", "label": "圖片訓練功能", "text": "圖片訓練功能"}
                }
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
            'lang': 'zh'
        }
    current_state = user_states[user_id]
    current_mode = current_state['mode']
    current_lang = current_state['lang']

    logging.info(f"User {user_id} in mode '{current_mode}' sent text: '{message_text}'")

    if message_text == "menu" or message_text == "選單":
        send_main_menu(user_id)
        current_state['mode'] = 'main_menu'
        current_state['prompt'] = None
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
            # === 新增的清理邏輯：在開始新的圖片訓練前清空 UPLOAD_FOLDER ===
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
                logging.info(f"Cleared old UPLOAD_FOLDER contents for user {user_id} before new training.")
            os.makedirs(UPLOAD_FOLDER) # 確保資料夾存在

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
        reply_text = "請輸入您的提示詞 (Positive Prompt)！" if current_lang == "zh" else "Please enter your prompt (Positive Prompt)!"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        logging.info(f"User {user_id} entered image generation mode.")
        return

    if current_mode == 'image_training':
        reply_text = f"您目前在圖片訓練模式。請繼續上傳圖片，目前已上傳 {current_state['image_count']} 張。" if current_lang == "zh" else f"You are in image training mode. Please continue uploading images. Current: {current_state['image_count']} images."
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    if current_mode == 'image_generation':
        if current_state['prompt'] is None:
            current_state['prompt'] = message_text
            model_list = list_models()
            display_models = model_list[:4] if model_list else ["No models found.safetensors"]
            actions = [PostbackAction(label=model[:20], data=f"model={model}") for model in display_models]
            template = ButtonsTemplate(
                title="選擇模型" if current_lang == "zh" else "Choose Model",
                text="請選擇模型來生成圖片" if current_lang == "zh" else "Select model to generate image",
                actions=actions
            )
            line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text="Select model", template=template))
            logging.info(f"User {user_id} entered prompt: '{message_text}', awaiting model selection.")
        else:
            reply_text = "請先選擇模型。" if current_lang == "zh" else "Please choose a model first."
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    reply_text = "請輸入 'menu' 查看功能選單。" if current_lang == "zh" else "Please type 'menu' to see the function menu."
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

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
            current_state['mode'] = 'main_menu'

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
        user_states[user_id] = {'mode': 'main_menu', 'lang': 'zh'}
    current_state = user_states[user_id]
    current_lang = current_state['lang']

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
        logging.info(f"User {user_id} selected model: {model_name}, prompt: '{prompt}'")

        threading.Thread(target=run_sd_generation, args=(user_id, prompt, model_name, event.reply_token)).start()
        current_state['prompt'] = None
        current_state['mode'] = 'main_menu'
        return

    logging.warning(f"Unhandled postback event from user {user_id}: {data}")

# --- 執行完整的訓練與生圖流程 ---
def run_full_pipeline(user_id):
    global uploading
    drive_service_pipeline = authenticate_google_drive()
    if not drive_service_pipeline:
        line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，請檢查服務帳戶金鑰。"))
        return

    current_lang = user_states.get(user_id, {}).get('lang', 'zh')

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
        line_bot_api.push_message(user_id, TextSendMessage(text="模型訓練已成功完成！🎉 開始生成圖片..." if current_lang == "zh" else "Model training completed successfully! 🎉 Starting image generation..."))

        # === Step 3: 訓練成功後，執行生圖腳本 chien-generate_random_seeds.py ===
        print("🚀 開始執行生圖腳本 chien-generate_random_seeds.py...")
        # *** 不再清空 CONSOLIDATED_OUTPUT_FOLDER，讓圖片保留 ***
        # if os.path.exists(CONSOLIDATED_OUTPUT_FOLDER):
        #     for f in os.listdir(CONSOLIDATED_OUTPUT_FOLDER):
        #         os.remove(os.path.join(CONSOLIDATED_OUTPUT_FOLDER, f))
        #     logging.info(f"Cleared old contents of {CONSOLIDATED_OUTPUT_FOLDER} for user {user_id}.")

        generate_result = subprocess.run(
            ['python', GENERATE_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300
        )
        logging.info(f"Generate stdout for user {user_id}:\n{generate_result.stdout}")
        logging.error(f"Generate stderr for user {user_id}:\n{generate_result.stderr}")
        line_bot_api.push_message(user_id, TextSendMessage(text="圖片生成完成，開始上傳到 Google Drive..." if current_lang == "zh" else "Image generation complete, starting upload to Google Drive..."))

        # === Step 4: 將生成的圖片上傳到 Google Drive ===
        print("🚀 開始上傳圖片到 Google Drive...")
        image_urls = []
        # 從 CONSOLIDATED_OUTPUT_FOLDER 讀取所有圖片
        for filename in os.listdir(CONSOLIDATED_OUTPUT_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(CONSOLIDATED_OUTPUT_FOLDER, filename)
                public_link = upload_to_drive_and_get_public_link(
                    drive_service_pipeline, img_path, GOOGLE_DRIVE_OUTPUT_FOLDER_ID
                )
                if public_link:
                    image_urls.append(public_link)

        if len(image_urls) == 0:
            line_bot_api.push_message(user_id, TextSendMessage(text="⚠️ 未能上傳任何圖片到 Google Drive，請檢查設定或生圖結果。" if current_lang == "zh" else "⚠️ No images uploaded to Google Drive. Check settings or generation results."))
        else:
            for url in image_urls:
                try:
                    line_bot_api.push_message(user_id, ImageSendMessage(
                        original_content_url=url,
                        preview_image_url=url
                    ))
                except Exception as send_e:
                    logging.error(f"❌ 無法傳送圖片訊息 (LINE) for user {user_id}: {send_e}")
            line_bot_api.push_message(user_id, TextSendMessage(text="所有生成的圖片已上傳完成！" if current_lang == "zh" else "All generated images have been uploaded!"))

    except subprocess.CalledProcessError as e:
        error_message = f"腳本執行失敗。錯誤訊息：{e.stderr}" if current_lang == "zh" else f"Script execution failed. Error: {e.stderr}"
        logging.error(f"Script execution failed for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"圖片標註、訓練或生成失敗：{error_message}"))

    except FileNotFoundError as e:
        error_message = f"找不到必要的檔案：{e.filename}" if current_lang == "zh" else f"Required file not found: {e.filename}"
        logging.error(f"File not found for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"錯誤：{error_message}"))

    except subprocess.TimeoutExpired as e:
        error_message = f"腳本執行超時。命令：{e.cmd}" if current_lang == "zh" else f"Script execution timed out. Command: {e.cmd}"
        logging.error(f"Script timeout for user {user_id}: {error_message}")
        line_bot_api.push_message(user_id, TextSendMessage(text=f"錯誤：訓練或生圖腳本執行超時，請稍後再試或檢查設定。"))

    except Exception as e:
        error_message = f"執行過程中發生未知錯誤：{str(e)}" if current_lang == "zh" else f"An unknown error occurred during processing: {str(e)}"
        logging.error(f"Unknown error during pipeline for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=f"處理過程中出錯：{error_message}"))

    finally:
        # === 移除對 UPLOAD_FOLDER 和 CONSOLIDATED_OUTPUT_FOLDER 的清理 ===
        # 它們將在下次相關操作時被清理
        uploading = False
        user_states[user_id]['mode'] = 'main_menu'
        user_states[user_id]['image_count'] = 0
        logging.info(f"Process finished for user {user_id}. 'uploading' state reset to False. Local images are preserved.")


# --- 執行 Stable Diffusion 生圖流程 ---
def run_sd_generation(user_id, prompt, model_name, reply_token_for_push):
    current_lang = user_states.get(user_id, {}).get('lang', 'zh')

    filename_prefix = f"sd_gen_{user_id}_"
    # output_path_temp 不再使用，因為我們直接儲存到最終路徑

    try:
        prompt_en = GoogleTranslator(source='auto', target='en').translate(prompt) if current_lang == "zh" else prompt
        negative_prompt = "worst quality, low quality, blurry, nsfw, nude, naked, nipples, sex, bad anatomy, watermark, text"

        model_path = os.path.join(MODEL_FOLDER, model_name)
        SD_API.util_set_model(model_path)
        SD_API.util_wait_for_ready()
        logging.info(f"SD_API model set to {model_name} for user {user_id}.")

        result = SD_API.txt2img(
            prompt=prompt_en,
            negative_prompt=negative_prompt,
            width=512,
            height=768,
            steps=20,
            cfg_scale=7,
            sampler_name="DPM++ 2M Karras",
            seed=-1,
            enable_hr=False,
            save_images=True
        )

        seed = result.info["seed"]
        final_output_filename = f"{filename_prefix}{seed}.png"
        final_output_path = os.path.join(CONSOLIDATED_OUTPUT_FOLDER, final_output_filename)
        result.image.save(final_output_path)
        logging.info(f"SD image generated and saved to {final_output_path} for user {user_id}.")

        if drive_service:
            gdrive_url = upload_to_drive_and_get_public_link(drive_service, final_output_path, GOOGLE_DRIVE_OUTPUT_FOLDER_ID)
        else:
            gdrive_url = None
            logging.error("Google Drive service not available, cannot upload SD image.")
            line_bot_api.push_message(user_id, TextSendMessage(
                "Google Drive 服務未啟用，無法上傳圖片。" if current_lang == "zh" else "Google Drive service not available, cannot upload image."
            ))

        if gdrive_url:
            line_bot_api.push_message(user_id, [
                ImageSendMessage(original_content_url=gdrive_url, preview_image_url=gdrive_url),
                TextSendMessage(f"✅ {'圖片已生成，Seed:' if current_lang == 'zh' else 'Image generated, Seed:'} {seed}\n{gdrive_url}")
            ])
            logging.info(f"SD image sent to user {user_id}: {gdrive_url}")
        else:
            line_bot_api.push_message(user_id, TextSendMessage(
                "圖片生成完成，但上傳 Google Drive 失敗。" if current_lang == "zh" else "Image generation complete, but failed to upload to Google Drive."
            ))

    except Exception as e:
        logging.error(f"Error in Stable Diffusion image generation for user {user_id}: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(
            "生成圖片時發生錯誤，請稍後再試。" if current_lang == "zh" else "Error occurred during image generation, please try again later."
        ))
    finally:
        # === 移除對 final_output_path 的清理 ===
        # 讓手動生成的圖片也保留在 CONSOLIDATED_OUTPUT_FOLDER
        pass


# === 主程式 ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logging.info(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)