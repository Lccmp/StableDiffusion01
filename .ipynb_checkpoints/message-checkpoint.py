import os
import io
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage,
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent
)
from webuiapi import WebUIApi
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from deep_translator import GoogleTranslator

# === 模型清單（初始化使用） ===
models = [
    "anything-v5.safetensors",
    "chilloutmix_NiPrunedFp32Fix.safetensors",
    "Put Stable Diffusion checkpoints here.txt",
    "v1-5-pruned-emaonly.safetensors"
]

# === LINE 設定 ===
LINE_CHANNEL_ACCESS_TOKEN = '0rwjRDagzNVijhSZar4dqnoApiws7t+I7F+kPxM53uF5q+gAiR1iXQ30e3EYKyqo4eOwXTxL/cZ4xe1vekh51Nw/ga/lwnPJhBbjnhilVZQQ6qtS9kMM7YEDCLMuuQC0WtDr8OhpGirQnzba/PuCZgdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '1384907b561ef2dc6eca1bd570c41688'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === Flask App ===
app = Flask(__name__)

# === Stable Diffusion 設定 ===
SD_API = WebUIApi(host="127.0.0.1", port=7860)
MODEL_FOLDER = "/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Stable-diffusion"

# === Google Drive 設定 ===
UPLOAD_FOLDER = "1_9QLHupRK8e-CjfLh70mNdesVIuvemZ1"
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "elite-mix-441517-k7-618329de7f11.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

# === 狀態記錄 ===
user_language = {}
user_prompt = {}

# === 工具函式 ===
def list_models():
    return [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".safetensors")]

def upload_to_drive(file_path):
    file_name = os.path.basename(file_path)
    file_metadata = {"name": file_name, "parents": [UPLOAD_FOLDER]}
    media = MediaFileUpload(file_path, mimetype="image/png")
    file = drive_service.files().create(
        body=file_metadata, media_body=media, fields="id"
    ).execute()
    file_id = file.get("id")
    return f"https://drive.google.com/uc?id={file_id}"

# === Webhook 接收 ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    print(f"[handle_message] user:{user_id}, msg: {msg}")

    if msg.lower() in ["中文", "chinese"]:
        user_language[user_id] = "zh"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("語言已切換為中文！請輸入提示詞。"))
        return
    elif msg.lower() in ["英文", "english"]:
        user_language[user_id] = "en"
        line_bot_api.reply_message(event.reply_token, TextSendMessage("Language switched to English! Please enter prompt."))
        return

    lang = user_language.get(user_id, "en")
    if user_id not in user_prompt:
        user_prompt[user_id] = msg
        model_list = list_models()[:4]
        actions = [PostbackAction(label=model[:20], data=f"model={model}") for model in model_list]
        template = ButtonsTemplate(
            title="選擇模型" if lang == "zh" else "Choose Model",
            text="請選擇模型" if lang == "zh" else "Select model to generate image",
            actions=actions
        )
        line_bot_api.reply_message(event.reply_token, TemplateSendMessage(alt_text="Select model", template=template))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage("請先選擇模型。" if lang == "zh" else "Please choose a model."))

# === 處理模型按鈕選擇 ===
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data
    print(f"[handle_postback] user:{user_id}, data: {data}")
    lang = user_language.get(user_id, "en")

    if data.startswith("model="):
        model = data.split("=", 1)[1]
        prompt = user_prompt.get(user_id, "")
        if not prompt:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("請先輸入提示詞。" if lang == "zh" else "Please enter prompt first."))
            return

        try:
            # 中文翻譯成英文
            if lang == "zh":
                prompt_en = GoogleTranslator(source='auto', target='en').translate(prompt)
            else:
                prompt_en = prompt

            negative_prompt = "worst quality, low quality, blurry, nsfw, nude, naked, nipples, sex, bad anatomy, watermark, text"

            # 切換模型時用完整路徑比較保險
            model_path = os.path.join(MODEL_FOLDER, model)
            SD_API.util_set_model(model_path)
            SD_API.util_wait_for_ready()

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
            filename = f"gen_{user_id}_{seed}.png"
            os.makedirs("output", exist_ok=True)
            output_path = os.path.join("output", filename)
            result.image.save(output_path)

            gdrive_url = upload_to_drive(output_path)

            # 用 reply_message 回覆用戶，使用 reply_token
            line_bot_api.reply_message(event.reply_token, [
                ImageSendMessage(original_content_url=gdrive_url, preview_image_url=gdrive_url),
                TextSendMessage(f"✅ {'圖片已生成，Seed:' if lang == 'zh' else 'Image generated, Seed:'} {seed}\n{gdrive_url}")
            ])

            # 清除用戶提示詞
            if user_id in user_prompt:
                del user_prompt[user_id]

        except Exception as e:
            print(f"Error in image generation: {e}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                "生成圖片時發生錯誤，請稍後再試。" if lang == "zh" else "Error occurred during image generation, please try again later."
            ))

# === 主程式 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
