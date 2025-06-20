import os
import shutil
import subprocess
import threading
import torch # 即使不直接用Diffusers，但為了確保環境兼容性，保留torch的匯入
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage

# 匯入 Google Drive 相關模組 (如果你要上傳到雲端，則需要這些)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)

# LINE Bot 設定
LINE_CHANNEL_ACCESS_TOKEN = '0rwjRDagzNVijhSZar4dqnoApiws7t+I7F+kPxM53uF5q+gAiR1iXQ30e3EYKyqo4eOwXTxL/cZ4xe1vekh51Nw/ga/lwnPJhBbjnhilVZQQ6qtS9kMM7YEDCLMuuQC0WtDr8OhpGirQnzba/PuCZgdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '1384907b561ef2dc6eca1bd570c41688'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 資料夾與腳本路徑設定
UPLOAD_FOLDER = '/home/user/lora-train/lora-scripts/linebot_images/20_chien'
TAGGER_PYTHON = '/home/user/lora-train/tagger/venv/bin/python3'
TAGGING_SCRIPT = '/home/user/lora-train/tagger/phi3_fast.py'
TRAIN_SCRIPT = '/home/user/lora-train/lora-scripts/train-t1-Copy1.sh'
TRAIN_SCRIPT_DIR = '/home/user/lora-train/lora-scripts' # 設定訓練腳本所在資料夾

# === 生圖腳本設定 ===
# 你的生圖腳本的完整路徑
GENERATE_SCRIPT = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/chien-generate_random_seeds.py'
# 生圖腳本的輸出資料夾，與 chien-generate_random_seeds.py 中的 output_dir 變數一致
GENERATED_IMAGES_OUTPUT_FOLDER = "/home/user/lora-train/lora-scripts/random_output"


# === Google Drive 設定 (如果需要上傳) ===
GOOGLE_DRIVE_GENERATED_IMAGES_FOLDER_ID = "1_9QLHupRK8e-CjfLh70mNdesVIuvemZ1" # 請替換為你的 Google Drive 資料夾 ID
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "/home/user/lora-train/elite-mix-441517-k7-618329de7f11.json" # 你的服務帳戶金鑰路徑

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_IMAGES_OUTPUT_FOLDER, exist_ok=True) # 確保生圖輸出資料夾存在

uploading = False
user_id_for_process = None # 用來記錄觸發流程的用戶 ID

# --- Google Drive 認證初始化函數 ---
def authenticate_google_drive():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        print("Google Drive 認證成功！")
        return service
    except Exception as e:
        print(f"Google Drive 認證失敗: {e}")
        return None

# --- 上傳圖片到 Google Drive 並獲取公開連結函數 ---
def upload_to_drive_and_get_public_link(drive_service, file_path, folder_id):
    """
    上傳單一圖片到 Google Drive，並設定公開權限，回傳公開連結(URL)
    """
    from googleapiclient.http import MediaFileUpload
    
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        # 設定公開權限
        drive_service.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'},
        ).execute()

        # 取得公開連結
        image_url = f"https://drive.google.com/uc?export=view&id={file['id']}"
        return image_url
    except Exception as e:
        print(f"上傳 Google Drive 失敗: {e}")
        traceback.print_exc()
        return None

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"Line Bot Error: {e}")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    global uploading, user_id_for_process
    message = event.message.text.strip().lower()

    if message == 'send images':
        if not uploading:
            # ✅ 延遲清除：只有當新的上傳要開始，才清除上次的資料夾
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
                print("✅ 清除上一次的上傳資料夾")
            os.makedirs(UPLOAD_FOLDER)

            if os.path.exists(GENERATED_IMAGES_OUTPUT_FOLDER):
                shutil.rmtree(GENERATED_IMAGES_OUTPUT_FOLDER)
                print("✅ 清除上一次的生成圖片資料夾")
            os.makedirs(GENERATED_IMAGES_OUTPUT_FOLDER)

            uploading = True
            user_id_for_process = event.source.user_id
            reply = "請開始傳送圖片！（共 20 張）"
        else:
            reply = "已進入圖片上傳模式，請繼續上傳圖片"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )
        return  

def run_full_pipeline(user_id):
    """
    在單獨的線程中執行標註、訓練和生圖過程。
    """
    drive_service = None
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text="圖片上傳完成，開始標註啦!"))

        # === Step 1: 執行標註腳本 ===
        print("🚀 開始執行標註腳本...")
        tag_result = subprocess.run(
            [
                TAGGER_PYTHON,
                TAGGING_SCRIPT,
                UPLOAD_FOLDER,
                "--character_name", "角色名",
                "--trigger_word", "masterpiece"
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR
                                   )
        print("標註腳本標準輸出：", tag_result.stdout)
        print("標註腳本標準錯誤：", tag_result.stderr)
        line_bot_api.push_message(user_id, TextSendMessage(text="標註完成，執行模型訓練，需要大概10-15分鐘。"))

        # === Step 2: 標註成功後執行訓練 ===
        print("🚀 開始執行模型訓練腳本...")
        train_result = subprocess.run(
            ['bash', TRAIN_SCRIPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR,
            timeout=1800 # 增加訓練超時時間，例如 30 分鐘 (1800 秒)
        )
        print("訓練腳本標準輸出：\n", train_result.stdout)
        print("訓練腳本標準錯誤：\n", train_result.stderr)
        line_bot_api.push_message(user_id, TextSendMessage(text="模型訓練已成功完成！🎉 開始生成圖片..."))

        # === Step 3: 訓練成功後，執行生圖腳本 chien-generate_random_seeds.py ===
        print("🚀 開始執行生圖腳本 chien-generate_random_seeds.py...")
        # 確保生圖腳本的運行環境與 linebotall1.1.py 相同 (如果 webuiapi 需要在特定環境下運行)
        # 這裡假設你的 webuiapi 程式和 Flask 程式可以在同一個 'lora_env' 環境下運行
        # 如果你的 chien-generate_random_seeds.py 需要另一個環境，你需要自行修改 Python 執行器路徑
        generate_result = subprocess.run(
            ['python', GENERATE_SCRIPT], # 使用 'python3'，確保在當前環境中執行
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=TRAIN_SCRIPT_DIR, # 生圖腳本的相對路徑可能需要從 TRAIN_SCRIPT_DIR 執行
            timeout=300 # 生圖時間，例如 5 分鐘
        )
        print("生圖腳本標準輸出：\n", generate_result.stdout)
        print("生圖腳本標準錯誤：\n", generate_result.stderr)

        line_bot_api.push_message(user_id, TextSendMessage(text="圖片生成完成，開始上傳到 Google Drive..."))

        # === Step 4: 將生成的圖片上傳到 Google Drive ===
        print("🚀 開始上傳圖片到 Google Drive...")
        drive_service = authenticate_google_drive()
        if not drive_service:
            line_bot_api.push_message(user_id, TextSendMessage(text="❌ 無法連接到 Google Drive 服務，請檢查服務帳戶金鑰。"))
            return

        image_urls = []
        for filename in os.listdir(GENERATED_IMAGES_OUTPUT_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(GENERATED_IMAGES_OUTPUT_FOLDER, filename)
                public_link = upload_to_drive_and_get_public_link(
                    drive_service, img_path, GOOGLE_DRIVE_GENERATED_IMAGES_FOLDER_ID
                )
                if public_link:
                    image_urls.append(public_link)

        if len(image_urls) == 0:
            line_bot_api.push_message(user_id, TextSendMessage(text="⚠️ 未能上傳任何圖片到 Google Drive，請檢查設定。"))
        else:
            for url in image_urls:
                try:
                    line_bot_api.push_message(user_id, ImageSendMessage(
                        original_content_url=url,
                        preview_image_url=url
                    ))
                except Exception as send_e:
                    print(f"❌ 無法傳送圖片訊息 (LINE): {send_e}")
            line_bot_api.push_message(user_id, TextSendMessage(text="圖片已全部上傳完成！"))

    except subprocess.CalledProcessError as e:
        print("❌ 腳本執行失敗")
        error_output = f"Command: {e.cmd}\nReturn Code: {e.returncode}\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_output)
        final_reply = f"❌ 執行腳本錯誤：{os.path.basename(e.cmd[0])} 失敗。詳情請查看伺服器日誌。錯誤: {e.stderr}"
        line_bot_api.push_message(user_id, TextSendMessage(text=final_reply))

    except FileNotFoundError as e:
        print(f"❌ 找不到檔案：{e}")
        final_reply = f"❌ 找不到必要的檔案：{e.filename}"
        line_bot_api.push_message(user_id, TextSendMessage(text=final_reply))

    except subprocess.TimeoutExpired:
        print("⚠️ 腳本超時")
        final_reply = "⚠️ 某個腳本執行超時，可能尚未完成或發生錯誤。請檢查伺服器日誌。"
        line_bot_api.push_message(user_id, TextSendMessage(text=final_reply))

    except Exception as e:
        print(f"❌ 處理過程中出錯：{str(e)}")
        traceback.print_exc()
        final_reply = f"❌ 處理過程中出錯：{str(e)}"
        line_bot_api.push_message(user_id, TextSendMessage(text=final_reply))

    finally:
        # 清理資料夾
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        print("清理完成 UPLOAD_FOLDER")

        if os.path.exists(GENERATED_IMAGES_OUTPUT_FOLDER):
            shutil.rmtree(GENERATED_IMAGES_OUTPUT_FOLDER)
        print("清理完成 GENERATED_IMAGES_OUTPUT_FOLDER")

        # 將 uploading 狀態改為 False，允許下一次上傳
        uploading = False

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    global uploading, user_id_for_process

    if not uploading:
        return

    # 統計資料夾中的圖片數量，這裡只檢查 .jpg，如果 LINE 上傳的是 .png 或其他，需要調整
    current_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if current_count >= 20:
        print("⚠️ 已達 20 張圖片，忽略上傳")
        return

    new_index = current_count + 1
    # 儲存為 .png 格式，因為 LINE 上傳的圖片通常是 PNG
    filename = f"{new_index:02d}.png" # 建議儲存為 PNG
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    image_content = line_bot_api.get_message_content(event.message.id)
    with open(image_path, 'wb') as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    print(f"✅ 已儲存圖片 {filename}")

    if new_index == 20:
        print("✅ 收到 20 張圖片，準備開始標註、訓練與生圖")
        uploading = False

        # 在新的線程中執行標註、訓練和生圖
        threading.Thread(target=run_full_pipeline, args=(user_id_for_process,)).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5678, debug=False)