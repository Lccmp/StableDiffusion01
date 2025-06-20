from flask import Flask, request, jsonify, send_file, render_template
from diffusers import StableDiffusionPipeline
from accelerate import infer_auto_device_map, dispatch_model
import torch,gc
from io import BytesIO
from PIL import Image
import numpy as np
from rembg import remove  # 使用 rembg 套件進行去背
import os
from werkzeug.utils import secure_filename
import uuid
import subprocess
import fnmatch
from pathlib import Path
from tqdm import tqdm
from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags
import shutil 
from webuiapi import WebUIApi
import subprocess
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_DICT = {
    "realistic": "stabilityai/stable-diffusion-2-1",
    "illustration": "Lykon/dreamshaper-7",
    "american_comic": "ogkalu/Comic-Diffusion",
    "anime": "gsdf/Counterfeit-V2.5",
    "surreal": "dreamlike-art/dreamlike-photoreal-2.0",
    "3d_cartoon": "Yntec/ResidentCNZCartoon3D",
}

current_model = MODEL_DICT["illustration"]
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_upload_folder(character_name):
    folder_name = f"20_{character_name}"
    upload_folder = os.path.join(app.root_path, 'uploads', folder_name)
    os.makedirs(upload_folder, exist_ok=True)
    return upload_folder

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_model", methods=["POST"])
def set_model():
    global pipe, current_model
    data = request.get_json()
    new_model = data.get("model")
    if new_model in MODEL_DICT:
        current_model = MODEL_DICT[new_model]
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return jsonify({"message": "模型已切換", "model": new_model})
    return jsonify({"error": "無效的模型"}), 400

@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form.get("description")
    num_images = int(request.form.get("num_images", 1))
    image_quality = int(request.form.get("image_quality", 20))

    if not prompt:
        return jsonify({"error": "Description is required"}), 400

    # 固定的負面提示詞
    negative_prompt = (
        "low quality, blurry, distorted, ugly, bad anatomy, watermark, text, signature, "
        "bad anatomy, deformed body, extra limbs, missing limbs, bad hands, malformed fingers, "
        "long neck, nsfw"
    )

    # 生成圖片
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=image_quality,
        num_images_per_prompt=num_images
    ).images
    image_data = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append(img_str)

    # ✅ DEBUG print 放這裡才會被執行
    print(f"📤 Prompt received: {prompt}")
    print(f"🖼️ Generating {num_images} image(s) with {image_quality} steps.")
    print(f"✅ Returned images count: {len(image_data)}")

    return jsonify({"images": image_data})

@app.route("/remove_background", methods=["POST"])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image = Image.open(request.files['image']).convert("RGBA")
    output = remove(np.array(image))
    output_image = Image.fromarray(output)
    buffered = BytesIO()
    output_image.save(buffered, format="PNG")
    buffered.seek(0)
    return send_file(buffered, mimetype='image/png')

@app.route("/remove-bg")
def remove_page():
    return render_template("remove_bg.html")

@app.route("/upload_photos")
def upload_page():
    return render_template("photo.html")

@app.route("/generate-lora")
def lora_page():
    return render_template("generate_lora.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    if 'photos' not in request.files:
        return jsonify({'message': '沒有選擇任何檔案。'}), 400
    upload_root = os.path.join(app.root_path, 'uploads')
    if os.path.exists(upload_root):
        for subdir in os.listdir(upload_root):
            subdir_path = os.path.join(upload_root, subdir)
            if os.path.isdir(subdir_path):
                shutil.rmtree(subdir_path)   # 🔥 直接刪掉整個角色資料夾！
        print(f"✅ 已清除以舊照片資料夾")
        
    character_name = request.form.get('character_name')
    if not character_name:
        return jsonify({'message': '沒有提供角色名稱。'}), 400
    files = request.files.getlist('photos')
    filenames = []
    user_upload_folder = get_user_upload_folder(character_name)
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(user_upload_folder, filename))
            filenames.append(filename)
        else:
            return jsonify({'message': f'不允許的檔案類型：{file.filename}'}), 400
    return jsonify({'message': f'成功上傳 {len(filenames)} 張照片，角色名稱: {character_name}'}), 200

@app.route("/generate-tags", methods=["POST"])
def generate_tags():
    data = request.get_json()
    character_name = data.get("character_name", "").strip()
    trigger_word = data.get("trigger_word", "masterpiece")
    if not character_name:
        return jsonify({"error": "缺少角色名稱"}), 400
    
    image_folder = get_user_upload_folder(character_name)
    if not os.path.exists(image_folder):
        return jsonify({"error": "角色資料夾不存在"}), 400

    # 呼叫外部腳本
    cmd = [
        "python", "/home/user/lora-train/tagger/phi3_fast.py",  # 如果你的腳本不在同目錄，改成完整路徑
        image_folder,
        "--character_name", character_name,
        "--trigger_word", trigger_word,
        "--override"
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 600秒超時可調整
        if proc.returncode != 0:
            return jsonify({"error": "外部標註腳本錯誤", "detail": proc.stderr}), 500
        
        # 你可以自己解析 proc.stdout 內容，這裡先簡單回傳訊息
        return jsonify({"message": "標籤生成完成", "output": proc.stdout})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "標註腳本執行超時"}), 504
    except Exception as e:
        return jsonify({"error": f"標註腳本執行失敗: {str(e)}"}), 500

@app.route('/start-training', methods=['POST'])
def start_training():
    data = request.get_json()
    character_name = data.get("character_name")
    if not character_name:
        return jsonify({"error": "缺少角色名稱"}), 400
    try:
        subprocess.Popen(["bash", "/home/user/lora-train/lora-scripts/train-fish.sh", character_name])
        return jsonify({"message": f"已啟動訓練：{character_name}，（約要10-15分鐘）"}), 200
    except Exception as e:
        return jsonify({"error": f"啟動失敗：{str(e)}"}), 500

@app.route("/check-training-status/<character_name>")
def check_training_status(character_name):
    model_path = f"/home/user/FISH-WEB/newhtml/lora_output/{character_name}.safetensors"
    return jsonify({"status": "completed" if os.path.exists(model_path) else "running"})

@app.route('/download_lora/<character_name>')
def download_lora_model(character_name):
    model_path = f"/home/user/FISH-WEB/newhtml/lora_output/{character_name}.safetensors"
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return f"{character_name}.safetensors 不存在", 404

@app.route('/upload-lora', methods=['POST'])
def upload_lora():
    lora_file = request.files.get('lora_file')

    if not lora_file:
        return jsonify({'message': '請選擇檔案'}), 400

    save_dir = '/home/user/lora-train/lora-scripts/stable-diffusion-webui/models/Lora'
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.splitext(lora_file.filename)[0]
    save_path = os.path.join(save_dir, f"{filename}.safetensors")
    lora_file.save(save_path)

    global latest_uploaded_lora_name
    latest_uploaded_lora_name = filename

    return jsonify({
        'message': f'{filename}.safetensors 上傳成功！',
        'path': save_path  # ✅ 加上這行
    })


@app.route("/generate_lora_images", methods=["POST"])
def generate_lora_images():
    global latest_uploaded_lora_name
    
    prompt = request.form.get("prompt")
    lora_path = request.form.get("lora_path")
    lora_name = latest_uploaded_lora_name
    lora_weight = 1
    num_images = 4
    
    if not prompt or not lora_name:
        return jsonify({"error": "缺少 prompt 或 lora_name"}), 400
    full_prompt = f"<lora:{lora_name}:{lora_weight}>, {prompt},"
    negative_prompt = "worst quality, low quality, blurry, nsfw, nude, naked, nipples, sex, breasts, areola, genitals, pussy, penis, anal, bad anatomy, missing fingers, extra limbs, watermark, text"
    
    output_dir = os.path.join(app.root_path, "static", "lora_img")
    os.makedirs(output_dir, exist_ok=True)
    image_urls = []
    api = WebUIApi(host="127.0.0.1", port=7860)
    desired_model = "chilloutmix_NiPrunedFp32Fix.safetensors"
    if api.util_get_current_model() != desired_model:
        api.util_set_model(desired_model)
        api.util_wait_for_ready()
    for _ in range(num_images):
        result = api.txt2img(
            prompt=full_prompt,
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
        filename = f"{lora_name}_seed{seed}.png"
        filepath = os.path.join(output_dir, filename)
        result.image.save(filepath)
        
        # ✅ 前端會用這個 URL 來讀圖片
        image_urls.append(f"/static/lora_img/{filename}")
    return jsonify({"images": image_urls})

if __name__ == "__main__":
    os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
    app.run(host='0.0.0.0',debug=True, port=5001)
