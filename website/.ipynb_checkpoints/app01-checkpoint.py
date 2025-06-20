from flask import Flask, request, jsonify, send_file, render_template
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
from PIL import Image
import numpy as np
from rembg import remove  # 使用 rembg 套件進行去背
import os
from werkzeug.utils import secure_filename
import uuid
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 可選模型對應表
MODEL_DICT = {
    "realistic": "stabilityai/stable-diffusion-2-1",
    "illustration": "Lykon/dreamshaper-7",
    "american_comic": "nitrosocke/Arcane-Diffusion",
    "anime": "gsdf/Counterfeit-V2.5",
    "surreal": "stabilityai/stable-diffusion-xl-base-1.0",
}

# 初始模型
current_model = MODEL_DICT["realistic"]
pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_upload_folder(user_id):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    os.makedirs(user_folder, exist_ok=True)  # 確保資料夾存在
    return user_folder

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
        pipe.enable_attention_slicing()
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float32)
            pipe = pipe.to("cpu")
        return jsonify({"message": "模型已切換", "model": new_model})
    return jsonify({"error": "無效的模型"}), 400

@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form.get("description")
    num_images = int(request.form.get("num_images", 1))
    image_quality = int(request.form.get("image_quality", 20))

    if not prompt:
        return jsonify({"error": "Description is required"}), 400

    # 生成圖片
    images = pipe(prompt, num_inference_steps=image_quality).images[:num_images]
    buffered = BytesIO()
    images[0].save(buffered, format="PNG")
    buffered.seek(0)

    return send_file(buffered, mimetype='image/png')

@app.route("/remove_background", methods=["POST"])
def remove_background():
    print("🚀 remove_background 被呼叫啦！")
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.convert("RGBA")  # 確保圖片有 alpha 通道

    # 使用 rembg 去背
    output = remove(np.array(image))
    output_image = Image.fromarray(output)

    buffered = BytesIO()    
    output_image.save(buffered, format="PNG")
    buffered.seek(0)

    return send_file(buffered, mimetype='image/png')

@app.route("/remove-bg", methods=["GET"])
def remove_page():
    return render_template("remove_bg.html")

@app.route("/upload_photos", methods=["GET"])
def upload_page():
    return render_template("photo.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    if 'photos' not in request.files:
        return jsonify({'message': '沒有選擇任何檔案。'}), 400

    files = request.files.getlist('photos')
    filenames = []
    user_id = str(uuid.uuid4())  # 為當前上傳的使用者產生一個唯一的 ID
    user_upload_folder = get_user_upload_folder(user_id)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(user_upload_folder, filename)
            file.save(filepath)
            filenames.append(filepath)
        else:
            return jsonify({'message': f'不允許的檔案類型：{file.filename}'}), 400

    if filenames:
        return jsonify({'message': f'成功上傳 {len(filenames)} 張照片，使用者 ID: {user_id}，檔案儲存於：{user_upload_folder}'}), 200
    else:
        return jsonify({'message': '沒有有效的圖片檔案被上傳。'}), 400
    
    
@app.route('/start-training', methods=['POST'])
def start_training():
    try:
        subprocess.Popen(["bash", "/home/user/lora-train/lora-scripts/train-t1.sh"])
        return "訓練腳本已啟動"
    except Exception as e:
        return f"啟動失敗：{e}"

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)