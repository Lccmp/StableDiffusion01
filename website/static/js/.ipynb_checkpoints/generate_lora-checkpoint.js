// 放在外面讓 HTML 可以呼叫它
function fillExamplePrompt(type) {
    const promptField = document.getElementById("prompt");
    let example = "";

    switch (type) {
        case 1:
            example = "looking at viewer, clothing, human focus, brown hair, unknown artist, digital media (artwork), 1girl";
            break;
        case 2:
            example = "photorealistic, beautiful girl, soft lighting, looking at viewer, delicate face, natural makeup, smooth skin, standing pose, medium shot, cozy indoor setting, wearing casual clothes, brown hair, 1girl, realistic eyes, high detail";
            break;
    }
    promptField.value = example;
}

document.addEventListener("DOMContentLoaded", function () {
    let currentLoraPath = "";
    document.getElementById("uploadLoraForm").addEventListener("submit", function(e) {
        e.preventDefault();
        const form = this;
        const button = form.querySelector("button");
        const formData = new FormData(form);

        // 👉 禁用按鈕 + 顯示提示
        button.disabled = true;
        button.textContent = "上傳中...請稍候";

        fetch("/upload-lora", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            alert(data.message);
            currentLoraPath = data.path; // ✅ 假設後端有一起回傳模型路徑
            
            // ✅ 回復按鈕狀態
            button.disabled = false;
            button.textContent = "上傳模型(請等跳出上傳成功提示)";
        })
        .catch(err => {
            alert("上傳失敗：" + err);
            button.disabled = false;
            button.textContent = "上傳模型";
        });
    });

    document.getElementById("loraForm").addEventListener("submit", function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        formData.append("lora_path", currentLoraPath); // ✅ 加這行
        fetch("/generate_lora_images", {
            method: "POST",
            body: formData
        })
        .then(async res => {
            const text = await res.text();
            try {
                const data = JSON.parse(text);
                const preview = document.getElementById("previewArea");
                preview.innerHTML = "";

                if (data.images && Array.isArray(data.images)) {
                    data.images.forEach(path => {
                        const img = document.createElement("img");
                        img.src = path;
                        preview.appendChild(img);
                    });
                } else if (data.error) {
                    alert("後端錯誤：" + data.error);
                } else {
                    alert("回傳資料格式錯誤");
                }
            } catch (err) {
                console.error("回傳非 JSON，內容如下：", text);
                alert("伺服器回傳格式錯誤，請打開控制台(console)查看詳細訊息");
            }
        })
        .catch(err => alert("錯誤：" + err));
    });
});

