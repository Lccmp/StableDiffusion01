document.addEventListener("DOMContentLoaded", () => {
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".tab-panel");
    const tagButtons = document.querySelectorAll(".tag-btn");
    const descriptionTextarea = document.getElementById("description");
    const submitButton = document.getElementById("submit-button");
    const generatedImage = document.getElementById("result");
    const numImagesSlider = document.getElementById("num-images");
    const numImagesValue = document.getElementById("num-images-value");
    const imageQualitySlider = document.getElementById("image-quality");
    const imageQualityValue = document.getElementById("image-quality-value");
    const statusContainer = document.getElementById("status-container");
    const imageContainer = document.getElementById("image-container");
    const progressBar = document.getElementById("progress");
    let selectedModel = "";


    // 確認有沒有正確選取到標籤按鈕
    console.log(tagButtons);  // 這裡會顯示所有選取到的按鈕

    // 新增標籤按鈕的事件監聽器
    tagButtons.forEach(button => {
        button.addEventListener("click", () => {
            const text = button.getAttribute("data-text").trim();
            descriptionTextarea.value += text + ",";
        });
    });

    // 新增圖片選項的事件監聽器
    const styleButtons = document.querySelectorAll(".style-btn");
    styleButtons.forEach(button => {
        button.addEventListener("click", () => {
            selectedModel = button.getAttribute("data-model");
            styleButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.add("selected");

            // 發送請求到後端切換模型
            fetch("/set_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: selectedModel })
            })
            .then(res => res.json())
            .then(data => {
                if (data.model) {
                    alert("模型已切換為：" + data.model);
                } else {
                    alert("模型切換失敗：" + data.error);
                }
            });
        });
    });

    // Tab 切換功能
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            // 移除所有 Tab 的 active 樣式
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            // 切換對應的內容
            const target = tab.dataset.tab;
            panels.forEach(panel => {
                panel.classList.remove("active");
                if (panel.id === target) {
                    panel.classList.add("active");
                }
            });
        });
    });

    // 更新圖片張數滑動拉桿的值
    numImagesSlider.addEventListener("input", () => {
        numImagesValue.textContent = numImagesSlider.value;
    });

    // 更新圖片品質滑動拉桿的值
    imageQualitySlider.addEventListener("input", () => {
        imageQualityValue.textContent = imageQualitySlider.value;
    });

    // 處理表單提交
    submitButton.addEventListener("click", () => {
        if (!selectedModel) {  // 新增這段檢查
            alert("請選擇一個圖片風格！");
            return; // 如果沒選，直接停止送出
        }
        console.log("送出按鈕被點了！");
        const formData = new FormData();
        formData.append('description', descriptionTextarea.value);
        formData.append('num_images', numImagesSlider.value);
        formData.append('image_quality', imageQualitySlider.value);

        // 清空文字框內容並禁用送出按鈕
        descriptionTextarea.value = "";
        submitButton.disabled = true;
        submitButton.classList.add('disabled');

        // 顯示狀態容器並初始化進度條
        statusContainer.style.display = "flex";
        progressBar.style.width = "0%";

        // 模擬進度條更新
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 100) {
                progress += 10;
                progressBar.style.width = `${progress}%`;
            } else {
                clearInterval(interval);
            }
        }, 1000);

        fetch('/generate', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            // 清空原本圖片
            imageContainer.innerHTML = "";
            
            if (data.images && data.images.length > 0) {
                data.images.forEach(imgBase64 => {
                    const img = document.createElement("img");
                    img.src = `data:image/png;base64,${imgBase64}`;
                    img.style.width = "400px";
                    img.style.margin = "10px";
                    imageContainer.appendChild(img);
                });
                imageContainer.style.display = "block"; // 顯示圖片容器
            } else {
                imageContainer.style.display = "none"; // 沒圖時隱藏
                alert("沒有產生圖片，請試試其他 prompt");
            }

            // 重新啟用按鈕
            submitButton.disabled = false;
            submitButton.classList.remove('disabled');
            statusContainer.style.display = "none";
        })
        .catch(error => {
            console.error('生成失敗:', error);
            submitButton.disabled = false;
            submitButton.classList.remove('disabled');
            statusContainer.style.display = "none";
        });
    });
    
});