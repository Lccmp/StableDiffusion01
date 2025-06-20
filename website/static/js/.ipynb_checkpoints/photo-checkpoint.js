document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('photo-upload');
    const previewContainer = document.getElementById('preview-container');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');

    const trainButton = document.getElementById('train-button');
    const trainStatus = document.getElementById('train-status');
    
    const tagButton = document.getElementById('tag-button');
    const tagStatus = document.getElementById('tag-status');
    const characterInput = document.getElementById('character-name');
    const triggerInput = document.getElementById('trigger-word'); // hidden 欄位

    fileInput.addEventListener('change', (event) => {
        previewContainer.innerHTML = ''; // 清空之前的預覽
        const files = event.target.files;
        for (const file of files) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('preview-image');
                previewContainer.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });

    uploadButton.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            alert('請選擇要上傳的圖片。');
            return;
        }

        uploadStatus.textContent = '正在上傳...';
        uploadStatus.classList.remove('success', 'error');

        const formData = new FormData();
        for (const file of files) {
            formData.append('photos', file);
        }
        formData.append('character_name', characterInput.value.trim());
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                uploadStatus.textContent = result.message || '照片上傳成功！';
                uploadStatus.classList.add('success');
                previewContainer.innerHTML = '';
                fileInput.value = '';
            } else {
                const error = await response.json();
                uploadStatus.textContent = error.message || '上傳失敗，請稍後再試。';
                uploadStatus.classList.add('error');
            }
        } catch (error) {
            console.error('上傳過程中發生錯誤:', error);
            uploadStatus.textContent = '上傳過程中發生錯誤，請檢查網路連線。';
            uploadStatus.classList.add('error');
        }
    });

    tagButton.addEventListener('click', async () => {
        tagStatus.textContent = '正在生成標籤...';
        tagStatus.classList.remove('success', 'error');

        const characterName = characterInput.value.trim();
        const triggerWord = triggerInput.value.trim();

        if (!characterName) {
            tagStatus.textContent = '請先輸入角色名稱。';
            tagStatus.classList.add('error');
            return;
        }

        try {
            const response = await fetch('/generate-tags', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    character_name: characterName,
                    trigger_word: triggerWord,
                    }),
            });

            if (response.ok) {
                const data = await response.json();
                tagStatus.textContent = data.tags || '生成標籤成功！';
                tagStatus.classList.add('success');
            } else {
                const error = await response.json();
                tagStatus.textContent = error.message || '生成標籤失敗';
                tagStatus.classList.add('error');
            }
        } catch (error) {
            console.error('生成標籤時發生錯誤:', error);
            tagStatus.textContent = '生成標籤時發生錯誤，請稍後再試。';
            tagStatus.classList.add('error');
        }
    });

    // 🔽 加上訓練按鈕邏輯
    trainButton.addEventListener('click', () => {
        const characterName = characterInput.value.trim();
        trainStatus.textContent = "";
        trainStatus.classList.remove('success', 'error');

        if (!characterName) {
            trainStatus.textContent = '請輸入角色名稱再開始訓練！';
            trainStatus.classList.add('error');
            return;
        }

        trainStatus.textContent = "開始訓練中...約要（5-10分鐘）";

        fetch('/start-training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character_name: characterName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                trainStatus.textContent = data.message;
                trainStatus.classList.add('success');

                // 👉 開始每 5 秒檢查訓練狀態
                const intervalId = setInterval(() => {
                    fetch(`/check-training-status/${encodeURIComponent(characterName)}`)
                    .then(res => res.json())
                    .then(statusData => {
                        if (statusData.status === 'completed') {
                            clearInterval(intervalId); // 停止輪詢
                        
                            // 顯示下載連結
                            const downloadLink = document.createElement('a');
                            downloadLink.href =`/download_lora/${encodeURIComponent(characterName)}`;
                            downloadLink.textContent = '下載訓練完成的模型';
                            downloadLink.className = 'download-button';
                            trainStatus.appendChild(document.createElement('br'));
                            trainStatus.appendChild(downloadLink);
                        } else {
                            console.log("模型尚未完成，繼續等待...");
                        }
                    });
                }, 5 * 60 * 1000);  // 5分鐘輪詢一次
            } else {
                trainStatus.textContent = data.error || '訓練啟動失敗';
                trainStatus.classList.add('error');
            }
        });
        
    // 記得要有這行，結束 trainButton 的事件函式
    });
    //要加東西家在這裏面
});

