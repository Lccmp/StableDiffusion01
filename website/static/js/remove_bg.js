document.getElementById('imageInput').addEventListener('change', function (e) {
    const preview = document.getElementById('previewImg');
    const file = e.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            preview.src = event.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('bgForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('imageInput');
    formData.append('image', fileInput.files[0]);

    const resultImg = document.getElementById('resultImg');
    resultImg.style.display = 'none';

    try {
        const response = await fetch('/remove_background', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("上傳失敗！");
        }

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        resultImg.src = imageUrl;
        resultImg.style.display = 'block';
    } catch (error) {
        alert("去背失敗: " + error.message);
    }
});
