# 113_2_7-2_專題成果_Stable Diffusion 影像生成技術之應用

## 簡介
這個專案包含兩個主要部分：  
1. LINE Bot 機器人  
2. 網站介面

---

## 執行說明

### LINE Bot
LINE Bot 的主程式是 `linebotall.py`，請確保你已經安裝相關套件並設定好 LINE Bot 的 Channel Token 等資訊，之後直接執行這個檔案即可啟動機器人。

### 網站
網站主程式位於 `webside` 資料夾內的 `app.py`。  
啟動前請先安裝好相關套件，並在命令列執行：  
```bash
python webside/app.py

網站會啟動在本機的預設端口（例如5000或5001），可以透過瀏覽器訪問。

專案結構
/linebotall.py       # LINE Bot 主程式
/webside/app.py      # 網站主程式
/其他資料夾與檔案    # 其他資源或模組

注意事項
請確保你有安裝所有需求套件（requirements.txt 或自行安裝）

請依照需要設定環境變數或設定檔（如 API key 等）
