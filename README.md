# COCO-4CLS (MS COCO 4-class Image Classification)

本專案以 MS COCO (2017) 為資料來源，建立「**四分類影像分類**」模型（**不是 object detection**）。
核心做法：從 COCO 標註中挑出**只包含單一目標類別且不含其他類別**的影像，轉成乾淨的 classification dataset 來訓練。

## 題目要求對應 (Deliverables)
1. ✅ 完整專案代碼（可直接推 GitHub；見本 repo）
2. ✅ README.md（本檔包含：介紹/技術棧/本地運行/Docker/API/測試帳號）
3. ✅ `.env.example`
4. ✅ 至少一份測試 CSV：`data/sample_inference.csv`
5. ✅ 系統架構圖：
   - Mermaid：本 README 內
   - 圖檔產生腳本：`scripts/export_architecture_diagram.py`（會輸出 `docs/architecture.png`）

---

## 選擇的 4 類別 (可自行替換)
預設使用 COCO categories：
- `person`
- `car`
- `dog`
- `bicycle`

你可以在 `.env` 或 CLI 參數改成其他 4 類。

---

## 技術棧 (Tech Stack)
- Training / Inference: **PyTorch**, **torchvision**
- Backbone: **timm**（預訓練模型）
- Data: **pycocotools**, pandas
- Eval: **scikit-learn**
- Serving API: **FastAPI** + Uvicorn
- Docker 部署：Dockerfile / docker-compose

---

## Backbone 選擇與理由 + 修改點
### Backbone：EfficientNetV2-S（預訓練 ImageNet）
理由：
- 在 accuracy / params / speed 之間非常均衡
- timm 可一行載入 pretrained weights，適合小資料或快速達標
- 對於 224x224 影像分類很成熟穩定

修改點（加分項）：
- 只替換 classification head（`num_classes=4`）
- Head 加 Dropout
- Training 加 **Label Smoothing**
- 可選 Mixup/CutMix（預設關閉，避免干擾你對指標的解讀）

---

## 本地執行 (Local Run)

### 1) 建立環境 & 安裝
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
