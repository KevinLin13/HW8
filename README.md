# SVM Kernel Trick 3D 視覺化展示

## Live Demo

<https://svm0demo.streamlit.app/>

這是一個供課堂教學使用的完整示範專案，從直覺動畫、真實模型到互動式網頁，
逐步說明 SVM 如何處理無法以直線分隔的圓形資料。

![SVM Kernel Trick 3D 視覺化展示](<assets/SVM Kernel Trick 3D 視覺化展示.png>)

## 動畫展示

<video
  src="https://raw.githubusercontent.com/KevinLin13/HW8/main/outputs/svm_kernel_trick_3d.mp4"
  controls
  width="100%">
</video>

## 核心概念

原始資料由藍色內圈與紅色外圈組成，無法用一條直線乾淨分開。
Phase 1 使用以下概念映射建立直覺：

```text
φ(x, y) = (x, y, x² + y²)
```

距離中心較遠的外圈點會被抬到較高的位置，因此能在三維空間中用水平平面分隔。
這個平面投影回原始空間後，會形成圓形決策邊界。

重要：這個三維映射只是教學用的概念展示。真正的 RBF kernel 使用隱含的高維、
甚至可視為無限維的特徵空間，不能解釋成單純將資料映射到三維。

## 專案結構

```text
phase1_manim/       Manim 概念動畫
phase2_sklearn/     Linear 與 RBF SVM 靜態比較
phase3_streamlit/   Streamlit / Plotly 互動展示
outputs/            產生的圖片與影片
```

## 安裝

建議使用 Python 3.11 至 3.13。本專案的 Manim 動畫以一般文字顯示公式，
因此不需要額外安裝 LaTeX。

```bash
pip install -r requirements.txt
```

`requirements.txt` 僅包含 Streamlit 網頁所需套件，確保可部署至
Streamlit Community Cloud。若要在本機渲染 Phase 1 動畫，請另外安裝：

```bash
pip install -r requirements-manim.txt
```

Windows 若系統預設是 Python 3.14，請改用 Python 3.13 建立虛擬環境，
避免 `moderngl` 缺少預編譯 wheel 而要求安裝 C++ 編譯器。本專案目前可直接使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

目前已驗證的環境版本為 Python 3.13.14 與 Manim Community 0.20.1。

## Phase 1：Manim 動畫

```bash
manim -pqh phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D
```

未啟用虛擬環境時，也可以直接執行：

```powershell
.\.venv\Scripts\manim.exe -pqh phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D
```

若只想快速確認環境與動畫是否正常，可先使用低畫質：

```powershell
.\.venv\Scripts\manim.exe -ql phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D
```

動畫依序呈現原始 2D 資料、線性分隔失敗、資料抬升到 3D、特徵空間中的
hyperplane，以及投影回 2D 後的圓形邊界。Manim 預設會將影片放在
`media/videos/`；已驗證的低畫質影片也存放於
`outputs/svm_kernel_trick_3d.mp4`。

## Phase 2：sklearn 靜態示範

```bash
python phase2_sklearn/rbf_svm_decision_boundary.py
```

程式使用固定 random seed 的 `make_circles` 資料，同時訓練 Linear SVM 與
RBF SVM，並標示決策邊界、margin 與 support vectors。

預期輸出：

- `outputs/linear_svm_decision_boundary.png`
- `outputs/rbf_svm_decision_boundary.png`

## Phase 3：Streamlit 互動展示

```bash
streamlit run phase3_streamlit/app.py
```

側邊欄可調整 kernel、C、gamma、degree、noise 與 sample size。2D Plotly
圖會即時更新決策邊界，support vectors 可顯示或隱藏；3D 圖則可自由旋轉。

## 參數解釋

- `C` 越大，模型越努力正確分類訓練資料，邊界也可能更複雜。
- `gamma` 越大，單一樣本的影響範圍越小，邊界更有彈性，也更可能過度擬合。
- `kernel` 決定邊界形式：`linear` 為線性；`rbf`、`poly` 可形成非線性邊界。

Support vectors 是最直接影響決策邊界位置的訓練樣本。互動操作時可觀察參數
如何改變 support vectors 的數量與決策邊界形狀。
