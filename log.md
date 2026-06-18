# 開發與優化紀錄

日期：2026-06-18

## 1. 專案規格與執行原則

- 閱讀 `design.md`，確認專案目標是建立完整的 SVM Kernel Trick 教學展示。
- 閱讀並持續遵守 `CLAUDE.md`：
  - 實作前先確認假設與成功條件。
  - 優先使用最簡單且可驗證的方案。
  - 只修改與需求直接相關的內容。
  - 每個階段完成後實際執行與測試。
- 確認專案分成三個可獨立執行的階段：
  1. Manim 3D 概念動畫。
  2. sklearn Linear/RBF SVM 靜態比較。
  3. Streamlit/Plotly 互動式展示。

## 2. 初始專案建立

建立以下主要內容：

- `phase1_manim/svm_kernel_trick_3d.py`
- `phase2_sklearn/dataset_utils.py`
- `phase2_sklearn/rbf_svm_decision_boundary.py`
- `phase3_streamlit/app.py`
- `phase3_streamlit/model_utils.py`
- `phase3_streamlit/plot_utils.py`
- `README.md`
- `requirements.txt`
- `outputs/`

Phase 2 使用固定 random seed 的 `make_circles` 資料，建立 Linear SVM 與
RBF SVM 的決策邊界、margin 與 support vectors 圖片。

初次視覺檢查時發現 sklearn 的原始標籤方向與規格相反，因此在資料產生層統一
翻轉標籤，確保藍色代表內圈、紅色代表外圈，而不是只在繪圖層交換顏色。

輸出並驗證：

- `outputs/linear_svm_decision_boundary.png`
- `outputs/rbf_svm_decision_boundary.png`

## 3. Streamlit 初版與匯入修正

完成可調整下列參數的 Streamlit App：

- kernel
- C
- gamma
- degree
- noise
- sample size
- support vectors 顯示
- 3D 概念映射顯示

第一次以命令執行：

```powershell
.\.venv\Scripts\streamlit.exe run phase3_streamlit/app.py
```

發現 `ModuleNotFoundError: No module named 'phase3_streamlit'`。原因是 Streamlit
直接執行腳本時的 import root 與 package 測試模式不同。

修正方式：

- 保留 package import。
- 增加直接執行時的 local import fallback。
- 同時驗證 Streamlit CLI 啟動與 `AppTest` 測試模式。

## 4. Manim 安裝問題與解決過程

原始系統為 Python 3.14。安裝 Manim 時，`moderngl` 與 `glcontext` 缺少
Python 3.14 的 Windows 預編譯 wheel，因此嘗試本機編譯，最後因缺少
Microsoft C++ Build Tools 而失敗。

採用的解法：

1. 從 Python 官方來源取得 Python 3.13.14。
2. 安裝到專案內的 `.python313`，不取代系統 Python。
3. 使用 Python 3.13 建立 `.venv`。
4. 在 `.venv` 中成功安裝 Manim Community 0.20.1。
5. 將 Manim 公式由 `MathTex` 改為 Unicode `Text`，避免額外依賴 LaTeX。

驗證結果：

- Manim import 成功。
- ModernGL import 成功。
- 低畫質完整動畫渲染成功。
- 高畫質完整動畫渲染成功。

最終影片：

- H.264
- 1920×1080
- 60 FPS
- 約 46.1 秒
- `outputs/svm_kernel_trick_3d.mp4`

## 5. Streamlit 3D 視覺化優化歷程

### 5.1 抬升動畫與效能

初版 3D 圖只有映射後的靜態結果，無法看到資料由 2D 移動到 3D 的過程。

優化：

- 新增從 `z=0` 到 `z=x²+y²` 的 Plotly frames。
- 新增播放、暫停與抬升進度控制。
- 使用 Streamlit cache 減少重複產生資料與訓練模型。
- 2D 圖加入 transition，並保留縮放視角。

後續依展示需求將抬升速度由每 frame 70 ms 放慢至 180 ms，完整抬升時間約由
1.5 秒延長至 3.8 秒。

### 5.2 持續慢速旋轉

第一次旋轉設計是按鈕觸發單次環繞，不符合「持續慢慢旋轉」的需求。

重新設計：

- 使用瀏覽器端 `requestAnimationFrame` 持續更新 Plotly camera。
- 約 55 秒環繞一圈。
- 使用者拖曳或滾輪縮放時暫停。
- 停止操作約 1.4 秒後，從使用者的新視角繼續旋轉。
- 分頁不可見時停止更新，降低不必要的 CPU 使用。

### 5.3 空間與材質美化

逐步加入：

- 深色 3D 場景背景。
- 映射拋物面漸層、光照與等高紋理。
- 半透明決策平面。
- 平面交線與投影回原始空間的圓形邊界。
- 更大的 x/y/z 顯示範圍。
- 拉遠相機與旋轉半徑。
- 正交投影，減少透視變形。
- 穿越原點的 x、y、z 軸線。

曾依參考圖加入三個彩色座標軸面；後續依需求移除彩色軸面，只保留白色軸線、
網格、映射曲面與真正的決策平面。

### 5.4 真正的 3D 球體

原本資料點使用 `Scatter3d` marker，看起來仍像面向鏡頭的平面圓點。

優化：

- 使用 `Mesh3d` 建立低多邊形球體。
- 加入 ambient、diffuse、specular、roughness 與 fresnel 光照設定。
- 保留球體在抬升動畫中的 z 座標更新。
- 為避免大量球體互相遮擋，3D 教學圖使用兩類平衡、固定且可重現的代表性抽樣。
- 模型訓練與 2D 邊界仍使用完整資料，不受 3D 顯示數量影響。
- 新增「3D 顯示球體數量」控制，範圍為 30–140，預設 80。

### 5.5 決策切割面標記

加入：

- `Decision hyperplane`
- `z = 0.55`
- 指向決策平面的 3D 引導線
- 跟隨 3D 空間旋轉的標記

## 6. Streamlit 影片整合

為避免 GitHub README 不一定支援 `<video>` 內嵌播放，將
`outputs/svm_kernel_trick_3d.mp4` 直接嵌入 Streamlit：

- 使用 `st.video` 顯示 MP4。
- 影片區塊預設展開。
- 頁面載入後自動播放。
- 因瀏覽器 autoplay 規則，預設靜音，使用者可自行開啟聲音。

## 7. Streamlit Community Cloud 部署錯誤

部署日誌顯示 Streamlit Cloud 使用 Python 3.14 安裝根目錄
`requirements.txt`。因其中包含 Manim，Linux 環境嘗試編譯：

- `glcontext`
- `moderngl`
- `pycairo`
- `manimpango`

最後因缺少 X11/Cairo 系統開發套件而失敗。

修正方式：

- 將 Cloud 與 Web App 需要的套件保留在 `requirements.txt`。
- 將 Manim 移至 `requirements-manim.txt`。
- Phase 1 本機渲染時才另外安裝 Manim。
- 使用乾淨的 Python 3.14 虛擬環境模擬 Cloud 安裝。
- 驗證依賴安裝、Streamlit `AppTest` 與 HTTP 200 均成功。

## 8. GitHub 與部署

Repository：

- <https://github.com/KevinLin13/HW8>

Live Demo：

- <https://svm0demo.streamlit.app/>

建立 `.gitignore` 排除：

- `.python313/`
- `.venv/`
- Manim `media/` render cache
- Python cache
- 測試 log
- 編輯器與作業系統暫存檔

主要 commits：

- `3c0cbeb`：建立完整 SVM Kernel Trick 專案。
- `12a4755`：修正 Streamlit Cloud 依賴安裝。
- `8e15121`：在 Streamlit 嵌入 Manim MP4。
- `bae3fc2`：影片預設展開並自動播放。
- `2458ce6`：放慢 3D 抬升動畫。

## 9. 最終驗證

今日完成並驗證：

- Phase 1 Manim 高畫質動畫。
- Phase 2 Linear/RBF SVM 靜態圖。
- Phase 3 Streamlit 互動式展示。
- 四種 SVM kernel 可訓練與顯示。
- 2D decision boundary、margin、support vectors。
- 3D 球體、抬升動畫、持續慢速旋轉與決策平面標記。
- Streamlit 內嵌、自動播放 MP4。
- Python 3.14 Cloud-like 依賴安裝。
- Streamlit AppTest。
- Streamlit HTTP 200。
- GitHub push 與 Streamlit Community Cloud 部署。

