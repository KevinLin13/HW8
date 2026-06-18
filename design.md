project:
  name: "SVM Kernel Trick 3D 視覺化展示"
  language: "Python"
  goal: >
    建立一個完整的 Support Vector Machine Kernel Trick 教學展示專案。
    此專案要展示一組在 2D 平面中無法用直線分開的資料：
    藍色點位於中間，紅色點分布在外圈。
    透過概念性的特徵映射，將資料從 2D 空間轉換到 3D 空間，
    使原本無法線性分開的資料，在 3D 空間中可以被一個 hyperplane，
    也就是決策平面分開。
    接著使用 sklearn 實作真正的 RBF SVM 決策邊界，
    最後建立 Streamlit / Plotly 互動式展示介面。

  target_audience: >
    適合初學到中階的機器學習學生。
    目標是幫助學生理解為什麼 Kernel Trick 可以讓 SVM 處理非線性分類問題。

  important_conceptual_note: >
    Manim 動畫中的 3D 映射是教學用的概念視覺化。
    不可以宣稱真正的 RBF Kernel 只是把資料轉成 3D。
    真正的 RBF Kernel 對應到的是非常高維，甚至可視為無限維的特徵空間。
    這裡使用 3D 動畫，是為了建立直覺理解。

  expected_project_structure:
    root:
      - "README.md"
      - "requirements.txt"
      - "phase1_manim/"
      - "phase2_sklearn/"
      - "phase3_streamlit/"
      - "assets/"
      - "outputs/"

    phase1_manim:
      - "svm_kernel_trick_3d.py"

    phase2_sklearn:
      - "rbf_svm_decision_boundary.py"
      - "dataset_utils.py"

    phase3_streamlit:
      - "app.py"
      - "plot_utils.py"
      - "model_utils.py"

  dependencies:
    core:
      - "numpy"
      - "matplotlib"
      - "scikit-learn"
      - "pandas"

    animation:
      - "manim"

    interactive:
      - "streamlit"
      - "plotly"

  coding_style:
    - "使用清楚易懂的變數名稱。"
    - "加入教學用途的註解。"
    - "避免過度複雜的抽象設計。"
    - "每一個階段都應該可以獨立執行。"
    - "若使用隨機資料，請固定 random seed。"
    - "以可讀性與教學清楚度優先，不追求過度最佳化。"

  visual_style:
    background: "深色或乾淨的中性色背景"
    blue_class: "中間類別資料點"
    red_class: "外圈類別資料點"
    decision_surface: "半透明決策平面或曲面"
    support_vectors: "清楚標示支持向量"
    annotations: "文字說明要簡短、清楚、適合教學"

phases:
  - phase: 1
    name: "Manim 概念動畫"
    objective: >
      使用 Manim 建立一個 3D 動畫，說明 Kernel Trick 的核心概念。
      動畫應該從 2D 中無法線性分開的資料開始，
      接著透過簡單的特徵映射將資料抬升到 3D，
      再畫出一個可以分開兩類資料的 hyperplane，
      最後將 3D 的決策平面投影回 2D，
      顯示它在原始平面中會形成一個非線性的圓形決策邊界。

    file_to_create: "phase1_manim/svm_kernel_trick_3d.py"

    main_scene_class: "SVMKernelTrick3D"

    library:
      name: "Manim Community Edition"
      required_features:
        - "Scene"
        - "ThreeDScene"
        - "Axes"
        - "ThreeDAxes"
        - "Dot"
        - "Dot3D"
        - "Surface"
        - "ParametricFunction"
        - "MathTex"
        - "Tex"
        - "VGroup"
        - "ValueTracker"
        - "camera rotation"

    dataset_design:
      description: >
        使用一組手動定義的簡單 2D 資料。
        藍色點應該集中在原點附近。
        紅色點應該形成外圈。
        整體視覺上要清楚呈現「中心類別」與「外圈類別」的差異。

      blue_points:
        - [0.0, 0.0]
        - [0.2, 0.1]
        - [-0.2, 0.1]
        - [0.1, -0.2]
        - [-0.1, -0.1]
        - [0.25, -0.15]
        - [-0.25, 0.0]

      red_points:
        - [1.2, 0.0]
        - [-1.2, 0.0]
        - [0.0, 1.2]
        - [0.0, -1.2]
        - [0.9, 0.9]
        - [-0.9, 0.9]
        - [0.9, -0.9]
        - [-0.9, -0.9]
        - [1.3, 0.4]
        - [-1.3, -0.4]

    mapping:
      primary_mapping:
        formula: "phi(x, y) = (x, y, x^2 + y^2)"
        explanation: >
          此映射會讓距離原點越遠的資料點具有越高的 z 值。
          因此，外圈紅色點會被抬升到較高的位置，
          中間藍色點則會維持在較低的位置。
          這樣原本在 2D 中被包圍的資料，
          就會在 3D 的高度方向上變得可以分開。

      optional_gaussian_mapping:
        formula: "z = exp(-gamma * (x^2 + y^2))"
        explanation: >
          這可以作為 RBF-like 的替代映射展示。
          但主要教學動畫建議使用 z = x^2 + y^2，
          因為它更容易直覺理解：
          距離中心越遠，高度越高。

    animation_sequence:
      - step: "開場標題"
        content:
          title: "Support Vector Machine：Kernel Trick"
          subtitle: "從 2D 非線性分類到 3D 線性分隔"

      - step: "顯示原始 2D 資料"
        content:
          - "繪製 2D x-y 座標軸。"
          - "在中心附近放置藍色資料點。"
          - "在外圈放置紅色資料點。"
          - "加入文字標籤：Original 2D Space。"
          - "加入文字標籤：Not linearly separable。"

      - step: "展示線性分隔失敗"
        content:
          - "畫出 2 到 3 條不同方向的直線。"
          - "展示沒有任何一條直線可以乾淨地分開中間藍點與外圈紅點。"
          - "讓失敗的直線淡出。"

      - step: "介紹特徵映射"
        content:
          - "顯示公式：phi(x, y) = (x, y, x^2 + y^2)。"
          - "用視覺方式說明新的 z 值代表資料點與原點的距離關係。"
          - "加入註解：距離中心越遠的點會被抬得越高。"

      - step: "將資料點抬升到 3D"
        content:
          - "將 2D 座標軸轉換成 3D 座標軸。"
          - "將每個點從 (x, y, 0) 移動到 (x, y, x^2 + y^2)。"
          - "使用平滑動畫，讓觀眾看到資料點被抬升的過程。"
          - "旋轉鏡頭以顯示 3D 深度。"
          - "藍色點應該停留在較低位置。"
          - "紅色點應該移動到較高位置。"

      - step: "畫出 3D 決策平面"
        content:
          - "畫出半透明水平平面 z = c。"
          - "c 建議使用 0.7，或選擇其他能清楚分開藍點與紅點的數值。"
          - "加入標籤：Hyperplane in Feature Space。"
          - "可以選擇性畫出兩個較淡的平行 margin 平面。"

      - step: "投影回 2D"
        content:
          - "展示 z = c 且 z = x^2 + y^2，因此 x^2 + y^2 = c。"
          - "在原始 2D 平面上畫出圓形決策邊界。"
          - "加入標籤：Non-linear decision boundary in original space。"

      - step: "結尾總結"
        content:
          - "顯示總結：高維特徵空間中的線性分隔面，會對應回原始空間中的非線性決策邊界。"
          - "補充說明：真正的 RBF SVM 使用的是隱含的高維特徵空間，不只是 3D。"

    visual_requirements:
      camera:
        - "一開始使用 2D 俯視角。"
        - "平滑轉換到 3D 透視角。"
        - "在展示 3D hyperplane 時，使用緩慢的鏡頭旋轉。"
        - "最後回到 2D 俯視角，顯示圓形決策邊界。"

      objects:
        - "藍色點與紅色點必須清楚區分。"
        - "hyperplane 應該是半透明的。"
        - "圓形決策邊界必須清楚可見。"
        - "所有公式與文字都必須容易閱讀。"

    output:
      render_command: "manim -pqh phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D"
      expected_output_file: "outputs/svm_kernel_trick_3d.mp4"

    acceptance_criteria:
      - "動畫清楚呈現原始 2D 資料無法線性分開。"
      - "畫面中有顯示映射公式 phi(x, y) = (x, y, x^2 + y^2)。"
      - "資料點有明顯從 2D 被抬升到 3D。"
      - "3D 空間中有顯示可分隔資料的 hyperplane。"
      - "投影回 2D 後，顯示圓形非線性決策邊界。"
      - "最後說明不可錯誤宣稱 RBF Kernel 只是 3D 映射。"

  - phase: 2
    name: "sklearn RBF SVM 真實決策邊界"
    objective: >
      使用 sklearn 實作真正的 RBF Kernel SVM 分類器。
      此階段要展示 SVC(kernel='rbf') 在原始 2D 空間中學到的非線性決策邊界。
      同時也要比較 Linear SVM 與 RBF SVM 的差異，
      並清楚標示 support vectors。

    files_to_create:
      - "phase2_sklearn/dataset_utils.py"
      - "phase2_sklearn/rbf_svm_decision_boundary.py"

    dataset:
      method: "使用 sklearn.datasets.make_circles 或手動生成外圈資料。"
      recommended_function: "make_circles"
      parameters:
        n_samples: 300
        factor: 0.35
        noise: 0.06
        random_state: 42

      label_design:
        inner_class: "blue"
        outer_class: "red"

    models:
      - name: "Linear SVM"
        sklearn_class: "SVC"
        parameters:
          kernel: "linear"
          C: 1.0

      - name: "RBF SVM"
        sklearn_class: "SVC"
        parameters:
          kernel: "rbf"
          C: 1.0
          gamma: 1.0

    implementation_tasks:
      - "建立資料集工具函數，回傳 X 與 y。"
      - "訓練 Linear SVM 模型。"
      - "訓練 RBF SVM 模型。"
      - "在 2D 輸入空間上建立 mesh grid。"
      - "使用 model.decision_function 計算網格上每個點的決策分數。"
      - "畫出 decision_function = 0 的決策邊界。"
      - "畫出 decision_function = +1 與 -1 的 margin 線。"
      - "用較大的圓圈或外框標示 support vectors。"
      - "為 Linear SVM 建立一張圖。"
      - "為 RBF SVM 建立一張圖。"
      - "將圖片輸出到 outputs 資料夾。"

    plots:
      linear_svm_plot:
        filename: "outputs/linear_svm_decision_boundary.png"
        title: "Linear SVM：無法良好分開圓形資料"
        required_elements:
          - "藍色內圈類別資料點"
          - "紅色外圈類別資料點"
          - "線性決策邊界"
          - "支持向量"

      rbf_svm_plot:
        filename: "outputs/rbf_svm_decision_boundary.png"
        title: "RBF SVM：非線性決策邊界"
        required_elements:
          - "藍色內圈類別資料點"
          - "紅色外圈類別資料點"
          - "圓形或曲線型決策邊界"
          - "支持向量"
          - "margin 曲線"

    educational_annotations:
      include_text:
        - "Linear SVM 使用直線作為決策邊界。"
        - "RBF SVM 可以在原始 2D 空間中形成非線性決策邊界。"
        - "Support vectors 是決定邊界位置的關鍵樣本。"
        - "C 控制模型對錯誤分類的懲罰程度。"
        - "gamma 控制單一訓練樣本的影響範圍。"

    run_command: "python phase2_sklearn/rbf_svm_decision_boundary.py"

    acceptance_criteria:
      - "程式可以正確執行且不報錯。"
      - "資料集呈現圓形非線性分類結構。"
      - "Linear SVM 在視覺上無法良好分開資料。"
      - "RBF SVM 能產生彎曲或圓形決策邊界。"
      - "support vectors 有被清楚標示。"
      - "輸出圖片成功儲存在 outputs 資料夾。"

  - phase: 3
    name: "Streamlit 與 Plotly 互動式展示"
    objective: >
      建立一個互動式網頁應用程式，
      讓使用者可以調整 SVM 的參數，
      並即時觀察決策邊界的變化。
      此應用程式需要包含 2D 決策邊界視覺化，
      以及 3D 概念性 feature mapping 視覺化。

    files_to_create:
      - "phase3_streamlit/app.py"
      - "phase3_streamlit/model_utils.py"
      - "phase3_streamlit/plot_utils.py"

    framework:
      app: "Streamlit"
      visualization: "Plotly"
      machine_learning: "scikit-learn"

    app_title: "互動式 SVM Kernel Trick 視覺化展示"

    sidebar_controls:
      - name: "kernel"
        type: "selectbox"
        options:
          - "linear"
          - "poly"
          - "rbf"
          - "sigmoid"
        default: "rbf"

      - name: "C"
        type: "slider"
        min: 0.01
        max: 100.0
        default: 1.0
        scale: "log"

      - name: "gamma"
        type: "slider"
        min: 0.01
        max: 10.0
        default: 1.0
        scale: "log"
        note: "僅適用於 rbf、poly 與 sigmoid kernel。"

      - name: "degree"
        type: "slider"
        min: 2
        max: 6
        default: 3
        note: "僅適用於 polynomial kernel。"

      - name: "noise"
        type: "slider"
        min: 0.0
        max: 0.3
        default: 0.06

      - name: "sample_size"
        type: "slider"
        min: 50
        max: 1000
        default: 300

      - name: "show_support_vectors"
        type: "checkbox"
        default: true

      - name: "show_3d_mapping"
        type: "checkbox"
        default: true

    main_sections:
      - section: "概念說明"
        content:
          - "說明藍色點代表內圈類別，紅色點代表外圈類別。"
          - "說明 Linear SVM 難以處理圓形分布資料。"
          - "說明 kernel method 可以幫助模型建立非線性決策邊界。"
          - "明確說明 3D 視覺化只是概念展示，而 sklearn 的 RBF SVM 使用的是隱含的高維特徵空間。"

      - section: "2D 決策邊界"
        visualization:
          type: "Plotly contour plot"
          required_elements:
            - "資料點散佈圖"
            - "decision_function = 0 的決策邊界"
            - "可選擇顯示 decision_function = +1 與 -1 的 margin 線"
            - "清楚標示 support vectors"
            - "顯示目前使用的 kernel、C、gamma 參數"

      - section: "3D 概念映射"
        visualization:
          type: "Plotly 3D scatter and surface"
          mapping_formula: "z = x^2 + y^2"
          required_elements:
            - "3D 資料點散佈圖"
            - "藍色內圈資料點位於較低 z 值"
            - "紅色外圈資料點位於較高 z 值"
            - "半透明水平決策平面"
            - "可旋轉的 3D 視角"

      - section: "參數解釋"
        content:
          - "C：數值越大，模型越努力正確分類訓練資料，但可能導致邊界更複雜。"
          - "gamma：數值越大，每個資料點的影響範圍越小，邊界可能更彈性，也可能過度擬合。"
          - "kernel：linear 產生線性邊界；rbf 產生彈性的非線性邊界；poly 產生多項式型邊界。"

    implementation_tasks:
      - "建立可重複使用的圓形資料產生函數。"
      - "建立可重複使用的 SVC 訓練函數，根據使用者選擇的 kernel 與參數訓練模型。"
      - "建立可重複使用的決策邊界 mesh grid 計算函數。"
      - "建立 Plotly 2D contour 圖，顯示決策邊界。"
      - "建立 Plotly 3D 圖，顯示 z = x^2 + y^2 的概念映射。"
      - "當使用者啟用 support vectors 顯示時，要清楚標示支持向量。"
      - "在 Streamlit 介面中加入教學說明文字。"
      - "整體版面要乾淨，適合課堂展示。"

    model_behavior:
      sklearn_model: "SVC"
      required_methods:
        - "fit"
        - "decision_function"
      support_vector_attribute: "support_vectors_"

    plotly_2d_requirements:
      - "使用 contour plot 顯示決策分數。"
      - "使用 scatter markers 顯示資料點。"
      - "使用不同顏色區分兩個類別。"
      - "用較大的標記或外框顯示 support vectors。"
      - "加入標題與座標軸標籤。"

    plotly_3d_requirements:
      - "使用 Scatter3d 顯示抬升後的資料點。"
      - "使用 Surface 顯示決策平面。"
      - "允許使用者旋轉 3D 視角。"
      - "座標軸標籤要清楚：x、y、z = x^2 + y^2。"
      - "決策平面要使用半透明效果。"

    run_command: "streamlit run phase3_streamlit/app.py"

    acceptance_criteria:
      - "Streamlit app 可以成功啟動。"
      - "使用者可以調整 kernel、C、gamma、degree、noise 與 sample size。"
      - "2D 決策邊界會隨參數變化而更新。"
      - "support vectors 可以選擇顯示或隱藏。"
      - "3D 概念映射圖可以旋轉。"
      - "介面清楚說明 3D mapping 是概念展示。"
      - "整體展示適合用於課堂現場教學。"

README:
  required_content:
    - "專案概述"
    - "機器學習概念說明"
    - "概念性 3D 映射與真正 RBF Kernel 的差異"
    - "如何安裝依賴套件"
    - "如何執行 Phase 1 Manim 動畫"
    - "如何執行 Phase 2 sklearn 示範"
    - "如何執行 Phase 3 Streamlit app"
    - "預期輸出結果"
    - "教學補充說明"

  installation_example: |
    pip install -r requirements.txt

  phase1_run_example: |
    manim -pqh phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D

  phase2_run_example: |
    python phase2_sklearn/rbf_svm_decision_boundary.py

  phase3_run_example: |
    streamlit run phase3_streamlit/app.py

requirements_txt:
  content:
    - "numpy"
    - "pandas"
    - "matplotlib"
    - "scikit-learn"
    - "manim"
    - "streamlit"
    - "plotly"

final_deliverables:
  - "一支 Manim 動畫，展示 SVM Kernel Trick 的 3D 概念。"
  - "一支 sklearn 程式，展示真正的 RBF SVM 決策邊界。"
  - "一個 Streamlit / Plotly 互動式 app。"
  - "一份 README，說明概念與每個階段的執行方式。"
  - "輸出圖片與動畫影片應儲存在 outputs 資料夾。"

quality_checklist:
  concept_accuracy:
    - "不可宣稱 RBF Kernel 真的只是把資料映射到 3D。"
    - "必須說明 Manim 中的 3D 映射是為了建立直覺。"
    - "要正確區分 feature space 中的 hyperplane 與 original space 中的非線性決策邊界。"

  code_quality:
    - "所有 script 都應該可以獨立執行。"
    - "函數應該具有重複使用性。"
    - "重要的教學邏輯要加入註解。"
    - "避免使用會造成跨作業系統錯誤的硬編碼路徑。"

  visual_quality:
    - "資料點必須容易區分。"
    - "決策邊界必須清楚。"
    - "3D 場景必須容易閱讀。"
    - "文字註解不應遮住重要視覺元素。"

  classroom_usability:
    - "展示內容應該不需要高深數學背景也能理解。"
    - "流程應清楚呈現：原始資料、轉換、分隔、投影回原空間。"
    - "互動式 app 應該能讓學生容易觀察 C 與 gamma 對決策邊界的影響。"