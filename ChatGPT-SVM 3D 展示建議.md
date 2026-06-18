# SVM 3D 展示建議

**User:** Anonymous  
**Created:** 6/18/2026 10:13:20  
**Updated:** 6/18/2026 12:02:57  
**Exported:** 6/18/2026 12:23:04  
**Link:** [https://chatgpt.com/c/6a33543c-cce0-83e8-a84c-d7519efe13fb](https://chatgpt.com/c/6a33543c-cce0-83e8-a84c-d7519efe13fb)  

## Prompt:
2026/6/18 上午10:13:06

我想要做一個support vector machine展示，做3D展示 互動 ，比如說我在X Y軸上面有藍色的在中間紅色的點點在外面，然後我使用kernel function 選讓他能夠變成3D的這看到方向這個核心函數可能是一個Gaussian 或者是其他S v M能夠用的函數，然後我想要畫出這個hyper plane就是決策面我想要用的是Manim 套件 你有什麼建議？

## Response:
2026/6/18 上午10:13:20

> **搜尋 Manim 3D 場景與 SVM 核心文件**
>
> [ThreeDScene - Manim Community v0.20.1](https://docs.manim.community/en/stable/reference/manim.scene.three_d_scene.ThreeDScene.html)
> [SVC — scikit-learn 1.9.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
> [scikit-learn](https://pypi.org/project/scikit-learn/)
> [sklearn.svm.SVC — scikit-learn 0.20.2 documentation - Huihoo](https://docs.huihoo.com/scikit-learn/0.20/modules/generated/sklearn.svm.SVC.html)
> [Examples - Manim Slides documentation](https://manim-slides.eertmans.be/v5.1.8/reference/examples.html)
> [Manim - 3D Scene](https://www.youtube.com/watch?v=Ma0FT7xMi9U)
> [sklearn.svm.SVC — scikit-learn 0.15-git documentation](https://jaquesgrobler.github.io/online-sklearn-build/modules/generated/sklearn.svm.SVC.html)
> [Scikit-learn](https://en.wikipedia.org/wiki/Scikit-learn)
> [8.26.1.1. sklearn.svm.SVC](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.svm.SVC.html)
> [3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera](https://arxiv.org/abs/1910.02527)
>
> **搜尋 Manim 3D 場景與各大網站資料**
>
> 
>
> **考慮動畫與交互模式**
>
> 我正在考慮在Manim中進行高質量動畫展示，並建議結合Streamlit/Plotly進行交互，以便用戶可以更靈活地操作參數。
>
> **考慮動畫與交互模式**
>
> 
>
> 處理時間為 17s

我會先把展示目標拆成「概念是否正確、3D 視覺怎麼設計、Manim 怎麼實作」三層，並補上適合交給 coding agent 的開發建議。

你的想法很適合做成 **SVM Kernel Trick 的 3D 概念動畫**，但我會建議你把展示分成兩層：  

第一層用 **Manim 做概念動畫**，清楚展示「2D 無法線性分開 → Kernel 映射到 3D → 用平面切開」。  
第二層再用 **sklearn RBF SVM / Plotly / Streamlit** 做真正可互動的決策邊界展示。原因是 Manim 很適合做高品質動畫，但不是最適合做滑桿式互動；Manim 的 `ThreeDScene` 本身主要是用來製作 3D 場景與鏡頭動畫。([Manim Community | Documentation](https://docs.manim.community/en/stable/reference/manim.scene.three_d_scene.ThreeDScene.html?utm_source=chatgpt.com))

---

## 一、你的展示概念是正確的，但要注意一個重要觀念

你描述的是這種資料：

- 藍色點在中間
- 紅色點在外圈
- 在原本的 X-Y 平面上，無法用一條直線分開
- 透過 Kernel Function，把資料「抬升」到 3D
- 在 3D 中用一個 hyperplane，也就是決策平面，把兩類分開

這個概念非常適合用下列映射展示：

```text
原始資料：
(x, y)

轉換成 3D：
(x, y, z)

其中：
z = x² + y²
```

或是用更像 Gaussian / RBF 的形式：

```text
z = exp(-γ(x² + y²))
```

如果用：

```text
z = x² + y²
```

那麼越外圈的紅色點，z 值越高；中間藍色點，z 值越低。  
這時候只要用一個水平平面：

```text
z = c
```

就可以把藍色點與紅色點分開。

這是最容易被學生理解的版本。

---

## 二、但要避免說錯：RBF Kernel 不是真的只轉成 3D

這點很重要。

如果你使用的是真正的 **RBF Kernel SVM**，它的 kernel 是：

```text
K(x, x') = exp(-γ ||x - x'||²)
```

RBF Kernel 的概念不是單純把資料轉成 3 維，而是對應到一個非常高維，甚至可視為無限維的特徵空間。因此，在教學動畫中把它畫成 3D，是一種「概念視覺化」，不是嚴格等同於 sklearn 裡 RBF SVM 的完整數學空間。

scikit-learn 的 `SVC` 支援 `linear`、`poly`、`rbf`、`sigmoid`、`precomputed` 等 kernel，其中預設 kernel 是 `rbf`；RBF SVM 主要會受到 `C` 與 `gamma` 兩個參數影響。`C` 控制錯分容忍與邊界簡潔程度，`gamma` 控制單一樣本的影響範圍。([Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?utm_source=chatgpt.com))

所以我會建議你在動畫中這樣表達：

> 「這個 3D 映射是 Kernel Trick 的概念展示。真正的 RBF Kernel 不一定真的把資料轉成可視化的 3D，而是讓模型在高維特徵空間中找到線性分隔面，再對應回原始 2D 空間形成非線性決策邊界。」

這樣會比較嚴謹。

---

## 三、我建議的 Manim 展示流程

你可以做成 5 個畫面階段。

### Scene 1：2D 原始資料

畫出 X-Y 座標軸。

中間放藍色點：

```text
blue class = center cluster
```

外圈放紅色點：

```text
red class = outer ring
```

然後嘗試畫幾條直線，讓觀眾看到：

```text
沒有任何一條直線可以把中間藍點和外圈紅點完整分開
```

這裡可以加文字：

```text
Original 2D Space: Not linearly separable
```

---

### Scene 2：顯示 Kernel 映射公式

畫出公式：

```text
φ(x, y) = (x, y, x² + y²)
```

或 Gaussian 版本：

```text
φ(x, y) = (x, y, exp(-γ(x² + y²)))
```

我比較建議初學展示先用：

```text
z = x² + y²
```

因為它比較直覺：  
離中心越遠，z 越高。

如果你想更貼近 RBF 的語感，可以在旁邊補充：

```text
RBF-like mapping:
z = exp(-γ(x² + y²))
```

但主動畫建議先用 `z = x² + y²`。

---

### Scene 3：把 2D 點「抬升」到 3D

這是整個動畫最關鍵的地方。

原本每個點是：

```text
(x, y)
```

動畫中把它變成：

```text
(x, y, z)
```

其中：

```text
z = x² + y²
```

視覺上會變成：

- 中間藍色點留在低處
- 外圈紅色點被抬到高處
- 資料從原本「平面上包圍」變成「高度上分開」

這一段非常適合用 Manim 的 3D camera rotation，讓鏡頭從俯視慢慢旋轉到 3D 側視。

---

### Scene 4：畫出 3D Hyperplane

在 3D 空間中畫一個水平決策平面：

```text
z = threshold
```

例如：

```text
z = 1.0
```

視覺上：

- 平面下方是藍色類別
- 平面上方是紅色類別

這裡可以標示：

```text
Hyperplane in Feature Space
```

也可以畫出 margin：

```text
z = threshold + margin
z = threshold - margin
```

用兩個半透明平面表示 SVM 的 margin。

---

### Scene 5：投影回 2D，形成非線性決策邊界

最後把 3D 平面投影回 X-Y 平面。

如果：

```text
z = x² + y²
```

而決策平面是：

```text
z = c
```

那麼回到 2D 就會變成：

```text
x² + y² = c
```

也就是一個圓形決策邊界。

這時候觀眾就會理解：

```text
高維空間中的線性分隔面
↓
原始空間中的非線性決策邊界
```

這是 Kernel Trick 最適合教學展示的核心畫面。

---

## 四、建議的技術實作方式

我建議你不要一開始就直接用 sklearn 的 RBF SVM 做 3D 平面，因為真正的 RBF 特徵空間不是單純 3D，畫出來容易造成概念混亂。

比較好的架構是：

```text
Part A：Manim 概念動畫
使用人工定義映射：
z = x² + y²

Part B：sklearn 真實模型
使用：
SVC(kernel="rbf", C=..., gamma=...)

Part C：互動展示
使用 Streamlit + Plotly
讓使用者調整 C、gamma、kernel
```

這樣教學會非常完整。

---

## 五、Manim 實作建議

Manim 端可以使用：

```python
from manim import *
import numpy as np
```

主要使用：

```python
ThreeDScene
ThreeDAxes
Dot3D
Surface
ParametricFunction
set_camera_orientation
begin_ambient_camera_rotation
```

Manim Community 目前有專門的 `ThreeDScene` 類別，適合做 3D 座標、3D 物件與鏡頭動畫。([Manim Community | Documentation](https://docs.manim.community/en/stable/reference/manim.scene.three_d_scene.ThreeDScene.html?utm_source=chatgpt.com))

你的 Manim 專案可以這樣切：

```text
svm_kernel_trick_manim/
│
├── scene_01_2d_data.py
├── scene_02_kernel_mapping.py
├── scene_03_3d_lift.py
├── scene_04_hyperplane.py
├── scene_05_projection.py
└── main.py
```

或簡化成一支：

```text
svm_kernel_trick_3d.py
```

---

## 六、我建議的核心資料設計

可以先用固定資料，不要隨機，這樣動畫穩定。

```python
blue_points = [
    [0.0, 0.0],
    [0.2, 0.1],
    [-0.2, 0.1],
    [0.1, -0.2],
    [-0.1, -0.1],
]

red_points = [
    [1.2, 0.0],
    [-1.2, 0.0],
    [0.0, 1.2],
    [0.0, -1.2],
    [0.9, 0.9],
    [-0.9, 0.9],
    [0.9, -0.9],
    [-0.9, -0.9],
]
```

映射函數：

```python
def lift(point):
    x, y = point
    z = x**2 + y**2
    return np.array([x, y, z])
```

如果要 Gaussian 版本：

```python
def gaussian_lift(point, gamma=1.0):
    x, y = point
    z = np.exp(-gamma * (x**2 + y**2))
    return np.array([x, y, z])
```

但注意 Gaussian 版本會讓中心點比較高，外圈點比較低，因此決策平面的位置與分類方向會反過來。

---

## 七、決策面怎麼畫比較好？

### 教學版建議

使用：

```text
z = x² + y²
```

決策面：

```text
z = 0.7
```

這樣紅色外圈在上方，藍色中心在下方。

Manim 中可以用 `Surface` 畫水平面：

```python
plane = Surface(
    lambda u, v: axes.c2p(u, v, 0.7),
    u_range=[-1.5, 1.5],
    v_range=[-1.5, 1.5],
    fill_opacity=0.35,
)
```

這個平面就是 3D feature space 裡的 hyperplane。

---

## 八、互動展示部分的建議

如果你真的要「互動」，我會建議不要只靠 Manim。

比較好的展示組合是：

```text
Manim：負責講解概念與動畫
Streamlit：負責互動操作
Plotly：負責 3D 旋轉與滑桿
sklearn：負責真實 RBF SVM 模型
```

互動功能可以設計成：

```text
1. 使用者調整 gamma
2. 使用者調整 C
3. 使用者選 kernel：linear / poly / rbf / sigmoid
4. 即時更新 2D 決策邊界
5. 顯示 support vectors
6. 顯示 3D lifted view
```

其中 sklearn 的 SVC 可以用：

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0, gamma=1.0)
model.fit(X, y)
```

然後用 `decision_function` 在網格上計算分數，再畫出決策邊界。scikit-learn 官方也有 RBF SVM 參數範例，說明 `C` 與 `gamma` 對決策邊界的影響。([Scikit-learn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html?utm_source=chatgpt.com))

---

## 九、最推薦的專案展示架構

我會建議你做成這樣：

```text
主題：
Support Vector Machine Kernel Trick 3D Visualization

展示目標：
讓學生理解「原本 2D 無法線性分開的資料，
如何透過 Kernel Trick 的概念，
在高維空間中變得可以用 Hyperplane 分開。」

Phase 1：Manim 概念動畫
- 2D 資料點
- 無法用直線分開
- Kernel 映射公式
- 點從 2D 抬升到 3D
- 畫出 hyperplane
- 投影回 2D 形成圓形決策邊界

Phase 2：sklearn 真實 RBF SVM
- 建立 ring dataset
- 訓練 SVC(kernel="rbf")
- 畫出真實 2D 決策邊界
- 標示 support vectors

Phase 3：Streamlit / Plotly 互動展示
- 調整 C
- 調整 gamma
- 切換 kernel
- 觀察決策邊界變化
- 3D 視角可旋轉
```

---

## 十、我的具體建議

我建議你的 Manim 版本不要一開始就說「RBF Kernel 把資料變成 3D」，而是改成：

> 「我們先用一個可視化的 feature mapping 來理解 Kernel Trick：把 $(x, y)$ 映射成 $(x, y, x^2 + y^2)$。這會讓原本被包圍的資料，在高度方向上分開。真正的 RBF SVM 也是利用類似精神，把資料放到更高維的特徵空間中，使模型能找到線性分隔面。」

這樣既直覺，又不會在數學上說錯。

最適合你的展示版本是：

```text
2D：藍點在中心，紅點在外圈
↓
Mapping：z = x² + y²
↓
3D：紅點被抬高，藍點留低
↓
Hyperplane：水平平面 z = c
↓
回到 2D：圓形決策邊界
↓
再補充：真正 RBF SVM 會形成更彈性的非線性邊界
```

這個設計非常適合 AI / 機器學習課堂展示，也很適合交給 coding agent 實作。

---

**Sources:**

- [ThreeDScene - Manim Community v0.20.1](https://docs.manim.community/en/stable/reference/manim.scene.three_d_scene.ThreeDScene.html?utm_source=chatgpt.com)
- [SVC — scikit-learn 1.9.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?utm_source=chatgpt.com)

## Prompt:
2026/6/18 上午10:14:21

整理成一個完整3個phase的prompt in YAML so that I can tell antigravity agent to implement later

## Response:
2026/6/18 上午10:14:39

以下是一份可直接貼給 **Antigravity Agent / coding agent** 的完整 YAML prompt，分成 3 個 phase：Manim 概念動畫、sklearn 真實 RBF SVM、Streamlit / Plotly 互動展示。

```yaml
project:
  name: "SVM Kernel Trick 3D Visualization"
  language: "Python"
  goal: >
    Build a complete educational demonstration for Support Vector Machine Kernel Trick.
    The project should show how a 2D non-linearly separable dataset with blue points
    in the center and red points outside can become linearly separable after a
    conceptual feature mapping into 3D space. Then demonstrate a real RBF SVM
    decision boundary using sklearn, and finally create an interactive
    Streamlit / Plotly app.

  target_audience: >
    Beginner to intermediate machine learning students who need to understand
    why Kernel Trick helps SVM solve non-linear classification problems.

  important_conceptual_note: >
    The 3D mapping used in the Manim animation is a pedagogical visualization.
    It should not claim that true RBF Kernel literally maps data into only 3D.
    True RBF Kernel corresponds to a very high-dimensional, possibly infinite-dimensional,
    feature space. The 3D animation is used to build intuition.

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
    - "Use clear variable names."
    - "Write comments for educational purposes."
    - "Avoid overly complex abstractions."
    - "Keep each phase runnable independently."
    - "Use deterministic random seeds where randomness is needed."
    - "Prefer readable teaching code over overly optimized code."

  visual_style:
    background: "dark or clean neutral background"
    blue_class: "center points"
    red_class: "outer ring points"
    decision_surface: "semi-transparent plane or surface"
    support_vectors: "highlight clearly"
    annotations: "short, educational, and readable"

phases:
  - phase: 1
    name: "Manim Concept Animation"
    objective: >
      Create a Manim 3D animation explaining the Kernel Trick concept.
      The animation should start with 2D data that cannot be separated by a straight line,
      then lift the data into 3D using a simple feature mapping,
      draw a separating hyperplane, and finally project the result back to 2D
      as a circular non-linear decision boundary.

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
        Use a simple manually defined 2D dataset.
        Blue points should be near the origin.
        Red points should form an outer ring.
        The dataset should be visually obvious: center class versus surrounding class.

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
          This mapping lifts points farther from the origin to higher z values.
          Therefore, red outer-ring points become high in 3D space,
          while blue center points remain low.

      optional_gaussian_mapping:
        formula: "z = exp(-gamma * (x^2 + y^2))"
        explanation: >
          This can be shown as an alternative RBF-like mapping,
          but the main teaching animation should use z = x^2 + y^2
          because it is easier to understand visually.

    animation_sequence:
      - step: "Intro title"
        content:
          title: "Support Vector Machine: Kernel Trick"
          subtitle: "From 2D non-linear separation to 3D linear separation"

      - step: "Show original 2D dataset"
        content:
          - "Draw 2D x-y axes."
          - "Place blue points near the center."
          - "Place red points around the outside."
          - "Add label: Original 2D Space."
          - "Add label: Not linearly separable."

      - step: "Demonstrate failure of linear separation"
        content:
          - "Draw 2 or 3 different straight lines."
          - "Show that no straight line can cleanly separate center blue points from outer red points."
          - "Fade out failed lines."

      - step: "Introduce feature mapping"
        content:
          - "Show formula: phi(x, y) = (x, y, x^2 + y^2)."
          - "Explain visually that the new z value represents distance from the origin."
          - "Add annotation: Points farther from center get lifted higher."

      - step: "Lift points into 3D"
        content:
          - "Transform the 2D axes into 3D axes."
          - "Move each point from (x, y, 0) to (x, y, x^2 + y^2)."
          - "Use smooth animation so viewers see the lifting process."
          - "Rotate camera to show depth."
          - "Blue points should remain low."
          - "Red points should move higher."

      - step: "Draw separating hyperplane"
        content:
          - "Draw a semi-transparent horizontal plane z = c."
          - "Use c around 0.7 or another value that separates blue and red points."
          - "Label it: Hyperplane in Feature Space."
          - "Optionally draw two lighter parallel margin planes."

      - step: "Project back to 2D"
        content:
          - "Show that z = c and z = x^2 + y^2 imply x^2 + y^2 = c."
          - "Draw the circular decision boundary on the original 2D plane."
          - "Label it: Non-linear decision boundary in original space."

      - step: "Closing message"
        content:
          - "Show summary: Linear separator in feature space becomes non-linear boundary in original space."
          - "Add note: Real RBF SVM uses a higher-dimensional feature space, not merely 3D."

    visual_requirements:
      camera:
        - "Start with a 2D top-down view."
        - "Transition smoothly to 3D perspective."
        - "Use slow ambient rotation during the 3D hyperplane section."
        - "Return to top-down view for final 2D circular boundary."

      objects:
        - "Blue points must be visually distinct from red points."
        - "The hyperplane should be semi-transparent."
        - "The circular boundary should be clearly visible."
        - "All formulas should be readable."

    output:
      render_command: "manim -pqh phase1_manim/svm_kernel_trick_3d.py SVMKernelTrick3D"
      expected_output_file: "outputs/svm_kernel_trick_3d.mp4"

    acceptance_criteria:
      - "The animation clearly shows that the original 2D data is not linearly separable."
      - "The mapping phi(x, y) = (x, y, x^2 + y^2) is displayed."
      - "Points are visibly lifted into 3D."
      - "A 3D separating hyperplane is shown."
      - "The projected 2D decision boundary is circular."
      - "The final explanation does not incorrectly claim that RBF Kernel is only 3D."

  - phase: 2
    name: "sklearn RBF SVM Real Decision Boundary"
    objective: >
      Implement a real SVM classifier using sklearn with an RBF kernel.
      This phase should show the actual 2D non-linear decision boundary learned by SVC.
      It should also highlight support vectors and compare linear SVM versus RBF SVM.

    files_to_create:
      - "phase2_sklearn/dataset_utils.py"
      - "phase2_sklearn/rbf_svm_decision_boundary.py"

    dataset:
      method: "Use sklearn.datasets.make_circles or manually generated ring data."
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
      - "Create a dataset utility function that returns X and y."
      - "Train a linear SVM model."
      - "Train an RBF SVM model."
      - "Create a mesh grid over the 2D input space."
      - "Use model.decision_function on the mesh grid."
      - "Plot decision boundary where decision_function equals 0."
      - "Plot margins where decision_function equals +1 and -1."
      - "Highlight support vectors using larger circles."
      - "Create one figure for Linear SVM."
      - "Create one figure for RBF SVM."
      - "Save figures to outputs."

    plots:
      linear_svm_plot:
        filename: "outputs/linear_svm_decision_boundary.png"
        title: "Linear SVM: Cannot Separate Circular Data Well"
        required_elements:
          - "Blue inner class points"
          - "Red outer class points"
          - "Linear decision boundary"
          - "Support vectors"

      rbf_svm_plot:
        filename: "outputs/rbf_svm_decision_boundary.png"
        title: "RBF SVM: Non-linear Decision Boundary"
        required_elements:
          - "Blue inner class points"
          - "Red outer class points"
          - "Circular or curved decision boundary"
          - "Support vectors"
          - "Margin curves"

    educational_annotations:
      include_text:
        - "Linear SVM uses a straight decision boundary."
        - "RBF SVM creates a non-linear decision boundary in the original 2D space."
        - "Support vectors are the key samples that define the boundary."
        - "C controls the penalty for misclassification."
        - "gamma controls the influence range of each training point."

    run_command: "python phase2_sklearn/rbf_svm_decision_boundary.py"

    acceptance_criteria:
      - "The script runs without error."
      - "The dataset is circular and non-linearly separable."
      - "Linear SVM performs poorly or visibly cannot separate the data well."
      - "RBF SVM creates a curved decision boundary."
      - "Support vectors are highlighted."
      - "Output images are saved into the outputs folder."

  - phase: 3
    name: "Streamlit and Plotly Interactive Demo"
    objective: >
      Build an interactive web app that allows users to adjust SVM parameters
      and observe changes in the decision boundary. The app should include
      both a 2D decision boundary view and a 3D conceptual feature mapping view.

    files_to_create:
      - "phase3_streamlit/app.py"
      - "phase3_streamlit/model_utils.py"
      - "phase3_streamlit/plot_utils.py"

    framework:
      app: "Streamlit"
      visualization: "Plotly"
      machine_learning: "scikit-learn"

    app_title: "Interactive SVM Kernel Trick Visualization"

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
        note: "Only used for rbf, poly, and sigmoid kernels."

      - name: "degree"
        type: "slider"
        min: 2
        max: 6
        default: 3
        note: "Only used for polynomial kernel."

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
      - section: "Concept explanation"
        content:
          - "Explain that blue points are the inner class and red points are the outer class."
          - "Explain that linear SVM struggles with circular data."
          - "Explain that kernel methods help create non-linear boundaries."
          - "Clarify that the 3D view is conceptual, while sklearn RBF SVM uses an implicit higher-dimensional feature space."

      - section: "2D decision boundary"
        visualization:
          type: "Plotly contour plot"
          required_elements:
            - "Scatter plot of data points"
            - "Decision boundary where decision_function = 0"
            - "Optional margin lines where decision_function = +1 and -1"
            - "Support vectors highlighted"
            - "Current kernel, C, gamma displayed"

      - section: "3D conceptual mapping"
        visualization:
          type: "Plotly 3D scatter and surface"
          mapping_formula: "z = x^2 + y^2"
          required_elements:
            - "3D scatter points"
            - "Blue inner points at lower z"
            - "Red outer points at higher z"
            - "Semi-transparent horizontal decision plane"
            - "Rotatable 3D view"

      - section: "Parameter interpretation"
        content:
          - "Explain C: larger C tries harder to classify training points correctly, possibly creating a more complex boundary."
          - "Explain gamma: larger gamma makes each point influence a smaller region, often creating more flexible or overfitted boundaries."
          - "Explain kernel choice: linear creates straight boundary, rbf creates flexible non-linear boundary, poly creates polynomial boundary."

    implementation_tasks:
      - "Create a reusable function to generate circular data."
      - "Create a reusable function to train SVC with selected kernel and parameters."
      - "Create a reusable function to compute decision boundary mesh."
      - "Create a Plotly 2D contour figure for the decision boundary."
      - "Create a Plotly 3D figure for the conceptual mapping z = x^2 + y^2."
      - "Highlight support vectors when enabled."
      - "Add explanatory text in the Streamlit interface."
      - "Keep layout clean and suitable for classroom demonstration."

    model_behavior:
      sklearn_model: "SVC"
      required_methods:
        - "fit"
        - "decision_function"
      support_vector_attribute: "support_vectors_"

    plotly_2d_requirements:
      - "Use contour plot for decision score."
      - "Use scatter markers for data points."
      - "Use different colors for classes."
      - "Use larger marker outlines for support vectors."
      - "Add title and axis labels."

    plotly_3d_requirements:
      - "Use Scatter3d for lifted points."
      - "Use Surface for the decision plane."
      - "Allow the user to rotate the 3D view."
      - "Use clear axis labels: x, y, z = x^2 + y^2."
      - "Keep the plane semi-transparent."

    run_command: "streamlit run phase3_streamlit/app.py"

    acceptance_criteria:
      - "The Streamlit app launches successfully."
      - "Users can change kernel, C, gamma, degree, noise, and sample size."
      - "The 2D decision boundary updates after parameter changes."
      - "Support vectors can be shown or hidden."
      - "The 3D conceptual mapping is rotatable."
      - "The app clearly explains that the 3D mapping is conceptual."
      - "The app is suitable for live classroom demonstration."

README:
  required_content:
    - "Project overview"
    - "Machine learning concept explanation"
    - "Difference between conceptual 3D mapping and true RBF Kernel"
    - "How to install dependencies"
    - "How to run Phase 1 Manim animation"
    - "How to run Phase 2 sklearn demo"
    - "How to run Phase 3 Streamlit app"
    - "Expected outputs"
    - "Teaching notes"

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
  - "A Manim animation showing SVM Kernel Trick concept in 3D."
  - "A sklearn script showing real RBF SVM decision boundary."
  - "A Streamlit / Plotly interactive app."
  - "A README explaining the concept and how to run each phase."
  - "Saved output images and video in the outputs folder."

quality_checklist:
  concept_accuracy:
    - "Do not say RBF Kernel literally maps data to only 3D."
    - "Explain that the Manim 3D mapping is for intuition."
    - "Correctly distinguish hyperplane in feature space from non-linear boundary in original space."

  code_quality:
    - "All scripts should run independently."
    - "Functions should be reusable."
    - "Use comments to explain important teaching logic."
    - "Avoid hard-coded paths that break across operating systems."

  visual_quality:
    - "Points must be easy to distinguish."
    - "Decision boundaries must be clear."
    - "3D scene must be readable."
    - "Text annotations should not overlap important visual elements."

  classroom_usability:
    - "The demonstration should work without requiring advanced math knowledge."
    - "The sequence should clearly show before, transformation, after."
    - "The interactive app should make C and gamma effects easy to observe."
```

## Prompt:
2026/6/18 上午10:15:29

改成中文

## Response:
2026/6/18 上午10:15:43

以下是改成**繁體中文**後的完整 3 階段 YAML Prompt，可直接交給 **Antigravity Agent / coding agent** 實作。

```yaml
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
```

## Prompt:
2026/6/18 下午12:00:34

將附檔的內容 產生資訊圖表, 生圖:excalidraw style A4 直式 2K

[Attachment: README(1).md]

## Response:
2026/6/18 下午12:00:47

![SVM Kernel Trick 3D 視覺化展示](https://chatgpt.com/backend-api/estuary/content?id=file_0000000085587207b2973011afd4f2cf&ts=494932&p=fs&cid=1&sig=edea926bc433f2fe640d12e8c4d654333f41aa844d2d2bed4ab9c6310e5ded31&v=0)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)