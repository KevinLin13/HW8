"""Interactive classroom demonstration of the SVM kernel trick."""

from pathlib import Path

import plotly.io as pio
import streamlit as st

try:
    from phase3_streamlit.model_utils import create_circle_dataset, train_svm
    from phase3_streamlit.plot_utils import create_2d_figure, create_3d_mapping_figure
except ModuleNotFoundError:
    from model_utils import create_circle_dataset, train_svm
    from plot_utils import create_2d_figure, create_3d_mapping_figure


def create_auto_rotating_html(figure, animate_lift: bool) -> str:
    """Embed a Plotly figure with slow, continuous camera rotation."""
    html = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        auto_play=False,
        div_id="svm-3d-plot",
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    lift_script = """
        const liftFrames = graph._transitionData._frames
            .map(frame => frame.name)
            .filter(name => name.startsWith("lift-"));
        if (liftFrames.length) {
            Plotly.animate(graph, liftFrames, {
                frame: {duration: 70, redraw: true},
                transition: {duration: 70, easing: "cubic-in-out"},
                mode: "immediate"
            });
        }
    """ if animate_lift else ""
    rotation_script = f"""
    <script>
    (() => {{
        const graph = document.getElementById("svm-3d-plot");
        if (!graph) return;
        {lift_script}

        let angle = Math.PI / 4;
        let paused = false;
        let resumeTimer = null;
        let previousTime = performance.now();
        let previousRender = 0;
        const radius = 3.18;
        const height = 1.55;
        const radiansPerMs = (2 * Math.PI) / 55000;

        const pauseRotation = () => {{
            paused = true;
            if (resumeTimer) clearTimeout(resumeTimer);
        }};
        const resumeRotation = () => {{
            if (resumeTimer) clearTimeout(resumeTimer);
            resumeTimer = setTimeout(() => {{
                const eye = graph.layout.scene?.camera?.eye;
                if (eye) angle = Math.atan2(eye.y, eye.x);
                previousTime = performance.now();
                paused = false;
            }}, 1400);
        }};

        graph.addEventListener("pointerdown", pauseRotation);
        window.addEventListener("pointerup", resumeRotation);
        graph.addEventListener("wheel", () => {{
            pauseRotation();
            resumeRotation();
        }}, {{passive: true}});

        const rotate = (now) => {{
            const elapsed = Math.min(now - previousTime, 100);
            previousTime = now;
            if (!paused && !document.hidden && now - previousRender >= 33) {{
                angle += elapsed * radiansPerMs;
                Plotly.relayout(graph, {{
                    "scene.camera.eye.x": radius * Math.cos(angle),
                    "scene.camera.eye.y": radius * Math.sin(angle),
                    "scene.camera.eye.z": height
                }});
                previousRender = now;
            }}
            requestAnimationFrame(rotate);
        }};
        requestAnimationFrame(rotate);
    }})();
    </script>
    """
    return html + rotation_script


st.set_page_config(
    page_title="Interactive SVM Kernel Trick",
    page_icon="🧭",
    layout="wide",
)

st.title("互動式 SVM Kernel Trick 視覺化展示")
st.markdown(
    """
藍色點代表內圈類別，紅色點代表外圈類別。Linear SVM 只能建立直線邊界，
而 kernel method 能讓 SVM 在原始空間中形成非線性決策邊界。
"""
)

video_path = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "svm_kernel_trick_3d.mp4"
)
with st.expander("▶ SVM Kernel Trick 3D 教學動畫", expanded=True):
    if video_path.exists():
        st.video(
            str(video_path),
            format="video/mp4",
            autoplay=True,
            muted=True,
        )
        st.caption("1080p60 · 自動靜音播放，可在播放器中開啟聲音")
    else:
        st.info("目前找不到動畫檔案，請確認 outputs/svm_kernel_trick_3d.mp4 已存在。")

with st.sidebar:
    st.header("模型與資料參數")
    kernel = st.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"), index=2)
    C = st.select_slider(
        "C",
        options=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        value=1.0,
    )
    gamma = st.select_slider(
        "Gamma",
        options=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        value=1.0,
        disabled=kernel == "linear",
        help="適用於 RBF、poly 與 sigmoid kernel。",
    )
    degree = st.slider(
        "Polynomial degree",
        min_value=2,
        max_value=6,
        value=3,
        disabled=kernel != "poly",
    )
    noise = st.slider("Noise", 0.0, 0.3, 0.06, 0.01)
    sample_size = st.slider("Sample size", 50, 1000, 300, 50)
    show_support_vectors = st.checkbox("顯示 support vectors", value=True)
    show_3d_mapping = st.checkbox("顯示 3D 概念映射", value=True)
    animate_3d_lift = st.checkbox(
        "啟用 3D 抬升動畫",
        value=True,
        disabled=not show_3d_mapping,
    )
    auto_rotate_3d = st.checkbox(
        "載入後自動旋轉 3D 視角",
        value=True,
        disabled=not show_3d_mapping,
    )
    display_3d_points = st.slider(
        "3D 顯示球體數量",
        min_value=30,
        max_value=140,
        value=80,
        step=10,
        disabled=not show_3d_mapping,
        help="只影響 3D 教學圖的顯示密度，不影響模型訓練。",
    )

with st.spinner("正在更新 SVM 決策邊界…"):
    X, y = create_circle_dataset(sample_size, noise)
    model = train_svm(X, y, kernel, C, gamma, degree)

st.subheader("2D 決策邊界")
metric_kernel, metric_support, metric_samples = st.columns(3)
metric_kernel.metric("Kernel", kernel.upper())
metric_support.metric("Support vectors", len(model.support_vectors_))
metric_samples.metric("Samples", sample_size)
st.caption(f"C={C}　Gamma={gamma}　Degree={degree}　Noise={noise:.2f}")
st.plotly_chart(
    create_2d_figure(model, X, y, show_support_vectors),
    key="svm-decision-boundary",
    width="stretch",
    config={"displaylogo": False, "scrollZoom": True},
)

if show_3d_mapping:
    st.subheader("3D 概念映射")
    st.warning(
        "這張圖使用 z = x² + y² 建立直覺。真正的 RBF SVM 使用隱含的高維、"
        "甚至可視為無限維的特徵空間，並不是單純映射到三維。"
    )
    if animate_3d_lift:
        st.caption("資料會從 z=0 抬升到 z=x²+y²，鏡頭則持續緩慢環繞展示 3D 結構。")
    mapping_figure = create_3d_mapping_figure(
        X,
        y,
        animate_3d_lift,
        display_3d_points,
    )
    if auto_rotate_3d:
        animation_html = create_auto_rotating_html(
            mapping_figure,
            animate_3d_lift,
        )
        st.iframe(animation_html, width="stretch", height=850)
    else:
        st.plotly_chart(
            mapping_figure,
            key="svm-3d-mapping",
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

st.subheader("參數如何影響模型？")
left, middle, right = st.columns(3)
left.markdown(
    "**C**\n\n數值越大，模型越努力正確分類訓練資料，但邊界可能變得更複雜。"
)
middle.markdown(
    "**Gamma**\n\n數值越大，單一樣本的影響範圍越小，邊界更有彈性，也更可能過度擬合。"
)
right.markdown(
    "**Kernel**\n\nLinear 產生線性邊界；RBF 與 poly 可以產生非線性邊界。"
)
