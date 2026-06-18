"""Plotly figure builders for the interactive SVM demonstration."""

import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC

try:
    from phase3_streamlit.model_utils import calculate_decision_grid
except ModuleNotFoundError:
    from model_utils import calculate_decision_grid


CLASS_COLORS = {0: "#2f80ed", 1: "#eb5757"}


def _select_display_points(
    X: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a balanced, deterministic subset for a readable 3D scene."""
    selected_indices = []
    per_class = max(1, max_points // 2)
    for label in (0, 1):
        class_indices = np.flatnonzero(y == label)
        count = min(per_class, len(class_indices))
        positions = np.linspace(0, len(class_indices) - 1, count, dtype=int)
        selected_indices.extend(class_indices[positions])
    selected_indices = np.array(selected_indices)
    return X[selected_indices], y[selected_indices]


def _create_sphere_mesh(
    centers_xy: np.ndarray,
    center_z: np.ndarray,
    radius: float,
    latitude_steps: int = 7,
    longitude_steps: int = 10,
):
    """Combine multiple low-poly spheres into one efficient mesh."""
    vertices_x: list[float] = []
    vertices_y: list[float] = []
    vertices_z: list[float] = []
    triangles_i: list[int] = []
    triangles_j: list[int] = []
    triangles_k: list[int] = []

    theta_values = np.linspace(0, np.pi, latitude_steps)
    phi_values = np.linspace(0, 2 * np.pi, longitude_steps, endpoint=False)
    vertices_per_sphere = latitude_steps * longitude_steps

    for sphere_index, ((center_x, center_y), z_value) in enumerate(
        zip(centers_xy, center_z)
    ):
        offset = sphere_index * vertices_per_sphere
        for theta in theta_values:
            sin_theta = np.sin(theta)
            for phi in phi_values:
                vertices_x.append(center_x + radius * sin_theta * np.cos(phi))
                vertices_y.append(center_y + radius * sin_theta * np.sin(phi))
                vertices_z.append(z_value + radius * np.cos(theta))

        for latitude in range(latitude_steps - 1):
            for longitude in range(longitude_steps):
                next_longitude = (longitude + 1) % longitude_steps
                current = offset + latitude * longitude_steps + longitude
                current_next = offset + latitude * longitude_steps + next_longitude
                upper = offset + (latitude + 1) * longitude_steps + longitude
                upper_next = offset + (latitude + 1) * longitude_steps + next_longitude
                triangles_i.extend((current, current_next))
                triangles_j.extend((upper, upper_next))
                triangles_k.extend((current_next, upper))

    return {
        "x": vertices_x,
        "y": vertices_y,
        "z": vertices_z,
        "i": triangles_i,
        "j": triangles_j,
        "k": triangles_k,
    }


def _add_contour_line(
    figure: go.Figure,
    xx: np.ndarray,
    yy: np.ndarray,
    scores: np.ndarray,
    level: float,
    name: str,
    color: str,
    width: int,
    dash: str | None = None,
) -> None:
    """Add one decision-function contour level."""
    figure.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=scores,
            contours={
                "start": level,
                "end": level,
                "size": 1,
                "coloring": "lines",
                "showlabels": False,
            },
            line={"color": color, "width": width, "dash": dash},
            showscale=False,
            name=name,
            hoverinfo="skip",
        )
    )


def create_2d_figure(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    show_support_vectors: bool,
) -> go.Figure:
    """Build the interactive 2D decision-boundary figure."""
    xx, yy, scores = calculate_decision_grid(model, X)
    figure = go.Figure()

    figure.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=scores,
            contours={"coloring": "heatmap", "showlines": False},
            colorscale=[[0, "#d9eaff"], [0.5, "#ffffff"], [1, "#ffdede"]],
            opacity=0.55,
            showscale=False,
            hoverinfo="skip",
            name="Decision score",
        )
    )
    _add_contour_line(figure, xx, yy, scores, -1, "Margin -1", "#777777", 1, "dash")
    _add_contour_line(figure, xx, yy, scores, 0, "Decision boundary", "#222222", 3)
    _add_contour_line(figure, xx, yy, scores, 1, "Margin +1", "#777777", 1, "dash")

    for label, name in ((0, "Inner class"), (1, "Outer class")):
        points = X[y == label]
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                name=name,
                marker={
                    "color": CLASS_COLORS[label],
                    "size": 8,
                    "line": {"color": "white", "width": 0.7},
                },
            )
        )

    if show_support_vectors:
        support_vectors = model.support_vectors_
        figure.add_trace(
            go.Scatter(
                x=support_vectors[:, 0],
                y=support_vectors[:, 1],
                mode="markers",
                name="Support vectors",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 15,
                    "line": {"color": "#f2c94c", "width": 2},
                },
            )
        )

    figure.update_layout(
        title="2D decision boundary and margins",
        xaxis_title="Feature x1",
        yaxis_title="Feature x2",
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "y": 1.08},
        transition={"duration": 350, "easing": "cubic-in-out"},
        uirevision="svm-2d-view",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def create_3d_mapping_figure(
    X: np.ndarray,
    y: np.ndarray,
    animate_lift: bool = True,
    max_display_points: int = 80,
) -> go.Figure:
    """Show the conceptual mapping z = x² + y² and a separating plane."""
    display_X, display_y = _select_display_points(X, y, max_display_points)
    z_values = display_X[:, 0] ** 2 + display_X[:, 1] ** 2
    plane_height = 0.55
    xy_limit = max(2.35, float(np.abs(X).max()) + 0.9)
    z_min = -1.25
    z_max = max(2.8, float(z_values.max()) + 0.65)
    surface_limit = min(1.9, xy_limit)
    surface_axis = np.linspace(-surface_limit, surface_limit, 48)
    surface_x, surface_y = np.meshgrid(surface_axis, surface_axis)
    surface_z = np.full_like(surface_x, plane_height)
    mapping_z = surface_x**2 + surface_y**2
    initial_progress = 0.0 if animate_lift else 1.0
    figure = go.Figure()
    sphere_meshes = {}
    for label, name in ((0, "Inner class"), (1, "Outer class")):
        points = display_X[display_y == label]
        point_z = z_values[display_y == label]
        lifted_z = point_z * initial_progress
        sphere_meshes[label] = _create_sphere_mesh(
            points,
            lifted_z,
            radius=0.085,
        )
        figure.add_trace(
            go.Mesh3d(
                **sphere_meshes[label],
                name=name,
                color=CLASS_COLORS[label],
                opacity=0.98,
                flatshading=False,
                lighting={
                    "ambient": 0.38,
                    "diffuse": 0.82,
                    "specular": 0.85,
                    "roughness": 0.28,
                    "fresnel": 0.18,
                },
                lightposition={"x": 100, "y": 180, "z": 260},
                hoverinfo="skip",
            )
        )

    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            opacity=0.28,
            colorscale=[[0, "#38bdf8"], [0.5, "#a78bfa"], [1, "#f472b6"]],
            showscale=False,
            name="Conceptual hyperplane",
            lighting={
                "ambient": 0.6,
                "diffuse": 0.75,
                "specular": 0.65,
                "roughness": 0.3,
                "fresnel": 0.25,
            },
            lightposition={"x": 100, "y": 150, "z": 250},
            hovertemplate="z = 0.55<extra>Conceptual hyperplane</extra>",
        )
    )

    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=mapping_z,
            opacity=0.2,
            colorscale=[
                [0, "#0ea5e9"],
                [0.35, "#6366f1"],
                [0.68, "#a855f7"],
                [1, "#f43f5e"],
            ],
            showscale=False,
            name="Mapping surface",
            contours={
                "z": {
                    "show": True,
                    "start": 0.2,
                    "end": 2.0,
                    "size": 0.25,
                    "color": "rgba(255,255,255,0.18)",
                    "width": 1,
                }
            },
            lighting={
                "ambient": 0.55,
                "diffuse": 0.9,
                "specular": 0.55,
                "roughness": 0.45,
                "fresnel": 0.2,
            },
            lightposition={"x": 100, "y": 150, "z": 250},
            hovertemplate="z = x² + y²<extra>Mapping surface</extra>",
        )
    )

    boundary_angles = np.linspace(0, 2 * np.pi, 180)
    boundary_radius = np.sqrt(plane_height)
    figure.add_trace(
        go.Scatter3d(
            x=boundary_radius * np.cos(boundary_angles),
            y=boundary_radius * np.sin(boundary_angles),
            z=np.full_like(boundary_angles, plane_height + 0.012),
            mode="lines",
            name="Plane intersection",
            line={"color": "#f8fafc", "width": 6},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=boundary_radius * np.cos(boundary_angles),
            y=boundary_radius * np.sin(boundary_angles),
            z=np.full_like(boundary_angles, 0.012),
            mode="lines",
            name="2D boundary projection",
            line={"color": "#facc15", "width": 4, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    label_x = min(xy_limit - 0.25, 1.72)
    label_y = min(xy_limit - 0.25, 1.72)
    figure.add_trace(
        go.Scatter3d(
            x=[boundary_radius * 0.72, label_x],
            y=[boundary_radius * 0.72, label_y],
            z=[plane_height + 0.025, plane_height + 0.28],
            mode="lines",
            line={"color": "#f8fafc", "width": 4},
            name="Hyperplane guide",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[label_x],
            y=[label_y],
            z=[plane_height + 0.33],
            mode="markers+text",
            marker={
                "size": 5,
                "color": "#f8fafc",
                "line": {"color": "#38bdf8", "width": 2},
            },
            text=["Decision hyperplane<br>z = 0.55"],
            textposition="top center",
            textfont={"size": 16, "color": "#ffffff"},
            name="Decision hyperplane label",
            showlegend=False,
            hovertemplate="Decision hyperplane<br>z = 0.55<extra></extra>",
        )
    )

    axis_specs = (
        (
            [-xy_limit, xy_limit],
            [0, 0],
            [0, 0],
            "x",
            [None, "x"],
        ),
        (
            [0, 0],
            [-xy_limit, xy_limit],
            [0, 0],
            "y",
            [None, "y"],
        ),
        (
            [0, 0],
            [0, 0],
            [z_min, z_max],
            "z",
            [None, "z"],
        ),
    )
    for axis_x, axis_y, axis_z, name, labels in axis_specs:
        figure.add_trace(
            go.Scatter3d(
                x=axis_x,
                y=axis_y,
                z=axis_z,
                mode="lines+text",
                text=labels,
                textposition="top center",
                textfont={"size": 18, "color": "#ffffff"},
                line={"color": "#f8fafc", "width": 7},
                name=f"{name}-axis",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if animate_lift:
        progress_values = np.linspace(0, 1, 21)
        figure.frames = [
            go.Frame(
                name=f"lift-{progress:.2f}",
                traces=[0, 1],
                data=[
                    go.Mesh3d(
                        z=_create_sphere_mesh(
                            display_X[display_y == label],
                            z_values[display_y == label] * progress,
                            radius=0.085,
                        )["z"]
                    )
                    for label in (0, 1)
                ],
            )
            for progress in progress_values
        ]

    figure.update_layout(
        title={
            "text": "3D Feature Space · z = x² + y²",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 26, "color": "#f8fafc"},
        },
        template="plotly_dark",
        paper_bgcolor="#070b18",
        plot_bgcolor="#070b18",
        font={"color": "#eef2ff"},
        height=820,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        uirevision="svm-3d-camera",
        scene={
            "bgcolor": "#070b18",
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z = x² + y²",
            "xaxis": {
                "range": [-xy_limit, xy_limit],
                "backgroundcolor": "rgba(15,23,42,0.35)",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showspikes": False,
                "showbackground": False,
                "tickfont": {"size": 13},
            },
            "yaxis": {
                "range": [-xy_limit, xy_limit],
                "backgroundcolor": "rgba(15,23,42,0.35)",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showspikes": False,
                "showbackground": False,
                "tickfont": {"size": 13},
            },
            "zaxis": {
                "range": [z_min, z_max],
                "backgroundcolor": "rgba(15,23,42,0.35)",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showspikes": False,
                "showbackground": False,
                "tickfont": {"size": 13},
            },
            "aspectratio": {"x": 1.25, "y": 1.25, "z": 1.0},
            "camera": {
                "eye": {"x": 2.25, "y": 2.25, "z": 1.55},
                "center": {"x": 0, "y": 0, "z": 0},
                "projection": {"type": "orthographic"},
            },
        },
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0.5,
            "xanchor": "center",
            "bgcolor": "rgba(7,11,24,0.78)",
            "font": {"size": 14},
        },
    )

    buttons = []
    if animate_lift:
        lift_frame_names = [
            f"lift-{progress:.2f}" for progress in np.linspace(0, 1, 21)
        ]
        buttons.append(
            {
                "label": "↟ 播放抬升",
                "method": "animate",
                "args": [
                    lift_frame_names,
                    {
                        "frame": {"duration": 180, "redraw": True},
                        "transition": {
                            "duration": 180,
                            "easing": "cubic-in-out",
                        },
                        "mode": "immediate",
                    },
                ],
            }
        )

    if buttons:
        buttons.append(
            {
                "label": "⏸ 暫停",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    },
                ],
            }
        )
        figure.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.02,
                    "y": 0.02,
                    "showactive": False,
                    "buttons": buttons,
                }
            ],
        )

    if animate_lift:
        figure.update_layout(
            sliders=[
                {
                    "active": 0,
                    "x": 0.32,
                    "y": 0.02,
                    "len": 0.64,
                    "currentvalue": {"prefix": "抬升進度：", "suffix": "%"},
                    "steps": [
                        {
                            "label": str(round(progress * 100)),
                            "method": "animate",
                            "args": [
                                [f"lift-{progress:.2f}"],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                        for progress in np.linspace(0, 1, 21)
                    ],
                }
            ]
        )
    return figure
