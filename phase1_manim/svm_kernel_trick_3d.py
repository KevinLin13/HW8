"""Manim animation explaining a conceptual SVM feature mapping."""

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN,
    GREY_B,
    LEFT,
    Line,
    ParametricFunction,
    RED,
    RIGHT,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    TAU,
    UP,
    VGroup,
    WHITE,
    Write,
    YELLOW,
    Axes,
    Create,
    Dot,
    Dot3D,
)


BLUE_POINTS = np.array(
    [
        [0.0, 0.0],
        [0.2, 0.1],
        [-0.2, 0.1],
        [0.1, -0.2],
        [-0.1, -0.1],
        [0.25, -0.15],
        [-0.25, 0.0],
    ]
)
RED_POINTS = np.array(
    [
        [1.2, 0.0],
        [-1.2, 0.0],
        [0.0, 1.2],
        [0.0, -1.2],
        [0.9, 0.9],
        [-0.9, 0.9],
        [0.9, -0.9],
        [-0.9, -0.9],
        [1.3, 0.4],
        [-1.3, -0.4],
    ]
)


class SVMKernelTrick3D(ThreeDScene):
    """Visualize a non-linear 2D problem becoming separable after lifting."""

    def construct(self):
        self.show_title()
        axes_2d, points_2d = self.show_original_space()
        self.show_failed_linear_boundaries(axes_2d)
        self.show_mapping_formula()
        self.play(FadeOut(axes_2d), FadeOut(points_2d))
        axes_3d, mapped_points, plane = self.show_feature_space()
        self.show_projection_back(axes_3d, mapped_points, plane)
        self.show_summary()

    def show_title(self):
        title = Text("Support Vector Machine: Kernel Trick", font_size=42)
        subtitle = Text(
            "From non-linear separation in 2D to linear separation in 3D",
            font_size=24,
            color=GREY_B,
        ).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle, shift=UP))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_original_space(self):
        axes = Axes(
            x_range=[-1.6, 1.6, 0.5],
            y_range=[-1.6, 1.6, 0.5],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY_B},
        )
        blue_dots = VGroup(
            *[Dot(axes.c2p(x, y), color=BLUE, radius=0.075) for x, y in BLUE_POINTS]
        )
        red_dots = VGroup(
            *[Dot(axes.c2p(x, y), color=RED, radius=0.075) for x, y in RED_POINTS]
        )
        points = VGroup(blue_dots, red_dots)
        label = Text("Original 2D Space", font_size=28).to_corner(UP + LEFT)
        note = Text("Not linearly separable", font_size=25, color=YELLOW).to_corner(
            UP + RIGHT
        )

        self.play(Create(axes), FadeIn(points, lag_ratio=0.08), FadeIn(label))
        self.play(Write(note))
        self.wait(1)
        self.play(FadeOut(label), FadeOut(note))
        return axes, points

    def show_failed_linear_boundaries(self, axes):
        candidates = VGroup(
            Line(axes.c2p(-1.5, -0.2), axes.c2p(1.5, 0.2), color=YELLOW),
            Line(axes.c2p(-0.5, -1.5), axes.c2p(0.5, 1.5), color=YELLOW),
            Line(axes.c2p(-1.4, 1.0), axes.c2p(1.4, -1.0), color=YELLOW),
        )
        message = Text("No straight line separates both classes", font_size=24).to_edge(
            DOWN
        )
        for line in candidates:
            self.play(Create(line), run_time=0.5)
            self.wait(0.25)
            self.play(FadeOut(line), run_time=0.35)
        self.play(Write(message))
        self.wait(0.8)
        self.play(FadeOut(message))

    def show_mapping_formula(self):
        formula = Text(
            "φ(x, y) = (x, y, x² + y²)",
            font_size=38,
            color=WHITE,
        )
        explanation = Text(
            "Farther from the center means a greater height",
            font_size=24,
            color=YELLOW,
        ).next_to(formula, DOWN)
        self.play(Write(formula), FadeIn(explanation, shift=UP))
        self.wait(1.4)
        self.play(FadeOut(formula), FadeOut(explanation))

    def show_feature_space(self):
        axes = ThreeDAxes(
            x_range=[-1.6, 1.6, 0.5],
            y_range=[-1.6, 1.6, 0.5],
            z_range=[0, 2.2, 0.5],
            x_length=6,
            y_length=6,
            z_length=4.2,
        )
        all_points = [(point, BLUE) for point in BLUE_POINTS] + [
            (point, RED) for point in RED_POINTS
        ]
        flat_points = VGroup(
            *[
                Dot3D(axes.c2p(x, y, 0), color=color, radius=0.07)
                for (x, y), color in all_points
            ]
        )
        lifted_points = VGroup(
            *[
                Dot3D(axes.c2p(x, y, x**2 + y**2), color=color, radius=0.07)
                for (x, y), color in all_points
            ]
        )
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-1.35, 1.35],
            v_range=[-1.35, 1.35],
            resolution=(24, 24),
            fill_opacity=0.12,
            checkerboard_colors=[GREEN, GREEN],
            stroke_opacity=0.15,
        )

        self.set_camera_orientation(phi=68 * DEGREES, theta=-52 * DEGREES)
        self.play(Create(axes), FadeIn(flat_points))
        self.play(FadeIn(paraboloid), Transform(flat_points, lifted_points), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(1.2)

        plane_height = 0.7
        plane = Surface(
            lambda u, v: axes.c2p(u, v, plane_height),
            u_range=[-1.45, 1.45],
            v_range=[-1.45, 1.45],
            resolution=(2, 2),
            fill_opacity=0.38,
            checkerboard_colors=[YELLOW, YELLOW],
            stroke_color=YELLOW,
        )
        plane_label = Text(
            "Hyperplane in Feature Space", font_size=24, color=YELLOW
        ).to_corner(UP + RIGHT)
        self.add_fixed_in_frame_mobjects(plane_label)
        self.play(FadeIn(plane), FadeIn(plane_label))
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(paraboloid), FadeOut(plane_label))
        return axes, flat_points, plane

    def show_projection_back(self, axes_3d, mapped_points, plane):
        self.play(FadeOut(axes_3d), FadeOut(mapped_points), FadeOut(plane))
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=1.2)

        axes = Axes(
            x_range=[-1.6, 1.6, 0.5],
            y_range=[-1.6, 1.6, 0.5],
            x_length=6,
            y_length=6,
        )
        points = VGroup(
            *[Dot(axes.c2p(x, y), color=BLUE, radius=0.075) for x, y in BLUE_POINTS],
            *[Dot(axes.c2p(x, y), color=RED, radius=0.075) for x, y in RED_POINTS],
        )
        radius = np.sqrt(0.7)
        boundary = ParametricFunction(
            lambda angle: axes.c2p(
                radius * np.cos(angle),
                radius * np.sin(angle),
            ),
            t_range=[0, TAU],
            color=YELLOW,
            stroke_width=5,
        )
        equation = Text(
            "z = c,  z = x² + y²  ⇒  x² + y² = c",
            font_size=30,
        ).to_edge(UP)
        label = Text(
            "Non-linear decision boundary in original space",
            font_size=24,
            color=YELLOW,
        ).to_edge(DOWN)

        self.play(Create(axes), FadeIn(points))
        self.play(Write(equation), Create(boundary))
        self.play(FadeIn(label))
        self.wait(2)
        self.play(
            FadeOut(axes),
            FadeOut(points),
            FadeOut(boundary),
            FadeOut(equation),
            FadeOut(label),
        )

    def show_summary(self):
        summary = Text(
            "A linear separator in feature space can become\n"
            "a non-linear boundary in the original space.",
            font_size=30,
            line_spacing=1.2,
        )
        caveat = Text(
            "The 3D lift builds intuition; an RBF kernel uses an implicit\n"
            "high-dimensional (potentially infinite-dimensional) feature space.",
            font_size=22,
            color=YELLOW,
            line_spacing=1.1,
        ).next_to(summary, DOWN, buff=0.5)
        self.play(Write(summary))
        self.play(FadeIn(caveat, shift=UP))
        self.wait(3)
