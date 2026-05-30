"""1D transfer function (LUT) editor.

MVP UI:
- A small set of control points (value in [0,1] -> RGBA).
- LUT is baked to a 1D OpenGL texture.
"""

from __future__ import annotations

from dataclasses import dataclass

import imgui
import numpy as np
from OpenGL import GL


@dataclass
class TfPoint:
    value: float
    rgba: tuple[float, float, float, float]


class TransferFunction:
    """Transfer function with OpenGL 1D LUT texture."""

    def __init__(self, size: int = 256) -> None:
        self._size = int(size)
        self._points: list[TfPoint] = [
            TfPoint(0.0, (0.0, 0.0, 0.0, 0.0)),
            TfPoint(0.25, (0.2, 0.2, 0.2, 0.1)),
            TfPoint(0.5, (0.6, 0.6, 0.6, 0.4)),
            TfPoint(0.75, (0.9, 0.9, 0.9, 0.7)),
            TfPoint(1.0, (1.0, 1.0, 1.0, 1.0)),
        ]

        self._lut_rgba = self._bake_lut()
        self._selected_idx: int | None = None

        # OpenGL texture is created lazily once an OpenGL context exists.
        self._tex_id: int = 0

    @property
    def lut_texture_id(self) -> int:
        return int(self._tex_id)

    def ensure_gl(self) -> None:
        """Ensure the 1D LUT OpenGL texture exists.

        Must be called after a valid OpenGL context is current.
        """

        if self._tex_id != 0:
            return
        self._tex_id = self._create_texture(self._lut_rgba)

    def draw_imgui(self) -> None:
        """Draw TF editor controls and update LUT if changed."""

        changed_any = False

        imgui.text("2D alpha editor")
        imgui.text_disabled(
            "LMB drag: move | double-click: add | RMB on point: remove"
        )
        changed_any |= self._draw_alpha_editor()

        selected = self._selected_point()
        if selected is not None:
            imgui.separator()
            imgui.text(f"Selected point #{self._selected_idx}")

            r, g, b, a = selected.rgba
            changed, color = imgui.color_edit4("RGBA", r, g, b, a)
            if changed:
                selected.rgba = (
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                    float(color[3]),
                )
                changed_any = True

            changed, value = imgui.slider_float("Value", selected.value, 0.0, 1.0)
            if changed:
                self._set_point_value(self._selected_idx, float(value))
                changed_any = True

            changed, alpha = imgui.slider_float("Alpha", selected.rgba[3], 0.0, 1.0)
            if changed:
                selected.rgba = (selected.rgba[0], selected.rgba[1], selected.rgba[2], float(alpha))
                changed_any = True

        imgui.separator()
        if imgui.button("Add point"):
            self._selected_idx = self.add_control_point()
            changed_any = True

        imgui.same_line()
        remove_enabled = self._can_remove_selected()
        if imgui.button("Remove selected") and remove_enabled:
            assert self._selected_idx is not None
            self.remove_control_point(self._selected_idx)
            changed_any = True
        if not remove_enabled:
            imgui.same_line()
            imgui.text_disabled("(select a non-endpoint point)")

        self._sort_and_pin_points()

        if changed_any:
            self._lut_rgba = self._bake_lut()
            if self._tex_id != 0:
                self._update_texture(self._lut_rgba)

    def add_control_point(self, value: float | None = None) -> int:
        """Add a point and return its index after sorting."""

        if value is None:
            value = 0.5
        value = float(np.clip(value, 0.0, 1.0))
        rgba = tuple(float(v) for v in self._sample_rgba(value))

        self._points.append(TfPoint(value, rgba))
        self._sort_and_pin_points()
        for i, p in enumerate(self._points):
            if abs(p.value - value) < 1e-6 and p.rgba == rgba:
                return i
        return len(self._points) - 1

    def remove_control_point(self, index: int) -> None:
        """Remove a non-endpoint control point by index."""

        if index <= 0 or index >= len(self._points) - 1:
            return
        del self._points[index]
        if self._selected_idx is not None:
            self._selected_idx = min(self._selected_idx, len(self._points) - 1)
            if self._selected_idx <= 0 or self._selected_idx >= len(self._points) - 1:
                self._selected_idx = None

    def _draw_alpha_editor(self) -> bool:
        """Draw the 2D value/alpha editor and process interactions."""

        avail_width, _ = imgui.get_content_region_available()
        width = float(max(220.0, min(avail_width, 380.0)))
        height = 170.0

        x0, y0 = imgui.get_cursor_screen_pos()
        imgui.invisible_button("tf_alpha_canvas", width, height)
        hovered = imgui.is_item_hovered()

        draw = imgui.get_window_draw_list()
        bg = imgui.get_color_u32_rgba(0.08, 0.08, 0.09, 1.0)
        border = imgui.get_color_u32_rgba(0.45, 0.45, 0.5, 1.0)
        draw.add_rect_filled(x0, y0, x0 + width, y0 + height, bg)
        draw.add_rect(x0, y0, x0 + width, y0 + height, border)

        def to_screen(v: float, a: float) -> tuple[float, float]:
            return (x0 + v * width, y0 + (1.0 - a) * height)

        def to_va(x: float, y: float) -> tuple[float, float]:
            v = float(np.clip((x - x0) / max(width, 1e-6), 0.0, 1.0))
            a = float(np.clip(1.0 - ((y - y0) / max(height, 1e-6)), 0.0, 1.0))
            return v, a

        line_color = imgui.get_color_u32_rgba(0.88, 0.9, 0.95, 1.0)
        for i in range(len(self._points) - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            x_a, y_a = to_screen(p0.value, p0.rgba[3])
            x_b, y_b = to_screen(p1.value, p1.rgba[3])
            draw.add_line(x_a, y_a, x_b, y_b, line_color, 2.0)

        for i, point in enumerate(self._points):
            px, py = to_screen(point.value, point.rgba[3])
            is_endpoint = i == 0 or i == len(self._points) - 1
            radius = 6.0 if i == self._selected_idx else 5.0
            color = imgui.get_color_u32_rgba(*point.rgba[:3], 1.0)
            outline = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.9)
            draw.add_circle_filled(px, py, radius, color, 12)
            draw.add_circle(px, py, radius + (1.0 if is_endpoint else 0.0), outline, 12, 1.5)

        changed = False
        io = imgui.get_io()
        mouse_x, mouse_y = io.mouse_pos

        nearest = self._find_nearest_point_index(
            mouse_x,
            mouse_y,
            x0,
            y0,
            width,
            height,
        )

        if hovered and imgui.is_mouse_clicked(0):
            self._selected_idx = nearest

        if hovered and imgui.is_mouse_double_clicked(0) and nearest is None:
            value, _ = to_va(mouse_x, mouse_y)
            self._selected_idx = self.add_control_point(value)
            changed = True

        if hovered and imgui.is_mouse_clicked(1) and nearest is not None:
            if 0 < nearest < len(self._points) - 1:
                self.remove_control_point(nearest)
                changed = True

        if self._selected_idx is not None and imgui.is_mouse_down(0):
            if not hovered and not imgui.is_item_active():
                return changed

            value, alpha = to_va(mouse_x, mouse_y)
            idx = self._selected_idx
            point = self._points[idx]

            if 0 < idx < len(self._points) - 1:
                left_v = self._points[idx - 1].value + 1e-4
                right_v = self._points[idx + 1].value - 1e-4
                value = float(np.clip(value, left_v, right_v))
                point.value = value
            elif idx == 0:
                point.value = 0.0
            else:
                point.value = 1.0

            point.rgba = (point.rgba[0], point.rgba[1], point.rgba[2], alpha)
            changed = True

        return changed

    def _find_nearest_point_index(
        self,
        mouse_x: float,
        mouse_y: float,
        x0: float,
        y0: float,
        width: float,
        height: float,
    ) -> int | None:
        """Find nearest point in editor space, if within selection threshold."""

        best_idx: int | None = None
        best_dist2 = float("inf")
        threshold2 = 12.0 * 12.0

        for i, p in enumerate(self._points):
            px = x0 + p.value * width
            py = y0 + (1.0 - p.rgba[3]) * height
            dist2 = (mouse_x - px) ** 2 + (mouse_y - py) ** 2
            if dist2 < best_dist2 and dist2 <= threshold2:
                best_dist2 = dist2
                best_idx = i

        return best_idx

    def _selected_point(self) -> TfPoint | None:
        """Return currently selected point when index is valid."""

        if self._selected_idx is None:
            return None
        if self._selected_idx < 0 or self._selected_idx >= len(self._points):
            self._selected_idx = None
            return None
        return self._points[self._selected_idx]

    def _can_remove_selected(self) -> bool:
        """Return whether selected point can be removed."""

        if self._selected_idx is None:
            return False
        return 0 < self._selected_idx < len(self._points) - 1

    def _set_point_value(self, idx: int | None, value: float) -> None:
        """Set point value while preserving endpoint constraints and ordering."""

        if idx is None or idx < 0 or idx >= len(self._points):
            return
        if idx == 0:
            self._points[idx].value = 0.0
            return
        if idx == len(self._points) - 1:
            self._points[idx].value = 1.0
            return

        left_v = self._points[idx - 1].value + 1e-4
        right_v = self._points[idx + 1].value - 1e-4
        self._points[idx].value = float(np.clip(value, left_v, right_v))

    def _sort_and_pin_points(self) -> None:
        """Sort points by value and keep boundary points pinned at 0 and 1."""

        selected_point = self._selected_point()
        self._points.sort(key=lambda point: point.value)
        self._points[0].value = 0.0
        self._points[-1].value = 1.0

        if selected_point is None:
            self._selected_idx = None
            return

        for i, point in enumerate(self._points):
            if point is selected_point:
                self._selected_idx = i
                return
        self._selected_idx = None

    def _sample_rgba(self, value: float) -> tuple[float, float, float, float]:
        """Sample RGBA from current TF curve at a scalar value."""

        values = np.array([p.value for p in self._points], dtype=np.float32)
        colors = np.array([p.rgba for p in self._points], dtype=np.float32)

        out = []
        for c in range(4):
            out.append(float(np.interp(value, values, colors[:, c])))
        return (out[0], out[1], out[2], out[3])

    def _bake_lut(self) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, self._size, dtype=np.float32)
        lut = np.zeros((self._size, 4), dtype=np.float32)

        values = np.array([p.value for p in self._points], dtype=np.float32)
        colors = np.array([p.rgba for p in self._points], dtype=np.float32)

        for c in range(4):
            lut[:, c] = np.interp(xs, values, colors[:, c]).astype(np.float32)

        return np.clip(lut, 0.0, 1.0).astype(np.float32)

    def _create_texture(self, lut: np.ndarray) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, tex)
        GL.glTexImage1D(GL.GL_TEXTURE_1D, 0, GL.GL_RGBA8, lut.shape[0], 0, GL.GL_RGBA, GL.GL_FLOAT, lut)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
        return int(tex)

    def _update_texture(self, lut: np.ndarray) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_1D, self._tex_id)
        GL.glTexSubImage1D(GL.GL_TEXTURE_1D, 0, 0, lut.shape[0], GL.GL_RGBA, GL.GL_FLOAT, lut)
        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)
