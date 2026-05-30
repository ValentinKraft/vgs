"""OpenGL rendering utilities.

Uses depth-sorted alpha compositing:
- Splats are sorted back-to-front each frame along the current view direction.
- A single RGBA accumulation target stores premultiplied alpha blending.

Composite:
    out_color = accum
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from gs_viewer.ply_loader import GaussianModelPly


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if status != GL.GL_TRUE:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader compile failed: {log}")
    return shader


def _link_program(vs: int, fs: int) -> int:
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if status != GL.GL_TRUE:
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"Program link failed: {log}")
    return program


_SPLAT_VS = """#version 330 core

layout(location=0) in vec3 a_pos;
layout(location=1) in float a_opacity;
layout(location=2) in float a_intensity;
layout(location=3) in vec3 a_scale;
layout(location=4) in vec4 a_quat;
layout(location=5) in float a_ao;
layout(location=6) in vec2 a_corner;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_focal_px;
uniform float u_splat_scale;
uniform vec2 u_viewport;

out float v_opacity;
out float v_intensity;
out float v_ao;
out vec2 v_px;
out mat2 v_inv_cov;
out float v_max_q;

mat3 quat_to_mat3(vec4 q) {
    vec4 nq = normalize(q);
    float w = nq.x;
    float x = nq.y;
    float y = nq.z;
    float z = nq.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return mat3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
    );
}

void main() {
    float scale_mul = max(u_splat_scale, 1e-3);
    vec4 viewPos = u_view * vec4(a_pos, 1.0);
    float z = max(-viewPos.z, 1e-3);

    mat3 rot = quat_to_mat3(a_quat);
    vec3 sigma = max(a_scale * scale_mul, vec3(1e-6));
    mat3 S = mat3(
        sigma.x * sigma.x, 0.0, 0.0,
        0.0, sigma.y * sigma.y, 0.0,
        0.0, 0.0, sigma.z * sigma.z
    );
    mat3 cov_world = rot * S * transpose(rot);
    mat3 view_rot = mat3(u_view);
    mat3 cov_view = view_rot * cov_world * transpose(view_rot);

    float x = viewPos.x;
    float y = viewPos.y;
    float inv_z = 1.0 / z;
    float inv_z2 = inv_z * inv_z;

    vec3 jx = vec3(
        u_focal_px * inv_z,
        0.0,
        -u_focal_px * x * inv_z2
    );
    vec3 jy = vec3(
        0.0,
        u_focal_px * inv_z,
        -u_focal_px * y * inv_z2
    );

    vec3 cov_jx = cov_view * jx;
    vec3 cov_jy = cov_view * jy;
    mat2 cov2 = mat2(
        dot(jx, cov_jx),
        dot(jx, cov_jy),
        dot(jy, cov_jx),
        dot(jy, cov_jy)
    );
    cov2[0][0] += 1e-3;
    cov2[1][1] += 1e-3;

    float tr = cov2[0][0] + cov2[1][1];
    float det = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[1][0];
    float disc = max(0.0, tr * tr * 0.25 - det);
    float root = sqrt(disc);
    float l1 = max(1e-6, tr * 0.5 + root);
    float l2 = max(1e-6, tr * 0.5 - root);

    vec2 e1;
    if (abs(cov2[0][1]) > 1e-6) {
        e1 = normalize(vec2(l1 - cov2[1][1], cov2[0][1]));
    } else {
        e1 = (cov2[0][0] >= cov2[1][1]) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }
    vec2 e2 = vec2(-e1.y, e1.x);

    float n_sigma = 3.0;
    float r1 = n_sigma * sqrt(l1);
    float r2 = n_sigma * sqrt(l2);
    vec2 px_offset = e1 * (a_corner.x * r1) + e2 * (a_corner.y * r2);

    vec4 clip = u_proj * viewPos;
    vec2 ndc_offset = vec2(
        2.0 * px_offset.x / max(u_viewport.x, 1.0),
        2.0 * px_offset.y / max(u_viewport.y, 1.0)
    );
    vec4 out_clip = clip;
    out_clip.xy += ndc_offset * clip.w;

    gl_Position = out_clip;

    float inv_det = 1.0 / max(det, 1e-8);
    v_inv_cov = mat2(
        cov2[1][1] * inv_det,
        -cov2[0][1] * inv_det,
        -cov2[1][0] * inv_det,
        cov2[0][0] * inv_det
    );

    v_opacity = a_opacity;
    v_intensity = a_intensity;
    v_ao = a_ao;
    v_px = px_offset;
    v_max_q = n_sigma * n_sigma;
}
"""


_SPLAT_FS = """#version 330 core

layout(location=0) out vec4 out_accum;
layout(location=1) out vec4 out_reveal;

in float v_opacity;
in float v_intensity;
in float v_ao;
in vec2 v_px;
in mat2 v_inv_cov;
in float v_max_q;

uniform sampler1D u_lut;
uniform float u_gauss_softness;

void main() {
    float softness = clamp(u_gauss_softness, 0.05, 10.0);
    vec2 d = v_px;
    float q = dot(d, v_inv_cov * d);
    float max_q = v_max_q * softness * softness;
    if (q > max_q) {
        discard;
    }

    float alpha_gauss = exp(-0.5 * q / (softness * softness));

    vec4 lut = texture(u_lut, clamp(v_intensity, 0.0, 1.0));

    float alpha = clamp(v_opacity, 0.0, 1.0) * lut.a * alpha_gauss;
    if (alpha <= 1e-4) {
        discard;
    }
    vec3 rgb = lut.rgb;

    // Optional AO modulation if provided (a_ao defaults to 1).
    rgb *= clamp(v_ao, 0.0, 1.0);

    // Weighted blended OIT (simple weight).
    vec3 premul = rgb * alpha;
    out_accum = vec4(premul, alpha);

    // Revealage starts at 1 and is multiplied by (1 - alpha).
    out_reveal = vec4(1.0, 1.0, 1.0, alpha);
}
"""


_COMPOSE_VS = """#version 330 core

const vec2 verts[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

out vec2 v_uv;

void main() {
    vec2 p = verts[gl_VertexID];
    v_uv = 0.5 * (p + vec2(1.0, 1.0));
    gl_Position = vec4(p, 0.0, 1.0);
}
"""


_COMPOSE_FS = """#version 330 core

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_accum;
uniform sampler2D u_reveal;
uniform vec3 u_background;

void main() {
    vec4 accum = texture(u_accum, v_uv);
    float a = clamp(accum.a, 0.0, 1.0);
    vec3 rgb = mix(u_background, accum.rgb, a);
    out_color = vec4(rgb, 1.0);
}
"""


@dataclass
class _GpuBuffers:
    vao: int
    vbo_quad: int
    vbo_instances: int
    count: int


class OitRenderer:
    """Depth-sorted alpha-compositing renderer."""

    def __init__(self) -> None:
        self._width: int = 0
        self._height: int = 0

        self._fbo: int = 0
        self._tex_accum: int = 0
        self._tex_reveal: int = 0

        self._splat_prog: int = 0
        self._compose_prog: int = 0

        self._fs_vao: int = 0

        self._gpu: _GpuBuffers | None = None
        self._current_model_id: int | None = None
        self._positions_cpu: np.ndarray | None = None
        self._instance_cpu: np.ndarray | None = None
        self._draw_count: int = 0
        self._last_sort_view: np.ndarray | None = None

        self._init_programs()
        self._cache_uniform_locations()
        self._fs_vao = GL.glGenVertexArrays(1)

    def ensure_size(self, width: int, height: int) -> None:
        if width == self._width and height == self._height:
            return
        self._width, self._height = width, height
        self._create_or_resize_targets(width, height)

    def begin_frame(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self._width, self._height)

        # Clear single accumulation target for sorted alpha compositing.
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glClearBufferfv(GL.GL_COLOR, 0, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def render_splats(
        self,
        model: GaussianModelPly,
        view: np.ndarray,
        proj: np.ndarray,
        lut_tex_id: int,
        splat_scale: float = 1.0,
        gauss_softness: float = 1.0,
    ) -> None:
        self._ensure_model_uploaded(model)
        assert self._gpu is not None
        view32 = np.asarray(view, dtype=np.float32)
        proj32 = np.asarray(proj, dtype=np.float32)
        self._sort_and_upload_instances(view32)
        if self._draw_count <= 0:
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])

        GL.glUseProgram(self._splat_prog)

        GL.glUniformMatrix4fv(self._loc_splat_view, 1, GL.GL_TRUE, view32)
        GL.glUniformMatrix4fv(self._loc_splat_proj, 1, GL.GL_TRUE, proj32)

        # Approximate focal length in pixels.
        focal_px = 0.5 * float(self._height) / float(np.tan(np.deg2rad(45.0) / 2.0))
        GL.glUniform1f(self._loc_splat_focal_px, focal_px)
        GL.glUniform1f(self._loc_splat_scale, max(0.0, float(splat_scale)))
        GL.glUniform2f(self._loc_splat_viewport, float(self._width), float(self._height))
        GL.glUniform1f(self._loc_splat_gauss_softness, max(0.05, float(gauss_softness)))

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_1D, lut_tex_id)
        GL.glUniform1i(self._loc_splat_lut, 0)

        # Standard premultiplied alpha compositing.
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glBindVertexArray(self._gpu.vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, self._draw_count)
        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)

    def composite_to_screen(
        self,
        width: int,
        height: int,
        background_rgba: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0),
    ) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)
        r, g, b, a = background_rgba
        GL.glClearColor(
            float(np.clip(r, 0.0, 1.0)),
            float(np.clip(g, 0.0, 1.0)),
            float(np.clip(b, 0.0, 1.0)),
            float(np.clip(a, 0.0, 1.0)),
        )
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self._compose_prog)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_accum)
        GL.glUniform1i(self._loc_compose_accum, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_reveal)
        GL.glUniform1i(self._loc_compose_reveal, 1)
        GL.glUniform3f(
            self._loc_compose_background,
            float(np.clip(r, 0.0, 1.0)),
            float(np.clip(g, 0.0, 1.0)),
            float(np.clip(b, 0.0, 1.0)),
        )

        # Core profile requires a VAO bound for glDrawArrays.
        GL.glBindVertexArray(self._fs_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

    def _init_programs(self) -> None:
        vs = _compile_shader(_SPLAT_VS, GL.GL_VERTEX_SHADER)
        fs = _compile_shader(_SPLAT_FS, GL.GL_FRAGMENT_SHADER)
        self._splat_prog = _link_program(vs, fs)

        cvs = _compile_shader(_COMPOSE_VS, GL.GL_VERTEX_SHADER)
        cfs = _compile_shader(_COMPOSE_FS, GL.GL_FRAGMENT_SHADER)
        self._compose_prog = _link_program(cvs, cfs)

    def _cache_uniform_locations(self) -> None:
        self._loc_splat_view = GL.glGetUniformLocation(self._splat_prog, "u_view")
        self._loc_splat_proj = GL.glGetUniformLocation(self._splat_prog, "u_proj")
        self._loc_splat_focal_px = GL.glGetUniformLocation(self._splat_prog, "u_focal_px")
        self._loc_splat_scale = GL.glGetUniformLocation(self._splat_prog, "u_splat_scale")
        self._loc_splat_viewport = GL.glGetUniformLocation(self._splat_prog, "u_viewport")
        self._loc_splat_gauss_softness = GL.glGetUniformLocation(self._splat_prog, "u_gauss_softness")
        self._loc_splat_lut = GL.glGetUniformLocation(self._splat_prog, "u_lut")
        self._loc_compose_accum = GL.glGetUniformLocation(self._compose_prog, "u_accum")
        self._loc_compose_reveal = GL.glGetUniformLocation(self._compose_prog, "u_reveal")
        self._loc_compose_background = GL.glGetUniformLocation(self._compose_prog, "u_background")

    def _create_or_resize_targets(self, width: int, height: int) -> None:
        if self._fbo == 0:
            self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        if self._tex_accum == 0:
            self._tex_accum = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_accum)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, width, height, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._tex_accum, 0)

        if self._tex_reveal == 0:
            self._tex_reveal = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_reveal)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16F, width, height, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self._tex_reveal, 0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _sort_and_upload_instances(self, view: np.ndarray) -> None:
        if self._gpu is None:
            return
        if self._positions_cpu is None or self._instance_cpu is None:
            return
        if self._last_sort_view is not None and np.allclose(view, self._last_sort_view, rtol=0.0, atol=1.0e-5):
            return

        positions = self._positions_cpu
        view_row = view[2, :]
        depth = -(positions @ view_row[:3] + view_row[3])
        visible_idx = np.flatnonzero(depth > 1.0e-4)
        if visible_idx.size == 0:
            self._draw_count = 0
            self._last_sort_view = view.copy()
            return

        # Back-to-front for alpha blending.
        order = np.argsort(depth[visible_idx])[::-1]
        ordered_visible_idx = visible_idx[order]
        sorted_instances = np.ascontiguousarray(self._instance_cpu[ordered_visible_idx], dtype=np.float32)
        self._draw_count = int(ordered_visible_idx.size)
        self._last_sort_view = view.copy()

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._gpu.vbo_instances)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, sorted_instances.nbytes, sorted_instances)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _ensure_model_uploaded(self, model: GaussianModelPly) -> None:
        model_id = id(model.positions)
        if self._current_model_id == model_id and self._gpu is not None:
            return

        # Prepare attributes.
        positions = model.positions.astype(np.float32)
        opacity = model.opacity.astype(np.float32)
        intensity = model.intensity01.astype(np.float32)
        scale = np.exp(model.log_scale).astype(np.float32)
        quat = model.quat.astype(np.float32)

        if model.ao is None:
            ao = np.ones((model.count,), dtype=np.float32)
        else:
            ao = model.ao.astype(np.float32)

        self._positions_cpu = positions
        instance_data = np.concatenate(
            [
                positions,
                opacity[:, None],
                intensity[:, None],
                scale,
                quat,
                ao[:, None],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        self._instance_cpu = np.ascontiguousarray(instance_data)
        self._draw_count = model.count
        self._last_sort_view = None

        quad = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        vbo_instances = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_instances)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._instance_cpu.nbytes, self._instance_cpu, GL.GL_DYNAMIC_DRAW)
        stride = 13 * np.dtype(np.float32).itemsize
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 1, GL.GL_FLOAT, False, stride, ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 1, GL.GL_FLOAT, False, stride, ctypes.c_void_p(4 * 4))
        GL.glVertexAttribDivisor(2, 1)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(5 * 4))
        GL.glVertexAttribDivisor(3, 1)
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribPointer(4, 4, GL.GL_FLOAT, False, stride, ctypes.c_void_p(8 * 4))
        GL.glVertexAttribDivisor(4, 1)
        GL.glEnableVertexAttribArray(5)
        GL.glVertexAttribPointer(5, 1, GL.GL_FLOAT, False, stride, ctypes.c_void_p(12 * 4))
        GL.glVertexAttribDivisor(5, 1)

        vbo_quad = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_quad)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(6)
        GL.glVertexAttribPointer(6, 2, GL.GL_FLOAT, False, 0, None)
        GL.glVertexAttribDivisor(6, 0)

        GL.glVertexAttribDivisor(0, 1)
        GL.glVertexAttribDivisor(1, 1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        self._gpu = _GpuBuffers(
            vao=vao,
            vbo_quad=vbo_quad,
            vbo_instances=vbo_instances,
            count=model.count,
        )
        self._current_model_id = model_id
