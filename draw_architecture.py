"""Generate publication-ready English and Chinese architecture diagrams for SeisFormer (without working headers or disclaimers)."""
from pathlib import Path
from html import escape
import math
import xml.etree.ElementTree as ET


LABELS = {
    "en": {
        "font": "Arial, Helvetica, sans-serif",
        "wave_title": "3-C Waveform X",
        "wave_sub": "100 Hz · 60 s (3 × 6,000)",
        "stem_title": "Temporal Conv1d Stem",
        "stem_sub1": "4-Stage Stride-2 Conv1d",
        "stem_sub2": "6,000 → 375 tokens (96 ch)",
        "cwt_title": "Morlet CWT Filterbank",
        "cwt_sub": "12 Scales (0.5–25 Hz)",
        "stft_title": "Multi-Component STFT",
        "stft_sub": "Window 128 · Hop 16",
        "fuse_title": "1×1 Conv Fusion",
        "fuse_sub": "Concat + Zero-Init",
        "tf_title": "Transformer ×3",
        "tf_sub1": "96 dim · 4 Heads",
        "tf_sub2": "RoPE + SwiGLU",
        "skip_label": "Multi-Scale Encoder Skip Features",
        "dec_title": "U-Net Skip Decoder",
        "dec_sub1": "Linear Upsample + Conv1d",
        "dec_sub2": "375 → 6,000 samples",
        "prob_title": "Phase Probability Curves",
        "prob_sub1": "2 × 6,000 Sigmoid Output",
        "prob_sub2": "Sample-Wise P & S Curves",
        "out_title": "Arrival Picks & Event Score",
        "out_sub1": "Picked Onsets (P, S)",
        "out_sub2": "Window Event Score q",
        "mlp_box": "MLP → q",
        "pool_label": "Valid-Token Mean Pooling → Window Event Detection Head",
    },
    "zh": {
        "font": "'Microsoft YaHei', 'PingFang SC', Arial, sans-serif",
        "wave_title": "三分量地震波形 X",
        "wave_sub": "100 Hz · 60 s (3 × 6,000)",
        "stem_title": "时域一维卷积编码",
        "stem_sub1": "四级步长 2 下采样卷积",
        "stem_sub2": "6,000 → 375 token (96 通道)",
        "cwt_title": "Morlet 小波滤波器组",
        "cwt_sub": "12 个频率尺度 (0.5–25 Hz)",
        "stft_title": "多分量短时傅里叶变换",
        "stft_sub": "STFT (窗长 128 · 步长 16)",
        "fuse_title": "1×1 卷积融合",
        "fuse_sub": "通道拼接 + 零初始化",
        "tf_title": "Transformer 编码 ×3",
        "tf_sub1": "96 维 · 4 注意力头",
        "tf_sub2": "RoPE + SwiGLU",
        "skip_label": "多尺度编码特征跳跃连接",
        "dec_title": "跳跃连接上采样解码",
        "dec_sub1": "四级线性插值 + 一维卷积",
        "dec_sub2": "375 → 6,000 采样点",
        "prob_title": "逐采样震相概率序列",
        "prob_sub1": "双通道 2 × 6,000 输出",
        "prob_sub2": "P / S 到时概率曲线",
        "out_title": "震相到时与事件概率",
        "out_sub1": "P / S 初至到时估计",
        "out_sub2": "窗级事件存在概率 q",
        "mlp_box": "MLP → q",
        "pool_label": "有效 Token 均值池化 → 窗级事件检测头",
    },
}


def build_svg(lang: str) -> str:
    L = LABELS[lang]
    W, H = 1640, 585
    Y_OFF = -72

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<style>text{{font-family:{L["font"]};fill:#243447}}</style>'
        f'<rect width="{W}" height="{H}" fill="white"/>'
    ]

    def t(x, y, v, size=16, bold=False, anchor="middle"):
        yy = y + Y_OFF
        s.append(
            f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(v)}</text>'
        )

    def path(d, color="#5c6b7a", width=2.0):
        s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def arrow_head(x, y, direction="right", color="#5c6b7a"):
        # Explicit polygon arrowhead so PDF/PNG renderers always draw crisp arrows
        if direction == "right":
            pts = f"{x},{y} {x-9},{y-4.5} {x-9},{y+4.5}"
        elif direction == "down":
            pts = f"{x},{y} {x-4.5},{y-9} {x+4.5},{y-9}"
        elif direction == "up":
            pts = f"{x},{y} {x-4.5},{y+9} {x+4.5},{y+9}"
        s.append(f'<polygon points="{pts}" fill="{color}"/>')

    def dashed_line(x1, y1, x2, y2, dash=7, gap=5, color="#7d92a6", width=1.8):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        pos = 0.0
        segs = []
        while pos < dist:
            end = min(pos + dash, dist)
            segs.append(f"M{x1+ux*pos:.1f} {y1+uy*pos:.1f} L{x1+ux*end:.1f} {y1+uy*end:.1f}")
            pos += dash + gap
        path(" ".join(segs), color=color, width=width)

    def rect(x, y, w, h, color, rx=0):
        yy = y + Y_OFF
        s.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{h}" rx="{rx}" fill="{color}" stroke="#8fa3b8" stroke-width="1.2"/>')

    def grid(x, y, w, h, color):
        yy0 = y + Y_OFF
        for k in reversed(range(3)):
            xx = x + k * 8
            yy = yy0 - k * 5
            s.append(f'<path d="M{xx} {yy} l{w} -14 v{h} l{-w} 14Z" fill="{color}" fill-opacity=".78" stroke="#7890a3"/>')
            for j in range(1, 7):
                path(f"M{xx+w*j/7} {yy-14*j/7} v{h}", "white", 0.8)
            for j in range(1, 5):
                path(f"M{xx} {yy+h*j/5} l{w} -14", "white", 0.8)

    def wave(x, y, w, h, j=0):
        yy = y + Y_OFF
        p = []
        for i in range(300):
            u = i / 299
            env = 0.04 + 0.7 * math.exp(-((u - 0.34) / 0.08) ** 2) + 0.4 * math.exp(-((u - 0.72) / 0.12) ** 2)
            p.append((x + u * w, yy + math.sin(u * 185 + j) * env * h))
        path("M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in p), "#3e6580", 1.3)

    # Outer border box
    rect(12, 84, 1616, 568, "#ffffff", 12)

    # 3-component input waveforms
    for j, label in enumerate(("E", "N", "Z")):
        t(42, 310 + j * 46, label, 15, True)
        wave(60, 305 + j * 46, 132, 23, j)
    t(130, 463, L["wave_title"], 17, True)
    t(130, 488, L["wave_sub"], 14)

    # Branching arrows from input
    path(f"M202 {352+Y_OFF} H224")
    path(f"M224 {352+Y_OFF} V{240+Y_OFF} H260")
    arrow_head(264, 240 + Y_OFF, "right")

    # Temporal Conv1d Stem
    t(340, 162, L["stem_title"], 17, True)
    for j, (w, h) in enumerate(((28, 120), (28, 96), (28, 77), (28, 60))):
        x = 268 + j * 41
        y = 240 - h / 2
        rect(x, y, w, h, "#94b8d7")
        yy = y + Y_OFF
        s.append(f'<path d="M{x} {yy} l6 -5 h{w} l-6 5Z" fill="#bed6ea" stroke="white"/>')
    t(340, 326, L["stem_sub1"], 14)
    t(340, 348, L["stem_sub2"], 14)

    # Spectral branches: Morlet CWT and STFT
    path(f"M224 {352+Y_OFF} V{565+Y_OFF}")
    path(f"M224 {425+Y_OFF} H262")
    arrow_head(266, 425 + Y_OFF, "right")
    path(f"M224 {565+Y_OFF} H262")
    arrow_head(266, 565 + Y_OFF, "right")

    grid(270, 396, 110, 62, "#98c6b5")
    t(336, 492, L["cwt_title"], 15, True)
    t(336, 513, L["cwt_sub"], 13.5)

    grid(270, 545, 110, 62, "#dfb690")
    t(336, 627, L["stft_title"], 15, True)
    t(336, 646, L["stft_sub"], 13.5)

    # Arrows into 1x1 fusion (collector at x=444, bar at x=517, text centered at x=528)
    path(f"M424 {240+Y_OFF} H444 V{348+Y_OFF} H510")
    arrow_head(514, 348 + Y_OFF, "right")
    path(f"M408 {428+Y_OFF} H444")
    path(f"M408 {578+Y_OFF} H444 V{348+Y_OFF}")
    rect(517, 282, 22, 139, "#a99bc9")
    t(528, 455, L["fuse_title"], 15, True)
    t(528, 477, L["fuse_sub"], 13.5)

    # Transformer Encoder x3
    path(f"M547 {350+Y_OFF} H602")
    arrow_head(606, 350 + Y_OFF, "right")
    for j in reversed(range(3)):
        rect(616 + j * 13, 293 - j * 9, 116, 115, "#ebe5f4", 7)
    for a in range(4):
        for b in range(4):
            path(f"M{636+a*25} {322+Y_OFF} L{636+b*25} {371+Y_OFF}", "#a898c2", 0.9)
        s.append(
            f'<circle cx="{636+a*25}" cy="{322+Y_OFF}" r="4" fill="#9a84b7"/>'
            f'<circle cx="{636+a*25}" cy="{371+Y_OFF}" r="4" fill="#9a84b7"/>'
        )
    t(686, 220, L["tf_title"], 17, True)
    t(686, 455, L["tf_sub1"], 14)
    t(686, 477, L["tf_sub2"], 14)

    # U-Net Skip Decoder (title placed at y=162 matching Temporal Conv1d Stem, skip line lands cleanly at x=946, y=242 below title or routes above title to x=946, y=136)
    path(f"M772 {350+Y_OFF} H820")
    arrow_head(824, 350 + Y_OFF, "right")
    for j in range(4):
        x = 836 + j * 54
        h = 56 + j * 23
        rect(x, 350 - h / 2, 32, h, "#93c2b7")

    # Dashed skip connection from Temporal Stem (x=340, y=140) over to Decoder top (x=932, y=245), with Decoder title placed at y=220 and skip line entering at x=1014 (4th decoder bar top) to avoid crossing text
    t(912, 220, L["dec_title"], 17, True)
    dashed_line(340, 140 + Y_OFF, 340, 114 + Y_OFF)
    dashed_line(340, 114 + Y_OFF, 1014, 114 + Y_OFF)
    dashed_line(1014, 114 + Y_OFF, 1014, 272 + Y_OFF)
    arrow_head(1014, 276 + Y_OFF, "down", color="#7d92a6")
    t(677, 105, L["skip_label"], 14)
    t(932, 455, L["dec_sub1"], 14)
    t(932, 477, L["dec_sub2"], 14)

    # Sample-wise P/S probability curves
    path(f"M1046 {350+Y_OFF} H1106")
    arrow_head(1110, 350 + Y_OFF, "right")
    t(1235, 220, L["prob_title"], 17, True)
    for j, (col, center) in enumerate((("#5d96c4", 0.32), ("#c97d84", 0.72))):
        y = 315 + j * 83
        yy = y + Y_OFF
        p = []
        for i in range(200):
            u = i / 199
            p.append((1134 + u * 204, yy - 63 * math.exp(-((u - center) / 0.04) ** 2)))
        path("M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in p), col, 2.3)
        path(f"M1133 {yy} H1341", "#ccd6e0", 1)
        t(1115, y - 17, ("P", "S")[j], 15, True)
    t(1235, 455, L["prob_sub1"], 14)
    t(1235, 477, L["prob_sub2"], 14)

    # Arrival Picks & Event Score
    path(f"M1351 {350+Y_OFF} H1398")
    arrow_head(1402, 350 + Y_OFF, "right")
    t(1515, 265, L["out_title"], 17, True)
    for j, (phase, xx, col) in enumerate((("P", 1454, "#5d96c4"), ("S", 1565, "#c97d84"))):
        y = 335 + j * 65
        yy = y + Y_OFF
        path(f"M1437 {yy} H1604", "#cfd8e2", 1.2)
        path(f"M{xx} {yy-14} V{yy+14}", col, 3)
        t(1418, y + 5, phase, 15, True)
    t(1515, 455, L["out_sub1"], 14)
    t(1515, 477, L["out_sub2"], 14)

    # Event Detection Head branch
    path(f"M752 {411+Y_OFF} H790 V{575+Y_OFF} H1129")
    arrow_head(1133, 575 + Y_OFF, "right")
    for i in range(6):
        rect(1143 + i * 10, 559, 7, 30, "#a898c2")
    path(f"M1211 {575+Y_OFF} H1254")
    arrow_head(1258, 575 + Y_OFF, "right")
    rect(1268, 545, 113, 60, "#f5e8d7", 8)
    t(1324, 581, L["mlp_box"], 15, True)
    path(f"M1392 {575+Y_OFF} H1515 V{508+Y_OFF}")
    arrow_head(1515, 504 + Y_OFF, "up")
    t(960, 608, L["pool_label"], 14)

    s.append("</svg>")
    return chr(10).join(s)


def render_svg_files(svg_path: Path):
    import fitz

    doc = fitz.open(str(svg_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    pix.save(str(svg_path.with_suffix(".png")))
    pdf_bytes = doc.convert_to_pdf()
    svg_path.with_suffix(".pdf").write_bytes(pdf_bytes)
    doc.close()


def draw(root: Path):
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Chinese academic diagram
    svg_zh = build_svg("zh")
    out_zh = fig_dir / "model_architecture_zh.svg"
    out_zh.write_text(svg_zh, encoding="utf-8")
    ET.parse(out_zh)
    render_svg_files(out_zh)

    # 2. English academic diagram
    svg_en = build_svg("en")
    out_en = fig_dir / "model_architecture_en.svg"
    out_en.write_text(svg_en, encoding="utf-8")
    ET.parse(out_en)
    render_svg_files(out_en)

    # Set default model_architecture.* to English for main.tex/main.pdf
    out_default = fig_dir / "model_architecture.svg"
    out_default.write_text(svg_en, encoding="utf-8")
    render_svg_files(out_default)

    print("Rendered clean English and Chinese architecture diagrams:", out_en, out_zh)


if __name__ == "__main__":
    draw(Path(__file__).resolve().parent)
