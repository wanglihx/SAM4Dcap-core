from PIL import Image
from PIL import ImageDraw


def draw_point_marker(image: Image.Image, x: int, y: int, point_type: str) -> Image.Image:
    """
    Draw a circular marker with soft color fill:
        - Positive:  light green fill + white border + white "+"
        - Negative:  light red   fill + white border + white "-"

    Marker size auto-scales with image size.

    Args:
        image: PIL Image(RGB)
        x, y: coordinates (int)
        point_type: "positive" or "negative"

    Returns:
        PIL Image with marker drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image size
    w, h = img.size

    # ===== Auto-scale marker size =====
    base = min(w, h)
    radius = max(6, int(base * 0.015))        # Circle radius ~1.5% of the shorter side
    line_w = max(2, radius // 4)              # Stroke width for +/-
    border_w = max(2, radius // 5)            # White border thickness

    # Clamp coordinates
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # ===== Color settings =====
    if point_type.lower() == "positive":
        fill_color = (180, 255, 180)   # Light green
    else:
        fill_color = (255, 180, 180)   # Light red

    border_color = (255, 255, 255)     # White
    sign_color = (255, 255, 255)       # White

    # ===== Draw circle (fill + white border) =====
    bbox = [x - radius, y - radius, x + radius, y + radius]

    # Fill circle
    draw.ellipse(bbox, fill=fill_color)

    # Overlaid white border
    draw.ellipse(bbox, outline=border_color, width=border_w)

    # ===== Draw plus/minus sign (white stroke) =====
    # Horizontal line
    draw.line(
        (x - radius + 3, y, x + radius - 3, y),
        fill=sign_color,
        width=line_w,
    )

    # Vertical line (only for "positive")
    if point_type.lower() == "positive":
        draw.line(
            (x, y - radius + 3, x, y + radius - 3),
            fill=sign_color,
            width=line_w,
        )

    return img
