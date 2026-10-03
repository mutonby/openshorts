def custom_filtergraph(orig_w, orig_h, out_w, out_h, panels):
    """
    Creates an FFmpeg filtergraph for an arbitrary number of user-defined panels.
    `panels` is a list of dictionaries, each specifying:
      - 'crop': {'x': f, 'y': f, 'w': f, 'h': f} (fractions of source dimensions)
      - 'dest': {'x': f, 'y': f, 'w': f, 'h': f} (fractions of output dimensions)
    """
    if not panels:
        # Fallback if empty payload
        return f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}[v]"

    count = len(panels)
    parts = [f"[0:v]split={count}" + "".join(f"[s{i}]" for i in range(count)) + ";"]

    for i, panel in enumerate(panels):
        crop = panel.get('crop', {'x': 0, 'y': 0, 'w': 1, 'h': 1})
        dest = panel.get('dest', {'x': 0, 'y': 0, 'w': 1, 'h': 1})

        # Source crop geometry
        crop_w = max(2, int(crop['w'] * orig_w))
        crop_h = max(2, int(crop['h'] * orig_h))
        crop_x = max(0, int(crop['x'] * orig_w))
        crop_y = max(0, int(crop['y'] * orig_h))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        crop_x -= crop_x % 2
        crop_y -= crop_y % 2

        # Destination geometry
        dest_w = max(2, int(dest['w'] * out_w))
        dest_h = max(2, int(dest['h'] * out_h))
        dest_x = max(0, int(dest['x'] * out_w))
        dest_y = max(0, int(dest['y'] * out_h))
        dest_w -= dest_w % 2
        dest_h -= dest_h % 2
        dest_x -= dest_x % 2
        dest_y -= dest_y % 2

        # Crop, then scale to cover the destination box, then crop to exact destination box
        parts.append(f"[s{i}]crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},"
                     f"scale={dest_w}:{dest_h}:force_original_aspect_ratio=increase,"
                     f"crop={dest_w}:{dest_h}[t{i}];")

    # Base black canvas
    parts.append(f"color=c=black:s={out_w}x{out_h}[bg0];")

    # Overlay each panel onto the canvas
    for i in range(count):
        dest = panels[i].get('dest', {'x': 0, 'y': 0, 'w': 1, 'h': 1})
        dest_x = max(0, int(dest['x'] * out_w))
        dest_y = max(0, int(dest['y'] * out_h))
        dest_x -= dest_x % 2
        dest_y -= dest_y % 2
        
        out_node = f"[bg{i+1}]" if i < count - 1 else "[v]"
        parts.append(f"[bg{i}][t{i}]overlay=x={dest_x}:y={dest_y}:shortest=1{out_node}")
        if i < count - 1:
            parts[-1] += ";"

    return "".join(parts)
