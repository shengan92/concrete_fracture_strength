# Masked crops (pre-labels for training)

512x512 px tiles cut from the full-resolution photos in `data/raw/`, with a matching label mask for each.
Currently covers images 01-10 (OPC-FoP7 and OPC-FoP8 groups): 236 tiles.

**These masks are automatic pre-labels from a color/texture heuristic, not ground truth.**
They are meant to be corrected by hand (napari, CVAT or ilastik) before training a supervised model.
Mark corrected tiles by changing `status` in `tiles.csv` from `prelabel` to `corrected`.

## Layout

| Folder | Contents |
|---|---|
| `images/` | RGB tile, `<image>_y<row>_x<col>.jpg` (top-left pixel in the full image) |
| `masks/` | 8-bit PNG, same name, one class code per pixel (see below) |
| `previews/` | Tile with the mask drawn on top, for quick review only |
| `tiles.csv` | One row per tile: source image, position, % of each class, `status` |

## Mask codes

| Code | Class | Preview color |
|---|---|---|
| 0 | paste | none |
| 1 | sand | yellow |
| 2 | sand void (socket where a grain pulled out) | orange |
| 3 | round air void | cyan |
| 4 | unknown / out of focus (ignore in training) | purple |
| 255 | outside analysis region (ignore in training) | gray |

Tiles are kept only if at least 50% of their pixels have codes 0-3.

## Known limitations of the pre-labels

- Gray sand grains that resemble paste are often missed (sand is undercounted).
- Dark sand grains and empty sockets look alike, so some orange is really dark sand.
- Masks were made at half resolution and upscaled, so edges are slightly blocky.

## Regenerate

```
python src/labeling/make_crops.py data/raw data/labels/masks_halfres data/crops
```

The half-resolution masks in `data/labels/masks_halfres/` come from `src/labeling/seg2.py`;
per-image sand percentages are in `data/labels/sand_composition.csv`.
