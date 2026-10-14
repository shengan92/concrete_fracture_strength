"""Cut full-resolution images and their heuristic label masks into matching tiles.

Usage (from repo root):
    python src/labeling/make_crops.py data/raw data/labels/masks_halfres data/crops

Masks are 8-bit PNGs at half resolution, codes:
    0 paste, 1 sand, 2 sand void (socket), 3 round air void, 4 unknown (out of focus),
    255 outside analysis region.
They are upscaled with nearest-neighbour to the full image size before tiling.
"""
import sys, os, glob, csv
import numpy as np
from PIL import Image

TILE = 512
MIN_VALID = 0.5          # keep a tile only if >=50% of it is labeled (codes 0-3)
CODES = {0: 'paste', 1: 'sand', 2: 'sand_void', 3: 'air_void', 4: 'unknown', 255: 'outside'}
PALETTE = {0: (0, 0, 0), 1: (255, 230, 0), 2: (255, 130, 0), 3: (0, 200, 255),
           4: (150, 60, 200), 255: (60, 60, 60)}

def colorize(mask):
    rgb = np.zeros(mask.shape + (3,), np.uint8)
    for k, c in PALETTE.items():
        rgb[mask == k] = c
    return rgb

def main(raw_dir, mask_dir, out_dir):
    for sub in ('images', 'masks', 'previews'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    rows = []
    for mpath in sorted(glob.glob(os.path.join(mask_dir, '*_mask_halfres.png'))):
        stem = os.path.basename(mpath).replace('_mask_halfres.png', '')
        img = Image.open(os.path.join(raw_dir, stem + '.jpg')).convert('RGB')
        W, H = img.size
        mask = np.asarray(Image.open(mpath).resize((W, H), Image.NEAREST))
        im = np.asarray(img)
        n = 0
        for y in range(0, H - TILE + 1, TILE):
            for x in range(0, W - TILE + 1, TILE):
                m = mask[y:y + TILE, x:x + TILE]
                valid = np.isin(m, [0, 1, 2, 3]).mean()
                if valid < MIN_VALID:
                    continue
                name = f'{stem}_y{y:04d}_x{x:04d}'
                Image.fromarray(im[y:y + TILE, x:x + TILE]).save(
                    os.path.join(out_dir, 'images', name + '.jpg'), quality=95)
                Image.fromarray(m).save(os.path.join(out_dir, 'masks', name + '.png'))
                prev = (0.45 * im[y:y + TILE, x:x + TILE] + 0.55 * colorize(m)).astype(np.uint8)
                keep = (m == 0)[..., None]
                prev = np.where(keep, im[y:y + TILE, x:x + TILE], prev)
                Image.fromarray(prev).save(os.path.join(out_dir, 'previews', name + '.jpg'), quality=85)
                lab = m[np.isin(m, [0, 1, 2, 3])]
                frac = {CODES[k]: round(100 * (lab == k).mean(), 2) for k in (0, 1, 2, 3)}
                rows.append(dict(tile=name, source_image=stem + '.jpg', x=x, y=y, size=TILE,
                                 valid_pct=round(100 * valid, 1), **{f'{k}_pct': v for k, v in frac.items()},
                                 status='prelabel'))
                n += 1
        print(stem, n, 'tiles')
    with open(os.path.join(out_dir, 'tiles.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('total', len(rows))

if __name__ == '__main__':
    main(*sys.argv[1:4])
