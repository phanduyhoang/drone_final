import json, cv2, sys

BASE = r'C:\Users\duyho\Desktop\thesis_experiments'
OUT  = BASE + r'\diagnose_out.txt'

lines = []

# Check trajectory closure
td = json.load(open(BASE + r'\square_loop_400steps_10px_smooth.json'))
traj = td['trajectory']
meta = td.get('meta', {})
lines.append('Meta: ' + str(meta))
lines.append('Total points: %d' % len(traj))
p0 = traj[0]; pN = traj[-1]
lines.append('First x=%.2f y=%.2f yaw=%.4f' % (p0['x'], p0['y'], p0['yaw']))
lines.append('Last  x=%.2f y=%.2f yaw=%.4f' % (pN['x'], pN['y'], pN['yaw']))
lines.append('Gap   dx=%.4f dy=%.4f' % (pN['x']-p0['x'], pN['y']-p0['y']))

xs = [p['x'] for p in traj]
ys = [p['y'] for p in traj]
lines.append('Traj X: %.1f to %.1f  span=%.1f' % (min(xs), max(xs), max(xs)-min(xs)))
lines.append('Traj Y: %.1f to %.1f  span=%.1f' % (min(ys), max(ys), max(ys)-min(ys)))

# Snow image
img = cv2.imread(BASE + r'\images\SD_TLS_190530_1m_Red_AlignSeparadoOK.tif', cv2.IMREAD_COLOR)
lines.append('Snow image shape: ' + str(img.shape))
H, W = img.shape[:2]

sz = meta.get('image_size', [10000, 10000])
mw, mh = float(sz[0]), float(sz[1])
sx = W / mw; sy = H / mh
lines.append('Scale sx=%.4f sy=%.4f' % (sx, sy))
scaled_xs = [p['x']*sx for p in traj]
scaled_ys = [p['y']*sy for p in traj]
lines.append('Scaled traj X: %.1f to %.1f' % (min(scaled_xs), max(scaled_xs)))
lines.append('Scaled traj Y: %.1f to %.1f' % (min(scaled_ys), max(scaled_ys)))
lines.append('Image W=%d H=%d' % (W, H))
margin = 150
fits = (min(scaled_xs) > margin and max(scaled_xs) < W-margin and
        min(scaled_ys) > margin and max(scaled_ys) < H-margin)
lines.append('Trajectory fits inside image (margin=150px): %s' % fits)

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
