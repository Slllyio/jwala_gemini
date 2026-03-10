import torch, yaml, os, glob

ckpt = torch.load('outputs/checkpoints/best_detect.pth', map_location='cpu', weights_only=False)
print('=== Checkpoint Metadata ===')
print(f'  Epoch saved at  : {ckpt.get("epoch")}')
print(f'  Best IoU        : {ckpt.get("best_iou")}')
cfg_snap = ckpt.get('config', {})
print(f'  Config snapshot :')
for k,v in cfg_snap.items():
    print(f'    {k}: {v}')

logs = (glob.glob('outputs/**/*.csv', recursive=True) +
        glob.glob('outputs/**/*.log', recursive=True) +
        glob.glob('logs/**/*', recursive=True))
print('\nTraining logs found:')
for l in logs[:20]: print(f'  {l}')
