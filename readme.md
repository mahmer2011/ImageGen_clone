The code will NOT find that by default (it looks for sam_vit_b.pth in the parent folder), so set SAM_CHECKPOINT explicitly in the same terminal where you run Python:

If you're using cmd.exe:
```
cd C:\Users\PC\Desktop\imgGen\imgGen
venv\Scripts\activate
set SAM_CHECKPOINT=C:\Users\PC\Desktop\imgGen\imgGen\sam_vit_b_01ec64.pth
set ENABLE_SAM=1
```

Note: In cmd.exe, use `set VARIABLE=value` (no spaces around the = sign).
