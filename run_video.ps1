$env:PYTHONIOENCODING="utf-8"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$outdir = "C:\dev\DART\video"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$outfile = "$outdir\output_$timestamp.mp4"
python demo_video.py --video "C:\Users\Rocks\Videos\2026-04-24 20-28-15.mkv" --classes person horse tiger sword cane stick --trt hf_backbone_fp16.engine --trt-enc-dec enc_dec_fp16.engine --checkpoint sam3.pt --imgsz 1008 --output $outfile
Write-Host "Sparad: $outfile"
