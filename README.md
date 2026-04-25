# DART-local — Personlig Windows-installation av DART / SAM3

> **Attribution:** Detta är en personlig kopia av [DART](https://github.com/mehmetkeremturkcan/DART) av [Mehmet Kerem Turkcan](https://github.com/mehmetkeremturkcan).
> All ära för forskningen, modellen och kärnimplementationen tillhör originalförfattaren.
> Se hans [arXiv-paper](https://arxiv.org/abs/2603.11441) och [HuggingFace-sida](https://huggingface.co/mehmetkeremturkcan/DART) för detaljer.
>
> **Vad jag har gjort:** Anpassat projektet för att köra lokalt på Windows 11 med en Nvidia GPU (testat på RTX 4080),
> skrivit den här installationsguiden, samt gjort mindre kodfixar (se nedan).

---

## Installationsguide för Windows + Nvidia GPU (TensorRT)

### Förutsättningar

- Windows 11
- Conda-miljö aktiverad, stå i mappen `C:\dev\DART`
- Python 3.11+, PyTorch med CUDA installerat

---

### Steg 1 — Installera TensorRT

```powershell
pip install tensorrt
```

---

### Steg 2 — Ladda ner modellvikterna

Hämta SAM3-vikterna (~3.45 GB) från HuggingFace:

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/pankjkkkkkk/sam3_pt/resolve/main/sam3.pt" -OutFile "sam3.pt"
```

---

### Steg 3 — Bygg TensorRT-motorn för Encoder-Decoder

Exportera till ONNX:

```powershell
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt --output enc_dec.onnx --max-classes 4 --imgsz 1008
```

Kompilera till FP16-motor:

```powershell
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16 --mixed-precision none
```

Resultat: `enc_dec_fp16.engine` skapas.

---

### Steg 4 — Ladda ner backbone-filer

För att bygga ViT-H backbone behöver `transformers`-biblioteket två filer.
Vi hämtar dem från en öppen spegel (undviker Facebooks inloggningsspärr):

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/pankjkkkkkk/sam3_pt/resolve/main/config.json" -OutFile "config.json"
Invoke-WebRequest -Uri "https://huggingface.co/pankjkkkkkk/sam3_pt/resolve/main/model.safetensors" -OutFile "model.safetensors"
```

---

### Steg 5 — Peka om export-skriptet

Öppna `scripts/export_hf_backbone.py` i en editor. Hitta raden (~rad 431):

```python
model = Sam3Model.from_pretrained("facebook/sam3", ...)
```

Ändra `"facebook/sam3"` till `"."` (nuvarande mapp):

```python
model = Sam3Model.from_pretrained(".", ...)
```

Spara filen.

---

### Steg 6 — Bygg TensorRT-motorn för Backbone

Lägg en valfri bild i mappen och döp den till `x.jpg`. Kör sedan:

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/export_hf_backbone.py --image x.jpg --imgsz 1008
```

> **Obs:** Skriptet kan kasta ett `401 Unauthorized`-fel i slutet när det försöker köra ett test mot HuggingFace. Det kan ignoreras — `hf_backbone_fp16.engine` har redan sparats.

---

### Steg 7 — Kör detektering

**Stillbild:**

```powershell
python demo_multiclass.py --image x.jpg --classes person car bicycle dog `
    --trt hf_backbone_fp16.engine --trt-enc-dec enc_dec_fp16.engine `
    --checkpoint sam3.pt --fast --detection-only -o x_annotated.jpg
```

**Video (realtid):**

```powershell
python demo_video.py --video din_video.mp4 --classes person car laptop phone `
    --trt hf_backbone_fp16.engine --trt-enc-dec enc_dec_fp16.engine `
    --checkpoint sam3.pt --imgsz 1008 --display
```

Byt ut klassnamnen efter `--classes` till vad du vill att modellen ska leta efter.

---

### Tips för högre prestanda

| Tips | Effekt |
|---|---|
| Lägg till `--compile default` | Ytterligare GPU-optimering |
| Sänk `--imgsz 1008` till `--imgsz 644` | Markant högre FPS, något lägre precision |
| Bygg engine med fler `--max-classes` | Stöd för fler klasser samtidigt |

---

## Mina kodfixar

Utöver installationsanpassningarna har följande kodrättningar gjorts:

- **`sam3/logger.py`** — Guard mot duplicerade log-handlers vid upprepade anrop till `get_logger()`
- **`sam3/model_builder.py`** — Smalare exception-hantering i `load_pruned_config`; `get_device_properties` använder `current_device()` istället för hårdkodad `0`
- **`sam3/model/sam3_multiclass_fast.py`** — Tydligt felmeddelande om text-cache har fel klasser i TRT-only-läge
- **`sam3/eval/postprocessors.py`** — Ersatte `assert`/`RuntimeError("TODO")` med `NotImplementedError` och beskrivande meddelanden
- **`demo_multiclass.py` / `demo_video.py`** — Validering av `--mask-blocks`-format vid uppstart
