# AI-Based Restoration of Degraded Images for Semiconductor Inspection – SEMICON India Hackathon 2026

## Executive Summary  
Semiconductor wafer inspections rely on ultra-high-resolution images (microscope or SEM) to detect minute defects. However, **speckle noise** (granular artifacts from coherent illumination) and **resolution loss** (downsampling) often render these images grainy or blurred. A single noisy pixel can hide a critical flaw, threatening yield and costing millions. We propose an **end-to-end AI solution** that jointly *denoises* and *super-resolves* inspection images. Using a lightweight convolutional network (inspired by Real-ESRGAN [35]) with residual-in-residual dense blocks, our model will remove speckle/Gaussian noise and upscale images (×2) in one pass. This yields clear, high-fidelity images that reveal hidden defects. The solution leverages modern frameworks (e.g. PyTorch, NVIDIA GPUs) for fast training/inference within a 24–36h hackathon. It is timely due to advances in deep learning and fab-level compute, and it promises a significant impact: e.g. **improving PSNR by ~30–40%** over degraded inputs, or similarly boosting defect recall in downstream inspection by >30%. All design choices (model, losses, pipeline) are optimized for a single-GPU environment, with fallbacks (bicubic+classical denoising) ready if needed. This report outlines the problem roots, unique value proposition, technical design (including a mermaid architecture), MVP roadmap, algorithms, data strategy, risk plan, pitch strategy, coding prompts, and key resources.  

## 1. Deep-Dive Problem Analysis & Root Cause  

**Stakeholder Pain Points:**  
- *Primary users* (wafer inspection engineers, fab yield managers) currently **work with noisy, low-res images**. They risk missing sub-micron defects because of grain and blur. This threatens yield (one hidden defect can scrap a wafer) and slows inspections (manual zooming, repeated scans).  
- *Secondary stakeholders* include semiconductor equipment manufacturers (KLA, Applied Materials) who need higher trust in their tools, and fab operators who seek to minimise scrap and rework. They’re frustrated by the gap between sensor capability and actionable imaging.  
- *Pain examples:* Speckle noise (granular, due to coherent light) randomly brightens/darkens pixels, making subtle pattern changes (defects) invisible. Similarly, **downsampling** eliminates fine features: e.g. a 512×512 design shrunk to 128×128 loses detail. Engineers currently cope with manual fixes or ignore small anomalies, risking escapes.

**Current State vs. Desired State:**  
- *Current:* The dataset consists of paired images: each input is a noisy, downsampled grayscale frame (128×128 or 256×256) and the ground truth is a clean high-res image (256×256 or 512×512). Today’s baseline solution might be **simple upsampling or denoising filters** (bilinear/bicubic + median filter), which blur details or insufficiently clean the noise.  
- *Desired:* An automated AI model that takes a degraded image and outputs a **sharp, clean full-resolution image** matching the ground truth. The output should preserve true microstructure detail (defect signatures) without introducing artifacts. In ideal state, **detection of defects improves**, and inspectors can rely on images directly.

**Feasibility & Constraints:**  
- *Latency:* The solution must be **fast**. The hackathon mandates inference speed as a metric. Target: complete a 256×256→512×512 restore in **under 5–10 seconds** on an NVIDIA GPU. Slow models (minutes per image) will fail.  
- *Data Privacy/IP:* Provided data is proprietary KLA inspection images. For hackathon, we use only the given training set; no external wafer images (which may have NDAs). Any model deployed in industry must respect wafer design secrecy. We avoid any external data sharing, and annotate nothing sensitive. Synthetic data generation (see section 6) should use public textures or GANs, not real product layouts.  
- *Dataset Availability & Annotation Cost:* The hackathon supplies a **paired training dataset**. No manual annotation needed (we only need input-output image pairs). If additional augmentation is needed, it must be synthetic. Real wafer images with ground-truth are rare, so gathering new data is costly (like averaging 64 SEM frames to get a clean image). We’ll rely on the provided set and create augmentations via transforms (flip, rotate, intensity jitter, synthetic noise injection).  
- *Compute & Scalability:* We assume a single high-end GPU (e.g. NVIDIA A100/H100 or even a common RTX). The model should be **compact (~1–10M params)** to train/test quickly. Memory should fit on one GPU with batch sizes 4–16. The pipeline (training and inference) should run on limited cloud (no cluster). We avoid extremely heavy transformers or diffusion networks for MVP; those are too slow to train in hours.  
- *Regulatory:* There are no specific regulations for AI in inspection; however, semiconductor industry demands reliability. We must document accuracy (PSNR/SSIM) and ensure no hallucinated defects. Output must be *reproducible* and *deterministic* for audit. Using open-source frameworks is fine, but any patented model architectures should be cited properly.

 *Figure: SEM images under different noise levels. “F01” (left) is a single-frame capture: extremely noisy speckle obscures the circular patterns. Averaging 64 frames yields “F64” (right), a high-SNR clean image. Our AI aims to learn this mapping from single-frame→clean.*  

This example highlights root causes: **speckle noise** (granular, high variance) and **information loss** from low resolution. The current gap is that engineers cannot automatically recover F64 from F01, hence needed.

## 2. Unique Value Proposition (UVP) & Innovation Edge  

- **Why Now?** Advanced GPUs and deep learning breakthroughs (transformers, GANs, diffusion) make it feasible to train sophisticated restoration models quickly. Recent methods (e.g. Real-ESRGAN, SwinIR) approach or exceed classical filter quality, even on real-world noise. The SEMICON-I4C hackathon provides curated data and mentorship (KLA guides), making it an opportune time to apply these advances. Moreover, wafer feature sizes keep shrinking (FinFETs, advanced nodes), magnifying the impact of any unseen defect – so improving image clarity has never been more critical.

- **Differentiation:** Our approach goes beyond standard tools by *jointly* addressing multiple degradations. Many open-source and commercial solutions target only denoising or only super-resolution. For example, ESRGAN (and derivatives) do SR, while DnCNN or BM3D focus on denoise. But here, images can have *both* speckle noise and blur simultaneously. We will use a unified network (e.g. multi-branch RRDB net) that explicitly handles noise removal *and* detail enhancement. Compared to off-the-shelf filters, our model learns the wafer-image domain, avoiding over-smoothing (SSIM losses and edge losses help keep sharpness). Compared to existing academic tools, we tailor data augmentations and loss functions for wafer imagery. For instance, adding an FFT-based high-frequency loss is unusual but targets the exact “fine details lost to downsampling.” These novel loss components and a lightweight residual architecture give us a faster, more robust restoration than generic solutions.  

- **Impact Metric:** We define success as a **quantitative leap in fidelity**, e.g. “**increase PSNR by ≥3–5dB** on the test set vs. bicubic baseline.” (Hackathon expects PSNR/SSIM anyway.) Alternatively, “**reduce defect false-negatives by 30%**” could be measured if integrated with a detector. For this hackathon, we set: **MVP: PSNR boost ≥5dB over degraded inputs** on validation. (Empirically, CNN denoisers often yield 20–30dB starting PSNR for F01→F64, so adding 5dB is substantial.) In real impact terms, even a few dB improvement can translate to uncovering small defects. We will highlight percentage improvement and reduction in mean squared error in our pitch.

## 3. Comprehensive System Architecture  

### High-Level Architecture (HLA)  
Our system is modular: user input → preprocessing → model inference → postprocessing → output. We will expose a simple API (or CLI) that accepts an image (or batch) and returns the restored image. Internally, we may split the network into two semi-independent paths (noise estimation vs. enhancement), but they run together for efficiency (see [7]). The diagram below outlines the flow:

```mermaid
graph LR
    A[Client / UI (Web/CLI)] --> B[API Server (FastAPI/Flask)]
    B --> C[Preprocessor] 
    C --> D[Restoration Model (PyTorch)]
    D --> E[Postprocessor (clamp, etc.)]
    E --> F[Result Store / File System]
    F --> G[User (Download/Display)]
    B --> H[(Optional DB for logs)]
```
- **Client/UI:** For hackathon, a basic web interface (e.g. React or Streamlit) or simply a CLI is enough. It takes a degraded image as input.  
- **API Server:** A lightweight Python web server (e.g. FastAPI) routes requests. It handles file upload/download, triggers preprocessing and model inference. It also logs requests and timing (in a simple SQLite or JSON log file).  
- **Preprocessor:** Reads the `.npy` or image file, normalises/reshapes it. Applies any fixed transforms (e.g. clamp intensity, maybe adjust histogram). Also optionally handles batch grouping.  
- **Restoration Model:** The core is a *lightweight RRDB network* (Residual-in-Residual Dense Blocks). It has a noise-estimation branch that predicts a per-pixel noise map, and a main branch that upsamples and refines details, adding its output to a bicubic-upsampled baseline. The entire model is implemented in PyTorch (or PyTorch Lightning). We chose PyTorch for speed of coding and ease of GPU use.  
- **Postprocessor:** After inference, we clamp outputs to valid range [0,1], convert types, and possibly apply a small final smoothing or sharpening filter (if needed). We then save the output as a `.npy` and optionally `.png` preview.  
- **Storage:** Restored images are saved to disk (or an S3 bucket) by the API. A database (SQLite) can log performance metrics (PSNR/SSIM if we compute on the fly) and inference time. This isn’t strictly needed, but useful for the demo.  

### Tech Stack Recommendations  
We prioritise speed and simplicity for a 24–36h build:  
- **Model/AI Framework:** PyTorch (1.12+) on CUDA for GPU acceleration. It’s widely used and integrates well with fast prototyping. The example repo (Rahman-001) used PyTorch too. For a quicker workflow, we might use PyTorch Lightning or even HuggingFace’s Accelerate to handle training loops.  
- **Libraries:** Key Python libs: `numpy` for data I/O, `scipy`/`skimage` for augmentations, `torchvision` for transforms. For metrics, use `torchmetrics` or custom PSNR/SSIM code. If time allows, include `lpips` for perceptual score (packaged by `lpips` pip).  
- **Backend/Web Server:** FastAPI or Flask with Uvicorn for ASGI; needs to serve HTTP or just run CLI. These are quick to set up.  
- **Frontend/UI:** Minimal: maybe Streamlit or a simple HTML/JS page that posts an image to the server. Even a static HTML with a form can suffice. Streamlit is fastest: a few lines of Python to create a drag-drop interface.  
- **Database:** SQLite or even a CSV for logs (in budget time, probably skip full DB).  
- **Compute/DevOps:** Run on a single GPU instance (Google Colab Pro, AWS EC2 g4dn, or an on-prem A100 if available). Containerise the app with Docker for reproducibility: base image `pytorch/pytorch:latest-cudaXX`. Use `requirements.txt` for Python deps.  
- **Model Storage:** Save weights as `.pt` in the repo (using Git LFS if large) or link to a Hugging Face repo.  

### Component Interaction  
- The **API server** calls the preprocessing function to load inputs, then calls the PyTorch model for inference.  
- The **model** is just a Python class; it can be loaded once at startup. Inference is triggered by `model(input_tensor)` inside the API.  
- **APIs/Microservices:** For hackathon MVP, everything can run in one container. We can mock an API endpoint like `/infer` that takes file upload. Optionally, implement an internal queuing microservice if expecting batch jobs, but likely unnecessary.  
- **Data Flow:** 
   1. Client uploads `.npy` (or PNG).  
   2. Preprocessor reads it, converts to 4D tensor (batch=1,C=1,H,W).  
   3. Model predicts noise-map and high-res output.  
   4. Postprocessor clamps and saves output files.  
   5. Response returned to client (maybe as download link or shown image).  

This simple pipeline ensures the system is transparent and modular for debugging.  

## 4. Core Feature Roadmap (MVP vs Scale)  

### Phase 1 – Hackathon MVP (24–36h)  
Focus on essentials that can be coded and demonstrated quickly:
- **Data Loading & Augmentation (2–3h):** Write a `Dataset` class to load paired `.npy` images. Implement basic augmentations: random flips/rotations, random crop (though fixed size already). Add synthetic speckle noise and Gaussian blur as augmentation to make model robust.  
- **Model Definition (4–6h):** Implement the lightweight RRDB network as per [7]. Use 8 residual dense blocks (like ESRGAN-lite) with channel attention every 3 blocks. Include the bicubic skip-connection. (Alternatively, a simple UNet could suffice if time is tight.)  
- **Loss Functions (1–2h):** Code a composite loss: Charbonnier (smooth L1) + SSIM + optional Sobel/FFT terms. Use standard PyTorch functions or third-party (e.g. `pytorch_msssim`).  
- **Training Script (3–4h):** Make `train.py` that parses `--lr_dir`, `--gt_dir`, `--epochs`, etc.. Include a validation split (e.g. 10%) to monitor PSNR/SSIM. Save best model. (Use 16–32 batch size depending on GPU.)  
- **Inference Script (2h):** Write `infer.py` that loads model and a directory of degraded images, writes outputs. Should match their format: input 128×128, output 256×256. Measure and print inference time. (Based on [5], similar interface is required.)  
- **Demo Setup (2h):** Build a quick UI (e.g. Streamlit) or simple Python CLI. For fastest demo, Streamlit can show upload button + compare images. If not, just prepare a short video or live run.  
- **Testing & Benchmark (2h):** Run on a held-out set. Compute PSNR/SSIM (Slide 6 expects these). Make before/after image montages for slides.

### Phase 2 – Extensions/Future (for Pitch)  
Mention improvements beyond MVP:
- **Enhanced Models:** Explore diffusion-based or transformer models (e.g. SwinIR) for better quality.  
- **Unsupervised or Domain-Adaptation:** Use Noise2Noise/Noise2Void techniques to leverage unpaired data.  
- **Real-time Optimization:** Convert model to TensorRT/ONNX for faster inference.  
- **Batch Processing & Scheduler:** Add a job queue or cloud function for high throughput.  
- **UI and Integration:** Full web app with authentication, cloud deployment, and API for fabs.  
- **Advanced Features:** Support variable scaling (×2, ×4), or multi-modal input (color support if future images have channels). Add visualization dashboard with metrics.

**Feature Comparison Table (MVP vs. Scale):**

| Feature                         | Effort (hrs) | MVP Demo Value    | Scale/Extension             |
|---------------------------------|-------------:|------------------|-----------------------------|
| Basic Denoising+SR Model        | 5            | Core restoration | Deeper network (SwinIR)     |
| Composite Loss (L1+SSIM+Edge)   | 2            | Sharp outputs    | Perceptual/diffusion losses |
| Data Augmentation (rotate/flip) | 2            | Robust to basic transforms | GAN-based synthetic images |
| CLI inference & batch script    | 2            | Required for eval| Wrap in Docker, add API     |
| Simple UI (Streamlit/Flask)     | 2            | Demonstration    | Full web portal, dashboard  |
| Model Monitoring & Logging      | 1            | PSNR/SSIM print  | Real-time metrics, alerts   |
| **Total**                       | **14**       | **✔ Demo**       | **(full product)**          |

Each MVP task is time-boxed; e.g. model code in 4–6h, training script in 3–4h, etc. The core demo is ready by end of Day 1; Day 2 refinements (augmentation, UI polishing, GPU tuning).

## 5. Algorithmic/Core Logic Design  

### Model Choices  
We need a network that does **joint denoising and ×2 super-resolution**. Options:
- **RRDBNet (ESRGAN-lite):** Uses residual dense blocks with skip connection; seen in [7] with ~6M params. Good trade-off: fast and already proven.  
- **UNet-based SR:** U-Net architectures (with pixel shuffle or deconv) can be easier to implement, but may lack the high-frequency detail of RRDB. Could be fallback if short on time.  
- **Transformer (SwinIR):** State-of-art for restoration. Too heavy to train in hours, but could be mentioned as Phase 2.  
- **Denoising Diffusion Model:** Very promising for generality, but too slow for quick MVP. Possibly a 2-stage: remove noise with U-Net, then a separate SR net. However, given time, we stick to one network with combined capacity.

Thus **proposed**: an *RRDB-based convolutional net with dual head:* one branch estimates noise pattern (optional) and one does residual upsampling.

### Training Strategy  
- **Data:** Use provided pairs. Split ~90/10 for train/val. (We expect OOD test later, but no extra data to simulate OOD easily.)  
- **Loss Functions:**  
  - *Charbonnier (smooth L1)* on pixel values: robust to outliers (speckle extremes).  
  - *SSIM loss:* preserves structural similarity.  
  - *Edge/Sobel loss:* to sharpen edges (defects live at boundaries).  
  - *FFT/magnitude loss:* explicitly penalises missing high-frequency content.  
Composite: `Loss = α * Charbonnier + β * (1-SSIM) + γ * Sobel + δ * FFT`. Weights tuned by quick experiments (start α=1, β=1, γ=0.1, δ=0.01).  
- **Optimizer:** Adam or AdamW, lr ~1e-4 with scheduler (reduce on plateau). Use mixed precision (fp16) if available for speed.  
- **Training:** 50–100 epochs, early stopping if val PSNR saturates. Two-stage curriculum (like [5]): pretrain with only Charbonnier (L1) for stability, then fine-tune with full loss. This often speeds convergence.  

### Evaluation Metrics  
- **PSNR, SSIM:** Standard metrics. We compute on validation each epoch (or subset).  
- **LPIPS:** Learned perceptual similarity; align with hackathon’s request.  
- **Task-driven:** While true defect detection success is ultimate goal, we may simulate it by running a toy detector on outputs (optional). In any case, emphasize “PSNR↑” as main metric.  
- **Speed:** Inference time per image on target GPU (for Hackathon, measured by their script). Aim <10s for 256×256 input.  

### Pseudocode & Data Schema  
#### Model Pseudocode  
```python
class RRDBRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_head = ConvNet1(...)  # estimate noise map
        self.base_upsample = nn.Upsample(scale_factor=2, mode='bicubic')  # baseline
        self.shallow = Conv2d(1, 64, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(8)])
        self.attn = ChannelAttentionLayer()  # every 3 blocks (as needed)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output_conv = Conv2d(16, 1, 3, 1, 1)  # after pixel shuffle (64->16->1)
    def forward(self, x):
        # x: [B,1,128,128] (noisy low-res input)
        noise_map = self.noise_head(x)           # [B,1,128,128]
        base = self.base_upsample(x)             # [B,1,256,256]
        y = self.shallow(x)                      # [B,64,128,128]
        y = self.rrdb_blocks(y)                  # [B,64,128,128]
        # (if attn blocks: inserted inside rrdb_blocks)
        y = self.pixel_shuffle(y)                # [B,16,256,256]
        res = self.output_conv(y)                # [B,1,256,256]
        out = base + res                         # [B,1,256,256]
        return torch.sigmoid(out)               # clamp to [0,1]
```

#### Core Data Schema  
- **Input data folder:** `/data/train/NoisyLR/` contains files like `00001.npy` (shape 128×128 float32, range ~[-0.3,2.2]).  
- **Ground truth:** `/data/train/GT/` with matching names (256×256, float32 [0,1]).  
- **Test data:** `/data/test/NoisyLR/` similar structure (to be restored by our inference script).  
- **Augmentation:** Implement in the Dataset: random horizontal/vertical flips, 90° rotations, and (with low prob) add extra Gaussian noise or contrast jitter. Ensure output is clamped.  
- **Checkpoint:** Save `weights/model_best.pt` (validation PSNR best) and `model_final.pt` after final epoch.  

This design covers the hardest parts: combined denoise+SR, multi-loss training, and efficient execution.

## 6. Datasets & Data Strategy  

- **Official Data:** KLA provides a paired dataset (internal, NDA) of *Noisy Low-Res* and *Ground-Truth High-Res* images. We do not know its size, but typical hackathon datasets are few thousand images. We rely solely on this for supervised training.  
- **Public Datasets:** There are limited open datasets of wafer SEM images. For example, the *Carinthia-S* SEM defect dataset has 4,591 images (with masks); useful for understanding defect types. Also, the Kaggle **WM811k wafer map** dataset provides wafer maps (not microscopic images) for classification tasks, but it shows defect patterns. For super-resolution-specific, no direct public paired dataset exists.  
- **Synthetic Data Generation:** To augment data, we can:  
  - *Create synthetic degraded images:* Use a small set of high-quality SEM images (from public sources or our own captures) and apply **simulated speckle** and **downsampling**. Speckle can be modelled by multiplying with a noise pattern (multiplicative Gaussian). For example, generate random speckle masks or use open-source SAR speckle formulas. Then downsample by 2× (with bicubic). These synthetic pairs enlarge training data.  
  - *Diffusion/GAN:* As in recent work, one could train a DDPM to generate new wafer textures or defect shapes. That study used a diffusion model to augment defect images, improving a YOLOv8 detector by augmenting 4626 images. We could similarly generate noisy/clean pairs by adding known degradations to random patterns or using a GAN trained on Carinthia images. Time-permitting, generating 1,000+ synthetic examples would help generalization.  
- **Annotation Strategy:** No manual annotation is needed for the restoration task. We only need pixel-wise ground truth, which is provided. For synthetic data, annotation is trivial (we generate both input and true output ourselves). For evaluation, no segmentation labels needed – we judge by image metrics.  
- **Data Augmentation:** Critical to improve robustness to out-of-distribution (OOD) wafers. We will apply: random flips/rotations (wafer circuits are symmetric), random crops (if original was larger), brightness/contrast adjustments, slight affine warps, and *mix-up* with synthetic noise. We might also mix examples from different semiconductor layers (if data allows) to diversify patterns.  
- **Privacy/IP Handling:** All provided data is proprietary. We will not distribute any real wafer patterns. Synthetic augmentation uses generic geometric shapes (e.g. lines, grids) or noise. The Carinthia-S (public) data has masks and is legal to use in research. We ensure no confidential layout is exposed.  
- **Sample Sizes:** Assume training set on order 1–3K images. For augmentation, we can easily double/triple that. In experiments, even 1000 images can train a small CNN. If dataset is very small (<500), we use more augmentation or pretrained weights (e.g. start from a publicly trained ESRGAN and fine-tune).  

## 7. Risk Analysis & Mitigation Plan  

1. **Model Fails or Sluggish Inference:** If the chosen network is too heavy or crashes, inference could exceed 10s/image or OOM. *Mitigation:* Prepare a simple fallback: **Bicubic Upsample + Non-Local Means (OpenCV)** or BM3D denoising as a base solution. This can run on CPU and will produce a somewhat sharper output than raw input. Keep a reference implementation ready. Additionally, implement model quantization or reduce batch to ensure fit in GPU memory. Logically, if time per image is too high, switch to a smaller model (e.g. 4 RRDB blocks instead of 8).  
2. **Training Instability/Overfitting:** With limited data, the model may not train well or may overfit. *Mitigation:* Use validation split to detect overfit. If performance stalls, revert to L1-only loss to stabilise, or use early stopping. We can also freeze deeper layers and train only a shallow network if needed. The Rahman-001 repo suggests pretraining on L1 (Charbonnier) and then fine-tuning; if full composite loss fails, stick to this two-stage training. If still underfitting, increase augmentation or pretrain on related data (e.g. train on natural images with synthetic noise for a few epochs).  
3. **Out-of-Distribution Samples:** The test set includes images “from different sources”. If our model hasn’t seen similar patterns, it may produce artifacts. *Mitigation:* Include diverse wafer patterns in augmentation. Use *Input Dropout* (Moon et al. 2025) strategy: randomly mask parts of input during training to force the model to rely on context, improving generalization. (In practice, we might randomly zero out pixels to simulate unseen variations.) Also, ensure training uses the “structurally different” images if provided, or use mixup: combine two training images. If a test image seems far OOD, fallback to baseline (e.g. output bicubic) and flag for human review.  
4. **Demo Glitches (Data I/O or Env Issues):** e.g. wrong file paths, dependency mismatches, no GPU. *Mitigation:* Containerise the entire stack early. Include a robust script in README to test inference. For GPU absence, code should check CUDA and fall back to CPU mode (with warning). Have a sanity test in place: an example image with known output to verify the pipeline quickly (before judges arrive).  

These strategies ensure that even if the AI model fails, we have a safe baseline output and an explanation for any shortcomings during the live demo.

## 8. Pitch Deck & Demo Strategy  

### The Hook (First 30s):  
_Open with impact_: “Imagine a trillion-dollar chip fab, where every wafer must be defect-free. Now imagine a tiny hidden flaw in one pixel on a noisy microscope image – a flaw that today’s engineers could miss. This hidden defect could wreck a billion-dollar product after it ships. We fix that blindspot. Our AI turns fuzzy inspection images into crystal-clear truth, revealing defects before they become failures.”  
(Cite problem gravity: yield losses from missed defects, then solution overview.)

### Demo Flow (3 Slides):  
1. **Input vs. Output Visualization:** Show a side-by-side comparison: “Degraded Input → Our Restored Output → Ground Truth”. Highlight noise and blur gone. Use real sample from validation. Possibly overlay PSNR/SSIM numbers.  
2. **Architecture Snapshot:** A compact pipeline diagram (similar to HLA) with labels (Data → Model → Output). Emphasize the “All-in-One Network” handling both noise and super-res. Bullet points on loss design or tech (Preprocessing, RRDB blocks, Postprocess).  
3. **Performance Metrics:** Bar chart or table of metrics. Example: “Baseline (bicubic) vs. Our Model: PSNR 28.5 vs 33.2 dB, SSIM 0.85 vs 0.93, Latency 0.8s (ours) vs 0.1s (bicubic).” Also show a demo snippet (e.g. a short screencast of running `infer.py` on H100, or use Streamlit interface). The point: we deliver **high quality + speed**.  

**KPIs to show live:**  
- **Inference Time:** Run our `infer.py` on 5 test images and display average time (target <5s/image).  
- **Quality Metrics:** Compute PSNR/SSIM on a held-out sample and display. (A live table in the UI or as console output.)  
- **Visual Clarity:** Include a canvas where user can toggle input/restored view. Judges love visual evidence: e.g. toggle on a small GUI.  
- If possible, **failure mode** example: Show one very noisy input that baseline misses vs. our result catching a defect. (To illustrate impact.)  

### Impact Story (Narrative):  
Frame the solution in terms of cost savings and reliability. “Our method ensures that the *evidence for a defect* is preserved. By injecting domain-specific knowledge (speckle vs Gaussian noise) into the model, we avoid the trap of “super-resolution without evidence” that recent research warns against. Judges care about scale: emphasize that a 30% increase in detectable defects translates to dozens of improved yields per wafer lot. In the pitch, quantify: e.g. “A single defect avoidance might save ~$X$ per wafer; if our tech raises defect-detection by Y%, this could save millions annually.” (Use generic numbers or past industry case studies if known.) Stress the national angle: better domestic fab reliability (align to “Make in India”/ SEMICON goals).  

This story ties our technical work to real-world impact on semiconductor manufacturing.

## 9. Coding Prompts (for ChatGPT / Script Generation)  

- **Prompt 1 – Training Script (`train.py`):**  
  *“Write a Python training script using PyTorch that loads paired grayscale images from directories `--lr_dir` and `--gt_dir`. The script should define a convolutional neural network (RRDB-based) to perform 2× super-resolution and denoising. Include argument parsing for `--epochs`, `--batch_size`, and `--save_path`. Use a composite loss (L1 + SSIM) and output PSNR/SSIM per epoch. Show an example command to run training.”*  
  *Example CLI:*  
  ```bash
  python train.py \
    --lr_dir ./data/train/NoisyLR \
    --gt_dir ./data/train/GT \
    --epochs 50 \
    --batch_size 8 \
    --save_path ./weights/model_best.pt
  ```

- **Prompt 2 – Inference Pipeline:**  
  *“Write a Python inference script (`infer.py`) that loads a trained PyTorch model from `--weights`. It should take `--input_dir` of degraded `.npy` images, run the model to restore each, and save the outputs in `--output_dir`. Include timing code to report average inference time per image. Provide the command usage example.”*  
  *Example CLI:*  
  ```bash
  python infer.py \
    --input_dir /path/to/test/NoisyLR \
    --output_dir ./restored \
    --weights ./weights/model_best.pt
  ```

- **Prompt 3 – Data Loader and Augmentation:**  
  *“Write a PyTorch `Dataset` class that loads pairs of `.npy` images (noisy low-res and GT high-res). Include random horizontal/vertical flips and 90° rotations. Normalize images to [0,1]. Show how to integrate this with a `DataLoader`.”*

- **Prompt 4 – Dockerfile Setup:**  
  *“Write a Dockerfile to run the inference API. Use a PyTorch CUDA base image, copy the current directory, install `requirements.txt`, and set the entrypoint to run `uvicorn main:app --host 0.0.0.0`. Include environment variable for half-precision (e.g., `USE_FP16`).”*  
  *Snippet:*  
  ```dockerfile
  FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- **Prompt 5 – Minimal Web Demo (Streamlit):**  
  *“Write a short Streamlit app that allows uploading a low-res noisy image (128×128), calls the model to restore it, and displays input/output side by side. Include a button to trigger inference and show PSNR.”*  
  *Minimal example:*  
  ```python
  import streamlit as st
  from PIL import Image
  st.title("Wafer Image Restorer")
  img = st.file_uploader("Upload noisy image", type=["png","jpg"])
  if img:
    input_image = preprocess(img)  # convert to model input
    output_image = model(input_image)
    st.image([input_image, output_image], caption=["Input","Restored"])
  ```

These prompts cover training, inference, data pipeline, and deployment, with commands and code that can be fed to a coding AI or used as boilerplate.

## 10. Resources & Prioritised Sources  

### Official/Hackathon Sources:  
- **SEMICON India Hackathon (i4C)** – official problem statement and requirements.  
- **KLA Webinar & Guidelines** – outlines technical context and metrics (Slides on SSIM/PSNR).  

### Key Papers (last 5 years):  
- Wang *et al*., **“Real-ESRGAN: Training Real-World Blind SR…”** (2021) – Practical model for real-world SR on complex noise. (Impl. code [35])  
- Liang *et al*., **“SwinIR: Image Restoration Using Swin Transformer”** (2021) – Transformer-based state-of-art on denoise & SR.  
- Yang *et al*., **“Does Super-Resolution Preserve Defect Evidence?”** (arXiv 2026) – Semiconductor-specific benchmark; highlights metric vs. task tradeoffs.  
- Moon *et al*., **“Relaxed Noise2Noise for SEM Denoising”** (M&M 2025) – SEM denoising via Noise2Noise variants; discusses training cost & generalization.  
- Wu *et al*., **“Denoising Diffusion for Wafer Defects”** (Mathematics, 2024) – Using diffusion models for wafer defect augmentation.  
- Lehtinen *et al*., **“Noise2Noise”** (2018) – foundational unsupervised denoising; our Relaxed Noise2Noise extends this concept.  

### Open-Source Repos to Prioritise:  
- **xinntao/Real-ESRGAN** (GitHub) – official Real-ESRGAN code.  
- **Lornatang/ESRGAN-PyTorch** – PyTorch reimplementation of ESRGAN (uses RRDB).  
- **BasicSR** – repository containing implementations of many SR and restoration models (EDSR, RCAN, etc.) (see [35] recommended projects).  
- **RuotianZhou/SwinIR** – code for the SwinIR transformer model (official).  
- **IJCAI-2025** repositiories (e.g. “WaferDC”) – example of long-tailed defect detection on wafer images, useful for understanding image patterns.  
- **OpenCV / SciPy** – for classical denoising (nonlocal means, bilateral filter) as fallbacks.  
- **PyTorch Lightning / HuggingFace Accelerate** – boilerplate for rapid training loops, especially under time pressure.  

These sources (peer-reviewed papers and reputable repos) provide the foundations and tools needed. We also recommend checking recent **IEEE/ACM conferences (CVPR, ICCV, NeurIPS)** for any novel diffusion-based restoration work that may appear. All cited information in this report is drawn from these references or official hackathon materials.  

