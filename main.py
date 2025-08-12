from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
from typing import Optional, List
from pdf2image import convert_from_bytes

app = FastAPI(
    title="Qwen2VL OCR API",
    description="API for PDF/Image-to-text extraction using Qwen2VL model",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Model loaded successfully")


def clean_output(text: str) -> str:
    """Clean model output by removing special tokens and formatting"""
    clean_text = text.replace("<|im_end|>", "").strip()
    clean_text = clean_text.replace("<|im_start|>", "").strip()
    clean_text = clean_text.replace("<|vision_start|>", "").strip()
    clean_text = clean_text.replace("<|image_pad|>", "").strip()
    clean_text = " ".join(clean_text.split())
    return clean_text


async def process_image_and_text_pil(
    img: Image.Image,
    text_prompt: Optional[str] = None
):
    """Process a PIL Image with optional text prompt"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text_prompt or "Extract all text and key information from this document in json format."},
                ],
            }
        ]

        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        return {
            "status": "success",
            "result": clean_output(generated_text)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/extract-text")
async def extract_text(
    file: UploadFile = File(..., description="PDF or image file to process"),
    pages: Optional[str] = Form(None, description="Comma-separated list of page numbers (1-indexed)"),
    text_prompt: Optional[str] = Form(None, description="Optional text prompt")
):
    """
    Extract text from specified pages of a PDF or from a single image.
    Pages are 1-indexed (e.g., '1,3,5').
    """
    filename = file.filename.lower()
    results = []

    try:
        if filename.endswith(".pdf"):
            file_bytes = await file.read()
            all_images = convert_from_bytes(file_bytes)

            # Determine pages to process
            if pages:
                page_numbers = [int(p.strip()) for p in pages.split(",") if p.strip().isdigit()]
                selected_pages = [all_images[p - 1] for p in page_numbers if 1 <= p <= len(all_images)]
            else:
                selected_pages = all_images  # process all pages if not specified

            # Process each page
            for i, page_img in enumerate(selected_pages, start=1):
                ocr_result = await process_image_and_text_pil(page_img, text_prompt)
                ocr_result["page"] = i
                results.append(ocr_result)

        else:
            # Single image case
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes))
            ocr_result = await process_image_and_text_pil(img, text_prompt)
            ocr_result["page"] = 1
            results.append(ocr_result)

        return JSONResponse(content={"status": "success", "results": results, "model": MODEL_ID})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "model": MODEL_ID,
        "device": DEVICE
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)