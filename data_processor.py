import os
import re
import fitz
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from config import HEADER_MARGIN, FOOTER_MARGIN

ocr_engine = RapidOCR()


def advanced_clean_text(text):
	text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
	text = re.sub(r'(^|\n)([A-Z][^\n.]{1,60})(\n)', r'\1\2\n\n', text)
	text = re.sub(r'\n(?=\d+\.|•|-|\*)', '\n\n', text)
	text = re.sub(r'(?<![.:])\n(?!\n)', ' ', text)
	text = re.sub(r'[ \t]+', ' ', text)
	text = re.sub(r'\n\s+', '\n', text)
	return text.strip()


def extract_text_from_pdf(pdf_path):
	doc = fitz.open(pdf_path)
	extracted_docs = []

	for page_num, page in enumerate(doc):
		rect = page.rect
		clip_rect = fitz.Rect(0, HEADER_MARGIN, rect.width, rect.height - FOOTER_MARGIN)

		blocks = page.get_text("blocks", sort=True, clip=clip_rect)
		text_blocks = []
		for b in blocks:
			if b[4].strip(): text_blocks.append(b[4])
		text = "\n\n".join(text_blocks)

		image_list = page.get_images(full=True)
		if image_list:
			for img in image_list:
				try:
					xref = img[0]
					base_image = doc.extract_image(xref)
					image_bytes = base_image["image"]
					if len(image_bytes) < 2000: continue

					ocr_result, _ = ocr_engine(image_bytes)
					if ocr_result:
						image_text = " ".join([res[1] for res in ocr_result])
						if len(image_text.strip()) > 5:
							text += f"\n\n[Diagram Text]: {image_text}"
				except:
					pass

		cleaned_text = advanced_clean_text(text)
		if cleaned_text:
			final_content = f"***PAGE_START***\n\n{cleaned_text}"
			new_doc = Document(
				page_content=final_content,
				metadata={"source": os.path.basename(pdf_path), "page": page_num}
			)
			extracted_docs.append(new_doc)

	return extracted_docs
