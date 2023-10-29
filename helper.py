import base64
from collections import defaultdict

import fitz
import requests
from fitz_utils import ProcessedDoc


def get_resume_layout_analysis(pdf_data):
    payload = {"pdf": pdf_data, "score": 0.7, "scale": 2}  # default  # default
    url = "http://0.0.0.0:8080/layout/pdf/base64"
    res = requests.post(url, json=payload)
    result = res.json()
    return result


def get_layout(pdf_bytes):
    multipage_docs = fitz.open(stream=pdf_bytes)
    previous_page_height = 0
    previous_block_no = 0

    scale = 2
    collated_resume_layout = defaultdict(list)

    for i, page in enumerate(multipage_docs):
        temp_docs = fitz.open()
        temp_docs.insert_pdf(multipage_docs, from_page=i, to_page=i + 1)
        # docs2.save(f"temp/temp-{i}.pdf")

        pdf_bytes = temp_docs.write()
        pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")

        current_resume_layout = get_resume_layout_analysis(pdf_data)

        for key, value in current_resume_layout.items():
            if key != "line_ocr":
                for ocr in current_resume_layout[key]:
                    new_bbox = [
                        ocr["bbox"][0],
                        ocr["bbox"][1] + previous_page_height,
                        ocr["bbox"][2],
                        ocr["bbox"][3] + previous_page_height,
                    ]

                    collated_resume_layout[key].append({**ocr, "bbox": new_bbox})
            else:
                for ocr in current_resume_layout[key]:
                    new_bbox = [
                        ocr["bbox"][0],
                        ocr["bbox"][1] + previous_page_height,
                        ocr["bbox"][2],
                        ocr["bbox"][3] + previous_page_height,
                    ]
                    new_block_no = ocr["block_no"] + previous_block_no

                    collated_resume_layout[key].append(
                        {**ocr, "bbox": new_bbox, "block_no": new_block_no}
                    )

                previous_block_no += max(
                    map(lambda x: x["block_no"], current_resume_layout[key])
                )

        previous_page_height = previous_page_height + int(page.rect.height * scale)

    doc = ProcessedDoc(stream=pdf_bytes)
    page = doc[0]
    first_page_cv2 = page.get_opencv_img(scale=fitz.Matrix(2, 2))
    return collated_resume_layout, first_page_cv2
