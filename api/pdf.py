import base64
from typing import Union

from fastapi import APIRouter, UploadFile

from ai.resume_parser import ResumeParser
from helper import get_layout

router = APIRouter()


def get_pdf_data(file):
    with open(file, "rb") as pdf:
        pdf_bytes = pdf.read()
        pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
    return pdf_data


@router.post("/file/parse")
async def parse_resume_file(
    pdf: Union[UploadFile, None] = None,
):
    pdf_bytes = await pdf.read()

    collated_resume_layout, first_page_cv2 = get_layout(pdf_bytes)

    resume_parser = ResumeParser(collated_resume_layout, first_page_cv2)

    profile_picture = resume_parser.get_profile_image()
    experience = resume_parser.get_experience()
    education = resume_parser.get_education()
    skills = resume_parser.get_skills()
    contact = resume_parser.get_contact()
    name = resume_parser.get_name()
    objective = resume_parser.get_objective()

    result = {
        "skills": skills,
        "contact": contact,
        "summary": objective,
        "personal": {**name, **profile_picture},
        "education": education,
        "experience": experience,
    }

    return result
