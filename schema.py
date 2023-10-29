from pydantic import BaseModel


class PdfSchema(BaseModel):
    pdf: str
    score: float = 0.7
    scale: int = 2
