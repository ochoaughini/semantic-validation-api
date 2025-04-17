from pydantic import BaseModel

class ValidationRequest(BaseModel):
    input_text: str
    reference_text: str
