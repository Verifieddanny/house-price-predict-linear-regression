from pydantic import BaseModel

class PredictionInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str 

class PredictionOutput(BaseModel):
    predicted_price: float