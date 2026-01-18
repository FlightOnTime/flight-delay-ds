from pydantic import BaseModel, field_validator


class FlightRequest(BaseModel):
    airline: str
    origin: str
    dest: str
    distance: float
    day_of_week: int
    flight_date: str
    crs_dep_time: int

    @field_validator('distance')
    @classmethod
    def validate_distance(cls, v):
        if v <= 0:
            raise ValueError('Distance deve ser positiva')
        return v

    @field_validator('day_of_week')
    @classmethod
    def validate_day(cls, v):
        if not 1 <= v <= 7:
            raise ValueError('DayOfWeek deve estar entre 1 e 7')
        return v
