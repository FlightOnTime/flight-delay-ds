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
    def validate_distance(cls, distance):
        if distance <= 0:
            raise ValueError('Distance must be positive')
        return distance

    @field_validator('day_of_week')
    @classmethod
    def validate_day(cls, day_of_week):
        if not 1 <= day_of_week <= 7:
            raise ValueError('DayOfWeek must be between 1 and 7')
        return day_of_week
