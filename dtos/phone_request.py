from pydantic import BaseModel, Field
import numpy as np

class PhoneRequest(BaseModel):
    price: float = Field(..., gt=0, description="Precio en USD, debe ser > 0")
    battery: float = Field(..., gt=0, description="Capacidad de batería en mAh, debe ser > 0")
    ram: float = Field(..., gt=0, description="RAM en GB, debe ser > 0")
    storage: float = Field(..., gt=0, description="Almacenamiento en GB, debe ser > 0")
    camera: float = Field(..., gt=0, description="Resolución de cámara en MP, debe ser > 0")
    screen: float = Field(..., gt=0, description="Tamaño de pantalla en pulgadas, debe ser > 0")
    weight: float = Field(..., gt=0, description="Peso en gramos, debe ser > 0")

    def toArray(self):
        return np.array([[
            self.price,
            self.ram,
            self.storage,
            self.battery,
            self.camera,
            self.screen,
            self.weight
        ]])