class UnifiedThyroidSystem:
    def __init__(self, blood, thermal, ultrasound):
        self.blood = blood
        self.thermal = thermal
        self.ultrasound = ultrasound

    def predict(self, x, modality):
        if modality == "blood":
            return self.blood(x)
        if modality == "thermal":
            return self.thermal(x)
        if modality == "ultrasound":
            return self.ultrasound(x)
        raise ValueError("Invalid modality")
