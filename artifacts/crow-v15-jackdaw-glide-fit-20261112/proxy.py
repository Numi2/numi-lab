"""One-bird jackdaw glide reference; never an implicit flapping-force model."""
import json
import math
from pathlib import Path

class JackdawGlideProxy:
    def __init__(self, parameters=None):
        self.parameters = parameters or json.loads(Path(__file__).with_name("fit.json").read_text())
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.minimum = self.parameters["minimum_data_speed_m_s"]
        self.maximum = self.parameters["maximum_data_speed_m_s"]

    def drag_newtons(self, airspeed_m_s, *, steady_glide, premoult):
        if not steady_glide or not premoult:
            raise ValueError("the reference is only fitted to premoult steady glide")
        if not math.isfinite(airspeed_m_s) or not self.minimum <= airspeed_m_s <= self.maximum:
            raise ValueError("airspeed is outside the measured glide envelope")
        return self.a * airspeed_m_s ** 2 + self.b / airspeed_m_s ** 2
