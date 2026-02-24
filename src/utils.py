import time

class FPSCounter:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        
        def get_fps(self):
            return 1 / (time.time() - self.start_time)
        