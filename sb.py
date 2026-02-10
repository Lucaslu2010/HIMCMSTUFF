class women():
    def __init__(self):
        self.lh = 0
        self.fsh = 0
        self.ostrogen = 0
        self.progestron = 0
        self.pregnent = 0
        self.ticking = 1
        self.beigan = 0
    def ticker(self):
        if not self.pregnent and self.ticking == 30 and self.beigan:
            self.pregnent = 1
            self.lh = 999
            self.fsh = 999
            self.ostrogen = 999
            self.progestron = 999
        elif self.ticking == 30:
            self.ticking = 1
        ticker += 1







#__main__
while True:
    beiganlema = input()
    women.ticker(beiganlema)
