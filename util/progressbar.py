import time, sys
import numpy as np
import datetime
class progressbar:
    def __init__(self, minV=0, maxV=100, barLength = 50):
        self.minV = minV
        self.maxV = maxV
        self.barLength = barLength
        self.persent = 0
    @staticmethod
    def format_time(seconds):
        'Formats time as the string "HH:MM:SS".'
        return str(datetime.timedelta(seconds=int(seconds)))

    def start(self):
        self.start_time = time.time()
    def finish(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
        self.finish = True
    def __get_persent__(self, progress):
        self.status = ""
        if progress < self.minV:
            self.status = "Halt..."
            return 0
        if progress > self.maxV:
            self.status = "Done..."
            return 1
        return (progress-self.minV) / (self.maxV - self.minV)

    def update_progress(self, progress, prefix_message='', suffix_message=''):
        self.persent = self.__get_persent__(progress)
        block = int(round(self.persent*self.barLength))
        sys.stdout.write(str(block))
        sys.stdout.flush()

        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        if self.persent == 0:
            ela_fomat = '_:__:__'
            eta_fomat = '_:__:__'
        else:
            eta_time = elapsed_time * (1-self.persent) / self.persent
            ela_fomat = self.format_time(elapsed_time)
            eta_fomat = self.format_time(eta_time)

        text = "\r{6}: [{0}] {1}% {2} elapsed:{3} ETA:{4} {5}".format("#" * block + "-" * (self.barLength - block),
                                                                      round(self.persent * 100, 2), self.status, ela_fomat,
                                                                      eta_fomat, suffix_message, prefix_message)
        sys.stdout.write(text)
        sys.stdout.flush()

if __name__ == '__main__':
    bar = progressbar(-100, 100)
    bar.start()
    for i in range(-23, 200):
        time.sleep(0.1)
        bar.update_progress(i-100,'Train', 'loss1:{} loss2:{}'.format(np.round(np.random.rand(),3), np.round(np.random.rand(1), 3)))
    bar.finish()