import os, logging, socket
import numpy as np
import torch

class GPU:
    device = torch.device('cpu')

    @staticmethod
    def get_free_gpu(memory=1000):
        skinner_map = {0: 2, 1: 0, 2: 1, 3: 3}
        a = os.popen("/usr/bin/nvidia-smi | grep 'MiB /' | awk -e '{print $9}' | sed -e 's/MiB//'")

        free_memory = []
        while 1:
            line = a.readline()
            if not line:
                break
            free_memory.append(int(line))

        gpu = np.argmin(free_memory)
        if free_memory[gpu] < memory:
            if socket.gethostname() == "skinner":
                for k, v in skinner_map.items():
                    if v == gpu:
                        return k
            return gpu

        logging.error('No free GPU available.')
        exit(1)

    @classmethod
    def set(cls, gpuid, memory=1000):
        gpuid = int(gpuid)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logging.info("searching for free GPU")
            if gpuid == -1:
                gpuid = GPU.get_free_gpu(memory)
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)
            if torch.cuda.device_count() == 1:  # sometimes this does not work
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(gpuid))
        else:
            gpuid = os.environ['CUDA_VISIBLE_DEVICES']
            logging.info('taking GPU {} as specified in envorionment variable'.format(gpuid))
            torch.cuda.set_device(gpuid)

        cls.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

        logging.info('Using GPU {}'.format(gpuid))
        return gpuid
