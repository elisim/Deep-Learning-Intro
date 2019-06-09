from src.siamese import Siamese
import numpy as np
s_net = Siamese()
s_net.build('vggface')
s_net.run_hyperas_experiment()