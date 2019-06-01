from siamese import Siamese
s_net = Siamese()
s_net.build('vggface')
s_net.run_hyperas_experiment()