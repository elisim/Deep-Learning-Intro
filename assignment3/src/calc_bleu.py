from nltk.translate.bleu_score import corpus_bleu
import itertools 

def to_ref(fpath):
	with open(fpath, 'r') as f:
		res = f.read().splitlines()
		res = [line.rstrip(' `').split() for line in res if line]
	return res

def to_can(fpath):
	res = to_ref(fpath)
	res = list(itertools.chain.from_iterable(res))
	return res 


def main():
    # sys.argv[1] is actual lyrics path
    # sys.argv[1] is generated lyrics path
	ref, can = sys.argv[1], sys.argv[2]
	ref = to_ref(ref)
	can = to_can(can)
	print(corpus_bleu([ref], [can]))
