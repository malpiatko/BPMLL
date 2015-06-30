import argparse, subprocess, os.path, csv


def extract(args):
	with open(args.features, 'r') as f_file:
		f_reader = csv.reader(f_file, delimiter=';')
		classes = next(f_reader)[1:]
		for f_line in f_reader:
			instance_name = os.path.splitext(f_line[0])[0]
			file_name = os.path.join(args.corpus, instance_name + ".wav")
			values = f_line[1:]
			make_args(classes, values)
			print (".", end="", flush=True)
			call_string = "SMILExtract -C {} -I {} -O {} -instname {} -arfftargetsfile {} -loglevel 1 ".format(
				args.config, file_name, args.output, instance_name, args.arfftargets)
			try:
				args_array = (call_string + make_args(classes, values)).split(" ")
				subprocess.check_call(args_array, stdout=subprocess.DEVNULL)
			except subprocess.CalledProcessError as e:
				print (e.output)
				return 1
	return 0

def make_args(classes, values):
	return " ".join([make_flag(c, v) for c, v in zip(classes, values)])

def make_flag(c, v):
	return "-" + c + " " + v

def silent_remove(output):
	try:
		os.remove(output)
	except OSError:
		pass



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('corpus', type=str, help="Path to the corpus")
	parser.add_argument('config', type=str, help="Path to the config file in format required by openSMILE")
	parser.add_argument('-f', '--features', type=str, help="Path to the csv file containing classes")
	parser.add_argument('-o', '--output', type=str, default="output.arff", help="Path to the output arff file")
	parser.add_argument('--arfftargets', type=str, default="arff_targets_gui_labels.conf.inc", help="Path to the config file with the labels")
	args = parser.parse_args()
	silent_remove(args.output)
	ret = extract(args)
	exit(ret)

