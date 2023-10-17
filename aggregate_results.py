import pandas as pd
import argparse
import os
import pdb

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Aggregate Results')

	parser.add_argument('-model', type=str, default='starcoder', help='model name')
	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')

	parser.add_argument('-dialogue_only', dest='dialogue_only', action='store_true', help='Only dialogues')
	parser.add_argument('-no-dialogue_only', dest='dialogue_only', action='store_false', help='Not only dialogues')
	parser.set_defaults(dialogue_only=False)

	return parser

def remove_redundancies(df):
	# group by all columns except 'Execution Accuracy' and 'Exact Match Accuracy'
	# and get the maximum value of 'Execution Accuracy'
	max_exec_acc_df = df.groupby(['Interpretation', 'Setting', 'Prompt Type', 'Instruction Position'], as_index=False)['Execution Accuracy'].max()

	# merge the original dataframe with the new one that contains only the rows with max 'Execution Accuracy'
	result_df = pd.merge(max_exec_acc_df, df, on=['Interpretation', 'Setting', 'Prompt Type', 'Instruction Position', 'Execution Accuracy'], how='inner')

	result_df.drop_duplicates(inplace=True)

	return result_df

def main(args):
	path = args.out_dir

	ls = []
	for run in os.listdir(path):
		if not run.endswith(".tsv"):
			trun = run.replace("INCITE-7B-", "")
			if args.model in trun.lower():
				# pdb.set_trace()
				if ("instruct" in args.model and "instruct" in trun.lower()) or ("instruct" not in args.model and "instruct" not in trun.lower()):
					run_path = os.path.join(path, run)
					filepath = os.path.join(run_path, 'results.tsv')
					try:
						run_df = pd.read_csv(filepath, sep='\t')
					except Exception as e:
						continue
					run_dict = run_df.to_dict('records')
					ls.extend(run_dict)

	df = pd.DataFrame.from_dict(ls)

	if args.dialogue_only:
		df = df[df['Prompt Type'] == 'dialogue']

	final_df = remove_redundancies(df)

	if args.dialogue_only:
		final_df.to_csv(path + "/dialogue_" + args.model + ".tsv", sep='\t', index=None)
	else:
		final_df.to_csv(path + "/" + args.model + ".tsv", sep='\t', index=None)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	main(args)