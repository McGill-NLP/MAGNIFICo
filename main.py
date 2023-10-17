import json
import os
import argparse
import datetime
import time
import pdb

import openai

import pandas as pd
import torch
import random

from models import LargeLanguageModel
from utils import run_eval, run_batch_eval
from mappings import get_instruction, get_setting_word

data_path = 'magnifico_data'
# settings = ['baseline', 'plausible', 'nonsense', 'adversarial']
# settings = ['baseline']
# prompt_types = ['direct', 'instr', '5-shot']
# prompt_types = ['direct', '5-shot']
# prompt_types = ['direct']
# instr_positions = ['end']

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Evaluate LLMs on MAGNIFICo')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='results/', help='Output Directory')
	parser.add_argument('-api_key', type=str, default='ai2', help='Which OpenAI API Key to use')
	parser.add_argument('-stop', type=list, default=[';', '\n\n', '--', "#"], help='When to stop generation')
	parser.add_argument('-interpretations', type=str, default='all', help='interpretations')
	parser.add_argument('-settings', type=str, default='baseline', help='settings')
	parser.add_argument('-prompt_types', type=str, default='direct', help='prompt types')
	parser.add_argument('-instr_positions', type=str, default='end', help='Position where to place instruction')
	parser.add_argument('-model_type', type=str, default='huggingface', choices=['completion', 'chat', 'cohere', 'huggingface', 'tgi'], help='Which type of model to use')
	parser.add_argument('-hf_model_type', type=str, default='llama', choices=['causal', 'llama', 'seq_to_seq'], help='Which type of hf model to use')
	parser.add_argument('-llama_weights_path', type=str, default='/home/mila/a/arkil.patel/scratch', help='Outer directory where hf converted LLaMA weights are kept') # /home/toolkit/llama_weights_hf
	parser.add_argument('-model', type=str, default='llama7b', help='Which model to use')
	parser.add_argument('-port', type=int, default=8080, help='Port on which the model is hosted')
	parser.add_argument('-timeout', type=int, default=1000000, help='Timeout for the model')
	parser.add_argument('-batch_size', type=int, default=1, help='Batch Size')
	parser.add_argument('-max_tokens', type=int, default=128, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-combi', dest='combi', action='store_true', help='Combinations')
	parser.add_argument('-no-combi', dest='combi', action='store_false', help='Not Combinations')
	parser.set_defaults(combi=False)

	return parser

def run_model(model, path, prompt_type, batch_size=1, instr="", ex_limit=3, instr_pos="end", convo_path=None):
	test_df = pd.read_csv(path + "/test.tsv", sep='\t')
	train_df = pd.read_csv(path + "/train.tsv", sep='\t')

	if model.model_type != 'huggingface':
		return run_eval(test_df, model, prompt_type, instr, train_df, ex_limit, instr_pos, convo_path)
	
	return run_batch_eval(test_df, model, prompt_type, batch_size, instr, train_df, ex_limit, instr_pos, convo_path)

def run_combi_model(model, test_df, train_df, prompt_type, instr="", ex_limit=3, instr_pos="end", convo_path=None):
	if model.model_type != 'huggingface':
		return run_eval(test_df, model, prompt_type, instr, train_df, ex_limit, instr_pos, convo_path)
	
	return run_batch_eval(test_df, model, prompt_type, 1, instr, train_df, ex_limit, instr_pos, convo_path)

def get_replaced_df(concept_subset, con1, con2, setting="plausible"):
	con1_rep = get_setting_word(con1, setting)
	con2_rep = get_setting_word(con2, setting)
	new_ls = []
	for i in range(len(concept_subset)):
		q = concept_subset.loc[i]['Question']
		p = concept_subset.loc[i]['Parse']
		new_ls.append([q.replace("concept_word1", con1_rep).replace("concept_word2", con2_rep), p])
	replaced_df = pd.DataFrame(new_ls, columns=["Question", "Parse"])
	
	return replaced_df 

def main(args):
	model = LargeLanguageModel(
		model_type=args.model_type,
		engine=args.model,
		hf_model_type=args.hf_model_type,
		llama_weights_path=args.llama_weights_path,
		max_tokens=args.max_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		n=args.n,
		stop=args.stop,
		presence_penalty=args.presence_penalty,
		frequency_penalty=args.frequency_penalty,
		port=args.port,
		timeout=args.timeout,
	)

	settings = args.settings
	prompt_types = args.prompt_types
	instr_positions = args.instr_positions

	results_ls = []
	num_ex = 0

	if args.interpretations == 'all':
		interpretations_ls = [interp for interp in os.listdir(data_path) if interp not in ['len_less_than','outlier_range','below_average','hire_date','not_intern','dock_count','combination']]
	elif args.interpretations == 'phrase':
		interpretations_ls = ['len_less_than','outlier_range','below_average','hire_date','not_intern','dock_count']
	else:
		interpretations_ls = [ele for ele in args.interpretations.split(",")]

	if args.combi:
		baseline_df = pd.read_csv(data_path + "/combination/baseline.tsv", sep='\t')
		concept_df = pd.read_csv(data_path + "/combination/concept.tsv", sep='\t')
		con1_ls = list(concept_df['Interpretation1'].unique())
		con2_ls = list(concept_df['Interpretation2'].unique())
		for con1 in con1_ls:
			for con2 in con2_ls:
				ls_idxs = concept_df.index[(concept_df['Interpretation1'] == con1) & (concept_df['Interpretation2'] == con2)].tolist()
				if len(ls_idxs) == 0:
					continue
				concept_subset = concept_df[(concept_df['Interpretation1'] == con1) & (concept_df['Interpretation2'] == con2)].reset_index(drop=True)
				baseline_subset = baseline_df.iloc[ls_idxs].reset_index(drop=True)

				# baseline
				print("Working on Baseline")
				train1 = pd.read_csv(data_path + "/" + con1 + "/baseline/train.tsv", sep='\t')
				train1 = train1.iloc[[0, 2, 4]].reset_index(drop=True)
				train2 = pd.read_csv(data_path + "/" + con2 + "/baseline/train.tsv", sep='\t')
				train2 = train2.iloc[[0, 2, 4]].reset_index(drop=True)
				train_df = pd.concat([train1, train2], ignore_index=True)
				test_df = baseline_subset
				print("Direct")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="direct", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'baseline', 'direct', 'end', ex_acc, em_acc])
				print("Few-shot")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="5-shot", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'baseline', '5-shot', 'end', ex_acc, em_acc])

				# plausible
				print("Working on Plausible")
				train1 = pd.read_csv(data_path + "/" + con1 + "/plausible/train.tsv", sep='\t')
				train1 = train1.iloc[[0, 2, 4]].reset_index(drop=True)
				train2 = pd.read_csv(data_path + "/" + con2 + "/plausible/train.tsv", sep='\t')
				train2 = train2.iloc[[0, 2, 4]].reset_index(drop=True)
				train_df = pd.concat([train1, train2], ignore_index=True)
				test_df = get_replaced_df(concept_subset, con1, con2, "plausible")
				print("Direct")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="direct", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'plausible', 'direct', 'end', ex_acc, em_acc])
				instr1 = get_instruction(con1, 'plausible')
				instr2 = get_instruction(con2, 'plausible')
				instr = instr1 + " " + instr2
				print("Instruction")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="instr", instr=instr, ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'plausible', 'instr', 'end', ex_acc, em_acc])
				print("Few-shot")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="5-shot", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'plausible', '5-shot', 'end', ex_acc, em_acc])

				# nonsense
				print("Working on Nonsense")
				train1 = pd.read_csv(data_path + "/" + con1 + "/nonsense/train.tsv", sep='\t')
				train1 = train1.iloc[[0, 2, 4]].reset_index(drop=True)
				train2 = pd.read_csv(data_path + "/" + con2 + "/nonsense/train.tsv", sep='\t')
				train2 = train2.iloc[[0, 2, 4]].reset_index(drop=True)
				train_df = pd.concat([train1, train2], ignore_index=True)
				test_df = get_replaced_df(concept_subset, con1, con2, "nonsense")
				print("Direct")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="direct", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'nonsense', 'direct', 'end', ex_acc, em_acc])
				instr1 = get_instruction(con1, 'nonsense')
				instr2 = get_instruction(con2, 'nonsense')
				instr = instr1 + " " + instr2
				print("Instruction")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="instr", instr=instr, ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'nonsense', 'instr', 'end', ex_acc, em_acc])
				print("Few-shot")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="5-shot", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'nonsense', '5-shot', 'end', ex_acc, em_acc])

				# adversarial
				print("Working on Adversarial")
				train1 = pd.read_csv(data_path + "/" + con1 + "/adversarial/train.tsv", sep='\t')
				train1 = train1.iloc[[0, 2, 4]].reset_index(drop=True)
				train2 = pd.read_csv(data_path + "/" + con2 + "/adversarial/train.tsv", sep='\t')
				train2 = train2.iloc[[0, 2, 4]].reset_index(drop=True)
				train_df = pd.concat([train1, train2], ignore_index=True)
				test_df = get_replaced_df(concept_subset, con1, con2, "adversarial")
				print("Direct")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="direct", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'adversarial', 'direct', 'end', ex_acc, em_acc])
				instr1 = get_instruction(con1, 'adversarial')
				instr2 = get_instruction(con2, 'adversarial')
				instr = instr1 + " " + instr2
				print("Instruction")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="instr", instr=instr, ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'adversarial', 'instr', 'end', ex_acc, em_acc])
				print("Few-shot")
				ex_acc, em_acc, outputs = run_combi_model(model, test_df, train_df, prompt_type="5-shot", instr="", ex_limit=3, instr_pos="end", convo_path=None)
				results_ls.append([con1, con2, len(test_df), 'adversarial', '5-shot', 'end', ex_acc, em_acc])
			
		results_df = pd.DataFrame(results_ls, columns=["Interpretation1", "Interpretation2", "Num Ex", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
		results_df.to_csv(args.out_dir + "/combi_results.tsv", sep='\t', index=None)

	elif prompt_types == 'dialogue':
		for concept in interpretations_ls:
			concept_path = os.path.join(data_path, concept)
			print("Concept: ", concept)
			for setting in os.listdir(concept_path):
				if setting in settings:
					print("Setting: ", setting)
					setting_path = os.path.join(concept_path, setting)

					prompt_type = prompt_types
					op_path = os.path.join(args.out_dir, "outputs")
					op_path = os.path.join(op_path, concept)
					op_path = os.path.join(op_path, setting)
					op_path = os.path.join(op_path, prompt_type)

					for instr_pos in instr_positions:
						start_time = time.time()
						print("Instruction Position: ", instr_pos)
						instr = ""

						if instr_pos in ['mid', 'end']:
							convo_path = os.path.join(concept_path, 'dialogue_data_' + instr_pos)
						else:
							convo_path = os.path.join(concept_path, 'dialogue_data')
						if setting == 'baseline':
							convo_path = os.path.join(convo_path, 'concept_edited')
						else:
							convo_path = os.path.join(convo_path, setting)

						ex_acc, em_acc, outputs = run_model(model=model, path=setting_path, prompt_type=prompt_type, batch_size=args.batch_size, instr=instr, instr_pos=instr_pos, convo_path=convo_path)
						results_ls.append([concept, setting, prompt_type, instr_pos, ex_acc, em_acc])
						results_df = pd.DataFrame(results_ls, columns=["Interpretation", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
						results_df.to_csv(args.out_dir + "/results.tsv", sep='\t', index=None)

						print("Execution Accuracy: ", str(ex_acc))
						print("-------------------------------------------")

						temp_op_path = os.path.join(op_path, instr_pos)
						if not os.path.exists(temp_op_path):
							os.makedirs(temp_op_path)
						op_data = pd.DataFrame(outputs, columns=["Question", "Predicted Parse", "Gold Parse", "Result"])
						op_data.to_csv(temp_op_path + '/predictions.tsv', sep='\t', index=None)
						end_time = time.time()
						time_taken = end_time - start_time
						print("Time Taken: " + str(time_taken) + " For " + str(len(outputs)) + " examples.")
						num_ex += len(outputs)

		print()
		print("Tested on a total " + str(num_ex) + " examples.")

		results_df = pd.DataFrame(results_ls, columns=["Interpretation", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
		results_df.to_csv(args.out_dir + "/results.tsv", sep='\t', index=None)
		
	else:
		for concept in interpretations_ls:
			concept_path = os.path.join(data_path, concept)
			print("Concept: ", concept)
			for setting in os.listdir(concept_path):
				if setting in settings:
					print("Setting: ", setting)
					setting_path = os.path.join(concept_path, setting)
					for prompt_type in prompt_types:
						print("Prompt Type: ", prompt_type)
						op_path = os.path.join(args.out_dir, "outputs")
						op_path = os.path.join(op_path, concept)
						op_path = os.path.join(op_path, setting)
						op_path = os.path.join(op_path, prompt_type)
						if 'instr' in prompt_type:
							if setting == 'baseline':
								continue
							for instr_pos in instr_positions:
								start_time = time.time()
								print("Instruction Position: ", instr_pos)
								instr = get_instruction(concept, setting)
								ex_acc, em_acc, outputs = run_model(model=model, path=setting_path, prompt_type=prompt_type, batch_size=args.batch_size, instr=instr, instr_pos=instr_pos, convo_path=None)
								results_ls.append([concept, setting, prompt_type, instr_pos, ex_acc, em_acc])
								results_df = pd.DataFrame(results_ls, columns=["Interpretation", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
								results_df.to_csv(args.out_dir + "/results.tsv", sep='\t', index=None)

								print("Execution Accuracy: ", str(ex_acc))
								print("-------------------------------------------")

								temp_op_path = os.path.join(op_path, instr_pos)
								if not os.path.exists(temp_op_path):
									os.makedirs(temp_op_path)
								op_data = pd.DataFrame(outputs, columns=["Question", "Predicted Parse", "Gold Parse", "Result"])
								op_data.to_csv(temp_op_path + '/predictions.tsv', sep='\t', index=None)
								end_time = time.time()
								time_taken = end_time - start_time
								print("Time Taken: " + str(time_taken) + " For " + str(len(outputs)) + " examples.")
								num_ex += len(outputs)
						else:
							start_time = time.time()
							instr_pos='end'
							instr = ''
							ex_acc, em_acc, outputs = run_model(model=model, path=setting_path, prompt_type=prompt_type, batch_size=args.batch_size, instr=instr, instr_pos=instr_pos, convo_path=None)
							results_ls.append([concept, setting, prompt_type, instr_pos, ex_acc, em_acc])
							results_df = pd.DataFrame(results_ls, columns=["Interpretation", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
							results_df.to_csv(args.out_dir + "/results.tsv", sep='\t', index=None)

							print("Execution Accuracy: ", str(ex_acc))
							print("-------------------------------------------")

							if not os.path.exists(op_path):
								os.makedirs(op_path)
							op_data = pd.DataFrame(outputs, columns=["Question", "Predicted Parse", "Gold Parse", "Result"])
							op_data.to_csv(op_path + '/predictions.tsv', sep='\t', index=None)
							end_time = time.time()
							time_taken = end_time - start_time
							print("Time Taken: " + str(time_taken) + " For " + str(len(outputs)) + " examples.")
							num_ex += len(outputs)

		print()
		print("Tested on a total " + str(num_ex) + " examples.")

		results_df = pd.DataFrame(results_ls, columns=["Interpretation", "Setting", "Prompt Type", "Instruction Position", "Execution Accuracy", "Exact Match Accuracy"])
		results_df.to_csv(args.out_dir + "/results.tsv", sep='\t', index=None)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	if args.api_key == "personal":
		openai.api_key = os.getenv("OPENAI_API_KEY_PERSONAL")
	elif args.api_key == "ai2":
		openai.api_key = os.getenv("MODELSEC")
	else:
		openai.api_key = args.api_key

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)
	
	device = torch.device("cuda:0")

	args.device = device

	str_settings = args.settings
	args.settings = [ele for ele in str_settings.split(",")]

	str_instr_positions = args.instr_positions
	args.instr_positions = [ele for ele in str_instr_positions.split(",")]

	str_prompt_types = args.prompt_types
	if str_prompt_types != "dialogue":
		args.prompt_types = [ele for ele in str_prompt_types.split(",")]

	main(args)