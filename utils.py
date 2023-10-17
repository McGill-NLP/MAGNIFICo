import os
import sqlite3
import pdb

class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))

def get_cols(q, parse):
	db = q.split(":")[0]
	conn = sqlite3.connect('spider/database/' + db + '/' + db + '.sqlite')
	conn.create_aggregate("stdev", 1, StdevFunc)
	c = conn.cursor()
	try:
		parse.replace("len(", "length(")
		parse.replace("LEN(", "LENGTH(")
		curs = c.execute(parse)
		cols = [description[0] for description in curs.description]
	except Exception as e:
		print(e)
		return "ERROR!"
	conn.commit()
	conn.close()
	return cols

def get_op_sql(q, parse):
	db = q.split(":")[0]
	conn = sqlite3.connect('spider/database/' + db + '/' + db + '.sqlite')
	c = conn.cursor()
	try:
		op = c.execute(parse).fetchall()
	except Exception as e:
		# print(e)
		return "ERROR!"
	conn.commit()
	conn.close()
	return op

def eval_sql(q, gold_parse, pred_parse):
	gold_op = get_op_sql(q, gold_parse)
	pred_op = get_op_sql(q, pred_parse)
	ex = 1
	em = 0
	if pred_op == "ERROR!":
		ex = 0
	elif len(pred_op) != len(gold_op):
		ex = 0
	else:
		for row in range(len(pred_op)):
			if set(pred_op[row]) != set(gold_op[row]):
				ex = 0
				break
	if gold_parse.strip().replace(" ", "") == pred_parse.strip().replace(" ", ""):
		em = 1
	return ex, em

def get_creates(db):
	all_text = ""
	for file in os.listdir('spider/database/' + db):
		if file.endswith(".sql"):
			with open('spider/database/' + db + '/' + file, 'r', encoding="utf-8") as f:
				sql_text = f.readlines()
			for l in sql_text:
				all_text = all_text + l
	if all_text == "":
		return []
	ls_cmd = all_text.split(";")
	cr_ls = []
	for cmd in ls_cmd:
		if "create table" in cmd.lower():
			temp_cmd = ""
			if "CREATE" in cmd:
				temp_cmd = "CREATE " + cmd.split("CREATE")[1].strip()
			elif "create" in cmd:
				temp_cmd = "create " + cmd.split("create")[1].strip()
			elif "Create" in cmd:
				temp_cmd = "Create " + cmd.split("Create")[1].strip()
			cr_ls.append(temp_cmd)
	return cr_ls

def get_relevant_tabs(op):
	tabs = []
	tokens = op.split()
	for i in range(len(tokens)):
		if tokens[i] == "from":
			if tokens[i+1] not in tabs:
				tabs.append(tokens[i+1])
	return tabs

def prompt_prefix(ques, ex_limit = 3, add_instr=False, instr="", instr_pos="end", parse="", postfix=True):
	db = ques.split(":")[0]
	try:
		conn = sqlite3.connect('spider/database/' + db + '/' + db + '.sqlite')
	except:
		print("Error at: ", db)
	c = conn.cursor()
	create_ls = get_creates(db)
	prefix = ""
	rel_tabs = get_relevant_tabs(parse)
#     print(rel_tabs)
	if add_instr and instr_pos == "top":
		prefix = prefix + instr + "\n\n"
	if len(create_ls) != 0:
		for loop, cr in enumerate(create_ls):
			table = cr.split("(")[0].split()[-1].strip().replace("\"", "")
			
			cols = get_cols(ques, "SELECT * FROM " + table + ";")
			cols_text = ""
			for col in cols:
				cols_text = cols_text + col + "\t"
			cols_text = cols_text.strip()
			
			if ex_limit == -1:
#                 print(table.replace("`", ""))
				if table.replace("`", "") in rel_tabs:
					prefix = prefix + cr + "\n\n"
			elif ex_limit == 0:
				prefix = prefix + cr + "\n\n"
			else:
				sel_query = "SELECT * FROM " + table + " LIMIT " + str(ex_limit) + ";"
				sel = get_op_sql(ques, sel_query)
				sel_text = ""
				for row in sel:
					row_text = ""
					for entry in row:
						row_text = row_text + str(entry) + "\t"
					row_text = row_text.strip()
					sel_text = sel_text + row_text + "\n"
				sel_text = sel_text.strip()
				prefix = prefix + cr + "\n" + "/*\n" + str(ex_limit) + " example rows:\n" + sel_query + "\n" \
							+ cols_text + "\n" + sel_text + "\n*/\n\n"
			if add_instr and instr_pos == "mid":
				if len(create_ls) == 1:
					prefix = prefix + instr + "\n\n"
				elif loop == int(len(create_ls)/2)-1:
					prefix = prefix + instr + "\n\n"
	else:
		tab_ls = []
		for temp_t in ques.split("|")[1:]:
			tab_ls.append(temp_t.split(":")[0].strip())
		for loop, table in enumerate(tab_ls):
			cols = get_cols(ques, "SELECT * FROM " + table + ";")
			cols_text = ""
			for col in cols:
				cols_text = cols_text + col + "\t"
			cols_text = cols_text.strip()
			
			if ex_limit == -1:
				if table.replace("`", "") in rel_tabs:
					prefix = prefix + "/*\nSELECT * FROM " + table + ";\n" + cols_text + "\n*/\n\n"
			elif ex_limit == 0:
				prefix = prefix + "/*\nSELECT * FROM " + table + ";\n" + cols_text + "\n*/\n\n"
			else:            
				sel_query = "SELECT * FROM " + table + " LIMIT " + str(ex_limit) + ";"
				sel = get_op_sql(ques, sel_query)
				sel_text = ""
				for row in sel:
					row_text = ""
					for entry in row:
						row_text = row_text + str(entry) + "\t"
					row_text = row_text.strip()
					sel_text = sel_text + row_text + "\n"
				sel_text = sel_text.strip()
				prefix = prefix + "/*\n" + str(ex_limit) + " example rows of table " + table + ":\n" + sel_query \
							+ "\n" + cols_text + "\n" + sel_text + "\n*/\n\n"
			if add_instr and instr_pos == "mid":
				if len(tab_ls) == 1:
					prefix = prefix + instr + "\n\n"
				elif loop == int(len(tab_ls)/2)-1:
					prefix = prefix + instr + "\n\n"
	
	if postfix:
		prefix = prefix + "-- Using valid SQLite, answer the following questions for the tables provided above."
	
	if add_instr and instr_pos=="end":
		prefix = prefix + "\n\n-- " + instr
	
	return prefix

def get_dialog(convo_file):
	with open(convo_file, 'r', encoding="utf-8") as file:
		dialog = file.read()
	return dialog.strip()

def get_prompt(ques, prompt_type, instr="", df=None, ex_limit=3, instr_pos="end", parse="", convo_path=None):
	if prompt_type == "direct":
		prefix = prompt_prefix(ques, ex_limit, parse=parse)
		prompt = prefix + "\n\n-- " + ques.split("|")[0].split(":")[1].strip() + "\nSELECT"
	elif prompt_type == "instr":
		prefix = prompt_prefix(ques, ex_limit, add_instr=True, instr=instr, instr_pos=instr_pos, parse=parse)
		prompt = prefix + "\n\n-- " + ques.split("|")[0].split(":")[1].strip() + "\nSELECT"
	elif prompt_type == "5-shot":
		prefix = prompt_prefix(ques, ex_limit, parse=parse)
		prompt = prefix
		for i in range(len(df)):
			ex = df.loc[i]
			t_q = ex['Question']
			t_parse = ex['Parse']
			prompt = prompt + "\n\n-- " + t_q.split("|")[0].split(":")[1].strip() + "\n" + t_parse
		prompt = prompt + "\n\n-- " + ques.split("|")[0].split(":")[1].strip() + "\nSELECT"
	elif prompt_type == "instr-5-shot":
		prefix = prompt_prefix(ques, ex_limit, add_instr=True, instr=instr, instr_pos=instr_pos, parse=parse)
		prompt = prefix
		for i in range(len(df)):
			ex = df.loc[i]
			t_q = ex['Question']
			t_parse = ex['Parse']
			prompt = prompt + "\n\n-- " + t_q.split("|")[0].split(":")[1].strip() + "\n" + t_parse
		prompt = prompt + "\n\n-- " + ques.split("|")[0].split(":")[1].strip() + "\nSELECT"
	elif prompt_type == "dialogue":
		prefix = prompt_prefix(ques, ex_limit, parse=parse, postfix=False)
		db = ques.split(":")[0]
		dialog = get_dialog(convo_path + "/" + db + ".txt")
		prompt = dialog + "\n\n" + "User1: Suppose you are given the following schema:\n\n" + prefix + "Using valid SQLite, answer the following question with the corresponding SQL query:\n" + ques.split("|")[0].split(":")[1].strip() + "\n\nUser2: SELECT"
	return prompt

def run_eval(df, model, prompt_type, instr="", train_df=None, ex_limit=3, instr_pos="end", convo_path=None):
	ls = []
	ex_acc = 0
	em_acc = 0
	for i in range(len(df)):
		# if i == 14:
		# 	continue
		ex = df.loc[i]
		ques = ex['Question']
		gold_parse = ex['Parse']
		prompt = get_prompt(ques, prompt_type, instr, train_df, ex_limit, instr_pos, gold_parse, convo_path)
#         prompt = get_prompt(ques, prompt_type, instr, train_df, ex_limit)
		temp_lim = ex_limit
		done = False
		con_flag = False
	
		while not done:
#             print(prompt)
#             print()
			try:
				response = model.predict(prompt)
				done = True
			except:
				temp_lim = temp_lim - 1
				print("Need to decrease limit to " + str(temp_lim))
				if temp_lim == -2:
					print("THIS IS TOO LONG!! " + ques)
					con_flag = True
					break
				prompt = get_prompt(ques, prompt_type, instr, train_df, temp_lim, instr_pos, gold_parse, convo_path)
				
		if con_flag:
			continue
			
		pred_parse = "SELECT " + response.strip()

		ex, em = eval_sql(ques, gold_parse, pred_parse)
		ex_acc += ex
		em_acc += em
		
		ls.append([ques, pred_parse.replace("\n", " ").replace("\t", " "), gold_parse, ex])
		print("Completed {} / {}...".format(i+1, len(df)), end = '\r', flush = True)
	return ex_acc/len(df), em_acc/len(df), ls

def run_single_eval(df, model, start_idx, end_idx, prompt_type, instr="", train_df=None, ex_limit=3, instr_pos="end", convo_path=None):
	ls = []
	ex_acc = 0
	em_acc = 0
	for i in range(start_idx, end_idx):
		ex = df.loc[i]
		ques = ex['Question']
		gold_parse = ex['Parse']
		prompt = get_prompt(ques, prompt_type, instr, train_df, ex_limit, instr_pos, gold_parse, convo_path)
#         prompt = get_prompt(ques, prompt_type, instr, train_df, ex_limit)
		temp_lim = ex_limit
		done = False
		con_flag = False
	
		while not done:
#             print(prompt)
#             print()
			try:
				response = model.predict([prompt])[0]
				done = True
			except:
				temp_lim = temp_lim - 1
				print("Need to decrease limit to " + str(temp_lim))
				if temp_lim == -2:
					print("THIS IS TOO LONG!! " + ques)
					con_flag = True
					break
				prompt = get_prompt(ques, prompt_type, instr, train_df, temp_lim, instr_pos, gold_parse, convo_path)
				
		if con_flag:
			continue
			
		pred_parse = "SELECT " + response.strip()

		ex, em = eval_sql(ques, gold_parse, pred_parse)
		ex_acc += ex
		em_acc += em
		
		ls.append([ques, pred_parse.replace("\n", " ").replace("\t", " "), gold_parse, ex])
		# print("Completed {} / {}...".format(i+1, len(df)), end = '\r', flush = True)
	return ex_acc, em_acc, ls

def run_batch_eval(df, model, prompt_type, batch_size=2, instr="", train_df=None, ex_limit=3, instr_pos="end", convo_path=None):
	ls = []
	ex_acc = 0
	em_acc = 0
	bnum = 0
	for i in range(0, len(df), batch_size):
		start_idx = i
		end_idx = min(i+batch_size, len(df))
		batch_prompts = []
		questions = []
		gold_parses = []
		for j in range(start_idx, end_idx):
			ex = df.loc[j]
			ques = ex['Question']
			gold_parse = ex['Parse']
			prompt = get_prompt(ques, prompt_type, instr, train_df, ex_limit, instr_pos, gold_parse, convo_path)
			batch_prompts.append(prompt)
			questions.append(ques)
			gold_parses.append(gold_parse)

		try:
			batch_responses = model.predict(batch_prompts)
			for k in range(len(batch_responses)):
				response = batch_responses[k]
				ques = questions[k]
				gold_parse = gold_parses[k]

				pred_parse = "SELECT " + response.strip()

				ex, em = eval_sql(ques, gold_parse, pred_parse)
				ex_acc += ex
				em_acc += em
				
				ls.append([ques, pred_parse.replace("\n", " ").replace("\t", " "), gold_parse, ex])
		except Exception as e:
			print("Error in batch processing: ", e)
			# pdb.set_trace()
			exs, ems, temp_ls = run_single_eval(df, model, start_idx, end_idx, prompt_type, instr, train_df, ex_limit, instr_pos, convo_path)
			ex_acc += exs
			em_acc += ems
			ls.extend(temp_ls)

		bnum += 1
			
		print("Completed {} / {}...".format(bnum, len(df)//batch_size), end = '\r', flush = True)
	return ex_acc/len(df), em_acc/len(df), ls