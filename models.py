import os
import time
import argparse
import pdb

import openai
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential
import tiktoken

# import cohere
# co = cohere.Client('PqX7WY8xt39eswu28aLukPw9FAMEQp5tt26cvdDq')

import torch
from typing import Optional

from pydantic import BaseModel
from transformers import (
	PreTrainedModel,
	PreTrainedTokenizer,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	AutoModelForCausalLM,
	LlamaForCausalLM,
	LlamaTokenizer,
	AutoModel
)

from text_generation import Client

class EvalModel(BaseModel, arbitrary_types_allowed=True):
	max_input_length: int = 512
	max_tokens: int = 512
	stop: list = [";"]

	def run(self, prompt: str) -> str:
		raise NotImplementedError

	def check_valid_length(self, text: str) -> bool:
		raise NotImplementedError


class SeqToSeqModel(EvalModel):
	model_path: str
	llama_weights_path: str
	model: Optional[PreTrainedModel]
	tokenizer: Optional[PreTrainedTokenizer]
	device: torch.device

	def load(self):
		if self.model is None:
			self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
			self.model.eval()
			self.model.to(self.device)
		if self.tokenizer is None:
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

	def run(self, prompt: str) -> str:
		self.load()
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

	def check_valid_length(self, text: str) -> bool:
		self.load()
		inputs = self.tokenizer(text)
		return len(inputs.input_ids) <= self.max_input_length


class CausalModel(SeqToSeqModel):
	def load(self):
		if self.model is None:
			if 'alpaca' in self.model_path:
				self.model = AutoModelForCausalLM.from_pretrained('/home/mila/a/arkil.patel/scratch/alpaca7b', device_map='auto', torch_dtype=torch.float16)
			elif 'starcoder' not in self.model_path:
				self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, cache_dir = '/home/mila/a/arkil.patel/scratch/transformers_cache')
			else:
				self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto', torch_dtype=torch.float16)
			self.model.eval()
			self.model.to(self.device)
		if self.tokenizer is None:
			if 'alpaca' in self.model_path:
				self.tokenizer = AutoTokenizer.from_pretrained('/home/mila/a/arkil.patel/scratch/alpaca7b', device_map='auto', torch_dtype=torch.float16)
			else:
				self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, torch_dtype=torch.float16)
		for param in self.model.parameters():
			param.requires_grad = False

	def run(self, prompt, temperature=0.0, num_return_sequences=1):
		# self.load()
		self.tokenizer.pad_token = self.tokenizer.bos_token
		self.tokenizer.padding_side = 'left'
		inputs = self.tokenizer(prompt, return_tensors="pt", return_length=True, padding=True).to(self.device)

		# try:
		with torch.no_grad():
			if temperature == 0.0:
				outputs = self.model.generate(
					inputs.input_ids,
					do_sample=False,
					num_return_sequences=1,
					max_new_tokens=self.max_tokens,
					pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
				)
			else:
				outputs = self.model.generate(
					inputs.input_ids,
					do_sample=True,
					temperature=temperature,
					num_return_sequences=num_return_sequences,
					max_new_tokens=self.max_tokens,
					pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
				)
		# except Exception as e:
		# 	print("Inside error: ", e)
		# 	pdb.set_trace()
		# 	del inputs
		# 	for param in self.model.parameters():
		# 		del param
		# 	torch.cuda.empty_cache()
		# 	raise Exception

		lengths = inputs.length
		maxlen = max(lengths)
		final_ops = []

		for iz in range(len(outputs)):
			final_op = self.tokenizer.decode(outputs[iz, maxlen:], skip_special_tokens=True)
			
			for stop_ele in self.stop:
				if stop_ele in final_op:
					final_op = final_op.split(stop_ele)[0]
					break

			final_ops.append(final_op)

		del inputs
		del outputs
		torch.cuda.empty_cache()

		return final_ops


class LlamaModel(SeqToSeqModel):
	use_template: bool = False
	"""
	Not officially supported by AutoModelForCausalLM, so we need the specific class
	Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
	"""

	def load(self):
		final_path = os.path.join(self.llama_weights_path, self.model_path)
		if self.tokenizer is None:
			self.tokenizer = LlamaTokenizer.from_pretrained(final_path, device_map='auto', torch_dtype=torch.float16) #torch_dtype=torch.float16
		if self.model is None:
			self.model = LlamaForCausalLM.from_pretrained(final_path, device_map='auto', torch_dtype=torch.float16)
			self.model.eval()
			self.model.to(self.device)

	def run(self, prompt: str, temperature=0.0, num_return_sequences=1) -> str:
		if self.use_template:
			template = (
				"Generate more creative instructions and corresponding preferred and rejected responses. "
			)
			text = template.format_map(dict(instruction=prompt))
		else:
			text = prompt

		# self.load()
		self.tokenizer.pad_token = self.tokenizer.bos_token
		self.tokenizer.padding_side = 'left'
		inputs = self.tokenizer(text, return_tensors="pt", return_length=True, padding=True).to(self.device)
		if temperature == 0.0:
			outputs = self.model.generate(
				inputs.input_ids,
				do_sample=False,
				num_return_sequences=1,
				max_new_tokens=self.max_tokens
			)
		else:
			outputs = self.model.generate(
				inputs.input_ids,
				do_sample=True,
				temperature=temperature,
				num_return_sequences=num_return_sequences,
				max_new_tokens=self.max_tokens
			)

		lengths = inputs.length
		maxlen = max(lengths)
		final_ops = []

		for iz in range(len(outputs)):
			final_op = self.tokenizer.decode(outputs[iz, maxlen:], skip_special_tokens=True)
			
			for stop_ele in self.stop:
				if stop_ele in final_op:
					final_op = final_op.split(stop_ele)[0]
					break

			final_ops.append(final_op)

		del inputs
		del outputs
		torch.cuda.empty_cache()

		return final_ops


class TGIModel():
	def __init__(self, port=8080, timeout=1000000):
		self.port = port
		self.timeout = timeout
		self.client = Client("http://127.0.0.1:" + str(port), timeout=timeout)
	def run(self, prompt, temperature=0.0, max_tokens=512, stop = []):
		fin_prompt = prompt
		if temperature == 0.0:
			response = self.client.generate(prompt=fin_prompt, do_sample=False, max_new_tokens=max_tokens, stop_sequences=stop).generated_text
		else:
			response = self.client.generate(prompt=fin_prompt, max_new_tokens=max_tokens, temperature=temperature, stop_sequences=stop).generated_text
		return response


def select_model(max_input_length=512, max_tokens=512, stop=[";"], model_type="causal", model_path="facebook/opt-1.3b", llama_weights_path='/home/mila/a/arkil.patel/scratch', model=None, tokenizer=None, device=torch.device("cuda:0"), use_template=False) -> EvalModel:
	model_map = dict(
		seq_to_seq=SeqToSeqModel,
		causal=CausalModel,
		llama=LlamaModel,
	)
	model_class = model_map.get(model_type)
	if model_class is None:
		raise ValueError(f"{model_type}. Choose from {list(model_map.keys())}")
	return model_class(max_input_length=max_input_length, max_tokens=max_tokens, stop=stop, model_path=model_path, llama_weights_path=llama_weights_path, model=model, tokenizer=tokenizer, device=device, use_template=use_template)


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.error.RateLimitError,  # Rate limit exceeded (20 requests per minute)
				openai.error.APIConnectionError,  # Sometimes we get a connection error
				openai.error.ServiceUnavailableError,
				openai.error.APIError
				# OpenAIAPIError, # Sometimes we get APIError: Internal Error
			)
		),
		wait=wait_random_exponential(
			multiplier=1.2,
			max=5,
		),
	)
def _get_completion_response(engine, prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, best_of, logprobs=1, echo=False):
	return openai.Completion.create(engine=engine,
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty,
		best_of=best_of,
		logprobs=logprobs,
		echo=echo
	)

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				# openai.error.RateLimitError,  # Rate limit exceeded (20 requests per minute)
				# openai.error.APIConnectionError,  # Sometimes we get a connection error
				# openai.error.ServiceUnavailableError,
				# openai.error.APIError
				# OpenAIAPIError, # Sometimes we get APIError: Internal Error
			)
		),
		wait=wait_random_exponential(
			multiplier=1.2,
			max=5,
		),
	)
def _get_chat_response(engine, prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty):
	return openai.ChatCompletion.create(
		model=engine,
		messages = [
			{"role": "system", "content": "You are SQLBot, a machine to generate accurate SQL queries for the questions asked based on the context provided. You must only generate the SQL Query."},
			{"role": "user", "content": prompt}
		],

		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty
	)

# @retry(
# 	retry=retry_if_exception_type(
# 			exception_types=(
# 				cohere.CohereError
# 			)
# 		),
# 		wait=wait_random_exponential(
# 			multiplier=3,
# 			max=10,
# 		),
# 	)
# def _get_cohere_response(engine, prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty):
# 	return co.generate(
# 		model=engine,
# 		prompt=prompt,
# 		max_tokens=max_tokens,
# 		temperature=temperature,
# 		p=top_p,
# 		num_generations=n,
# 		stop_sequences=stop,
# 		presence_penalty=presence_penalty,
# 		frequency_penalty=frequency_penalty,
# 		truncate='NONE'
# 	)

class LargeLanguageModel():
	def __init__(self, model_type, engine, hf_model_type, llama_weights_path, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, port=8080, timeout=1000000):
		self.model_type = model_type
		self.engine = engine
		self.hf_model_type = hf_model_type
		self.llama_weights_path = llama_weights_path
		self.max_tokens = max_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.n = n
		self.stop = stop
		self.presence_penalty = presence_penalty
		self.frequency_penalty = frequency_penalty
		self.encoding = None

		if self.model_type == "huggingface":
			self.model = select_model(model_type=self.hf_model_type, model_path=self.engine, llama_weights_path=self.llama_weights_path, max_tokens=self.max_tokens, stop=self.stop)
			self.model.load()
		elif self.model_type in ['chat', 'completion']:
			self.encoding = tiktoken.encoding_for_model(engine)
		elif self.model_type == "tgi":
			self.model = TGIModel(port = port, timeout = timeout)

	def predict(self, prompt):
		if self.model_type == "completion":
			response = _get_completion_response(
				engine=self.engine,
				prompt=prompt,
				max_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
				n=self.n,
				stop=self.stop,
				presence_penalty=self.presence_penalty,
				frequency_penalty=self.frequency_penalty,
				best_of=self.n+1,
				echo=False
			)
			response = response["choices"][0]['text'].strip()
		elif self.model_type == "chat":
			response = _get_chat_response(
				engine=self.engine,
				prompt=prompt, 
				max_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
				n=self.n,
				stop=self.stop,
				presence_penalty=self.presence_penalty,
				frequency_penalty=self.frequency_penalty
			)
			response = response["choices"][0]['message']['content'].strip()
		elif self.model_type == "tgi":
			cur_max_tokens = self.max_tokens
			while(True):
				if cur_max_tokens < 0:
					response = "BEYOND CONTEXT LENGTH"
					break
				try:
					response = self.model.run(prompt=prompt, temperature=self.temperature, max_tokens=cur_max_tokens, stop=self.stop).lstrip('\n').rstrip('\n')
					break
				except:
					cur_max_tokens = cur_max_tokens - 50
					print("Reducing Max Length to...", cur_max_tokens)
					continue
		# elif self.model_type == "cohere":
		# 	response = _get_cohere_response(
		# 		engine=self.engine,
		# 		prompt=prompt,
		# 		max_tokens=self.max_tokens,
		# 		temperature=self.temperature,
		# 		top_p=self.top_p,
		# 		n=self.n,
		# 		stop=self.stop,
		# 		presence_penalty=self.presence_penalty,
		# 		frequency_penalty=self.frequency_penalty
		# 	)
		# 	response = str(response[0]).strip()
		# 	time.sleep(21)
		else:
			response = self.model.run(prompt=prompt, temperature=self.temperature, num_return_sequences=self.n)

		return response



def test_model(
	prompt: str = "Write an email to a professor asking for deadline extension on the assignment.",
	model_type: str = "llama",
	model_path: str = "facebook/opt-iml-1.3b",
):
	model = select_model(model_type=model_type, model_path=model_path)
	print(model.run(prompt))

if __name__ == "__main__":
	test_model()