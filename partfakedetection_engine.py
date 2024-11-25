import argparse
import os
import sys
import time
import platform
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM

class LLM_Engine(object):
    '''def LLM_Engine(self, transcript_path):
        """
        :param model_dir: Path to LLM_Model dir
        :return: model_name in training/testing format
        """
        raise NotImplementedError'''
    def __init__(self, model_dir, model_name, device, MAX_TOKENS = 8092, MAX_TURNS = 10, Session_Time = 3600, load_in_8bit = False): 

        ## Basic configure
        self.MAX_TOKENs   = MAX_TOKENS
        self.MAX_TURNs    = MAX_TURNS
        self.Session_Time = Session_Time
        self.load_in_8bit = load_in_8bit
        
        ## Load llm model
        self.model_dir  = model_dir
        self.model_name = model_name
        self.device     = device

        checkpoint = os.path.join(self.model_dir, self.model_name)
        if not os.path.exists(checkpoint):
            exit("ERROR: Load LLM Model %s is not existed\n" % (checkpoint))
        
        if 'chatglm' in self.model_name.lower():
           # print("Load ChatGLM model\n")
            print("Step-1: Load tokenizer......\n")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            print("Step-2: Load llm Model......\n")
            self.model = AutoModel.from_pretrained(checkpoint)

        elif 'llama' in self.model_name.lower():
            #print("Load LLAMA model\n")
            print("Step-1: Load tokenizer......\n")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            print("Step-2: Load llm Model......\n")
            if self.load_in_8bit:
                self.model = LlamaForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)
            else:
                #self.model = LlamaForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True)
                self.model = LlamaForCausalLM.from_pretrained(checkpoint)
        else:
            #print("Load BLOOM model\n")
            print("Step-1: Load tokenizer......\n")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            print("Step-2: Load llm Model......\n")
            if self.load_in_8bit: 
                self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        
        if not self.load_in_8bit:
            self.model = self.model.half().to(self.device)

        self.model = self.model.eval()

        self.sessions  = {} # str:(session_id+user_id), list(time, prompt, answer)
    
    def flush_sessions(self):
        cur_time = time.time()
        remove_session = []
        for session, historys in self.sessions.items():
            if len(historys) > 0:
                prompt_time, prompt, prompt_tokenid, answer, answer_tokenid = historys[-1]
                if cur_time - prompt_time >= self.Session_Time:
                    remove_session.append(session)
        
        if len(remove_session) > 0:
            for session in remove_session:
                self.sessions.pop(session)


    def clear_historys_accordingto_prompt(self, aprompt, session_id, usr_id):

        session = session_id + "+" + usr_id
        if session in self.sessions:
            if ("结束" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("到此为止" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("over" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("stop" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("开启" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("开始" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("清空" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("终止" in aprompt and ("会话" in aprompt or "聊天" in aprompt)) or \
               ("中止" in aprompt and ("会话" in aprompt or "聊天" in aprompt)):
               #print("clear_historys_accordingto_prompt\n")
               self.sessions[session] = []


    def build_prompt(self, aprompt, session_id, usr_id):
        inputs = 'Human: ' + aprompt.strip() + ' \n\nAssistant: '
        aprompt_tokenid = self.tokenizer(inputs, return_tensors="pt").input_ids
        cur_time = time.time()
        session = session_id + "+" + usr_id
        if session in self.sessions:
            historys = self.sessions[session]

            inputs = ""
            if len(historys) > 0:
            
                # Remove the turns of chat that exceed the MAX_TURNs
                if len(historys) > self.MAX_TURNs:
                    historys[0 : len(historys) - self.MAX_TURNs] = []

                # Consider the max context tokens
                if len(historys) > 0:
                    flag = 0
                    num_newest_tokens = aprompt_tokenid.size(1)
                    for i in range(len(historys) - 1, -1, -1):
                        prompt_time, prompt, prompt_tokenid, answer, answer_tokenid = historys[i]
                        num_newest_tokens = num_newest_tokens + prompt_tokenid.size(1) + answer_tokenid.size(1)
                        if num_newest_tokens > self.MAX_TOKENs:
                            flag = 1
                            break
                    if i < len(historys) - 1:
                        if flag == 1:
                            historys = historys[i+1:]
                    else:
                        historys = historys[len(historys) - 1:]

                    for prompt_time, prompt, prompt_tokenid, answer, answer_tokenid in historys:
                        inputs = inputs + 'Human: ' + prompt.strip() + ' \n\nAssistant: ' + answer.strip()
            
            inputs = inputs + 'Human: ' + aprompt.strip() + ' \n\nAssistant: '
            historys.append((cur_time, aprompt, aprompt_tokenid, "", None))

        else:
            inputs   = 'Human: ' + aprompt.strip() + ' \n\nAssistant: '
            historys = []
            historys.append((cur_time, aprompt, aprompt_tokenid, "", None))
        self.sessions[session] = historys

        return inputs

    def post_aprompt(self, aprompt, session_id, usr_id):

        # Remove the expired sessions according to time
        self.flush_sessions()

        # Clean the history according to aprompt
        self.clear_historys_accordingto_prompt(aprompt, session_id, usr_id)

        input_strs = self.build_prompt(aprompt, session_id, usr_id)

        #print("\n\n########################\n")
        #print(input_strs)
        #print("########################\n")

        if input_strs is not None:
            inputs_ids = self.tokenizer(input_strs, return_tensors="pt").input_ids.to(self.device)
            if 'llama' in self.model_name.lower():
                output_tokenids = self.model.generate(inputs_ids, max_new_tokens = self.MAX_TOKENs, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
                rets = self.tokenizer.batch_decode(output_tokenids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else: 
                output_tokenids = self.model.generate(inputs_ids, max_new_tokens = self.MAX_TOKENs, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.35, repetition_penalty=1.2)
                rets = self.tokenizer.batch_decode(output_tokenids, skip_special_tokens=True)

            #print(rets[0].strip())
            output_strs = rets[0].strip()
            output_strs = output_strs.split("\n\nAssistant:")[-1]
            #output_strs = rets[0].strip()[len(input_strs):]
            
            session = session_id + "+" + usr_id
            if session in self.sessions:
                historys = self.sessions[session]
                prompt_time, prompt, prompt_tokenid, answer, answer_tokenid = historys[-1]
                answer         = output_strs
                answer_tokenid = output_tokenids
                historys[-1] = (prompt_time, prompt, prompt_tokenid, answer, answer_tokenid)
                self.sessions[session] = historys
            output_strs = output_strs.replace("Human:", "")
            output_strs = output_strs.replace("Human：", "")
            output_strs = output_strs.replace("Assistant:", "")
            output_strs = output_strs.replace("Assistant：", "")
        else:
            output_strs = None
        
        # Clean the history according to aprompt
        self.clear_historys_accordingto_prompt(aprompt, session_id, usr_id)

        return output_strs

    
    def post_bacth_prompt(self, posts):
        
        # Remove the expired sessions according to time
        self.flush_sessions()

        input_strs = []
        for aprompt, session_id, usr_id in posts:

            # Clean the history according to aprompt
            self.clear_historys_accordingto_prompt(aprompt, session_id, usr_id)

            input_strs.append(self.build_prompt(aprompt, session_id, usr_id))

        batch_output_strs = []
        if input_strs is not None:
            inputs_ids = self.tokenizer(input_strs, return_tensors="pt").input_ids.to(self.device)
            if 'llama' in self.model_name.lower():
                output_tokenids = self.model.generate(inputs_ids, max_new_tokens = self.MAX_TOKENs, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
                rets = self.tokenizer.batch_decode(output_tokenids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else: 
                output_tokenids = self.model.generate(inputs_ids, max_new_tokens = self.MAX_TOKENs, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.35, repetition_penalty=1.2)
                rets = self.tokenizer.batch_decode(output_tokenids, skip_special_tokens=True)

            for i in range(len(posts)):
                aprompt, session_id, usr_id = posts[i]
                #print(rets[0].strip())
                output_strs = rets[i].strip()
                output_strs = output_strs.split("\n\nAssistant:")[-1]
                #output_strs = rets[0].strip()[len(input_strs):]
                
                session = session_id + "+" + usr_id
                if session in self.sessions:
                    historys = self.sessions[session]
                    prompt_time, prompt, prompt_tokenid, answer, answer_tokenid = historys[-1]
                    answer         = output_strs
                    answer_tokenid = output_tokenids
                    historys[-1] = (prompt_time, prompt, prompt_tokenid, answer, answer_tokenid)
                    self.sessions[session] = historys
                output_strs = output_strs.replace("Human:", "")
                output_strs = output_strs.replace("Human：", "")
                output_strs = output_strs.replace("Assistant:", "")
                output_strs = output_strs.replace("Assistant：", "")
                batch_output_strs.append(output_strs)
                # Clean the history according to aprompt
                self.clear_historys_accordingto_prompt(aprompt, session_id, usr_id)
        
        return batch_output_strs
