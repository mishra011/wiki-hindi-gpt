from langchain_community.llms import CTransformers

path5 = "./model_hi_custom/"
# llm = CTransformers(model=path5,
#                         #streaming=True, 
#                         #callbacks=[StreamingStdOutCallbackHandler()],
#                         #max_tokens=max_tokens,
#                         model_type="gpt2"
#                         )


# from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained(path5, model_type="gpt2")


from langchain_community.llms import GPT4All

gpt4all = GPT4All(
    model=path5,
    max_tokens=2048,
)
