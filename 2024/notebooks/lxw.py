import httpx
import openai


gpt35_turbo = "gpt-35-turbo-1106"
gpt35_turbo_1106 = "gpt-3.5-turbo-1106"

gpt4_turbo = "gpt-4-1106-preview"
gpt4_turbo_1106 = "gpt-4-1106-preview"
gpt4_turbo_0125 = "gpt-4-0125-preview"

gpt4_o = 'gpt-4o'

dall_e_3 = 'dall-e-3'

ada_002 = 'text-embedding-ada-002'

def new_client(model = ''):
    if not model:
        client = openai.AzureOpenAI(
            api_key="",
            api_version="",
            azure_endpoint=""
        )
        return client
    if model in [gpt4_turbo_0125, dall_e_3, gpt4_o, dall_e_3]:
        client = openai.AzureOpenAI(
            api_key="",
            api_version="",
            azure_endpoint=""
        )
    else:
        client = openai.AzureOpenAI(
            api_key="",
            api_version="",
            azure_endpoint=""
        )    
    return client

def new_openai_client():
    http_client = httpx.Client(proxies="")
    client = openai.OpenAI(
        api_key="",
        http_client=http_client
    )
    return client

def chat_complete(user_prompt, system_prompt = None, model = None):  
    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    if system_prompt:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ] + messages
    if not model:
        model = gpt4_turbo_0125

    client = new_client(model = model)

    response = client.chat.completions.create(
        model = model,
        messages = messages
    )
    return response.choices[0].message.content

def print_response(response):
    NO = 1
    for choice in response.choices:
        print("--------" + str(NO) + "--------\r\n")
        print(choice.message.content)
        print("\r\n")
        NO += 1

import tiktoken

gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")
