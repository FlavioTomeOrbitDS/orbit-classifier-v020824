import utils_conf
import aiohttp
import pandas as pd

async def openAiApiCall(messages, model, temperature):    
    api_key = utils_conf.get_config_value('OPENAI_KEY')
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": model, "messages": messages, "temperature": temperature}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                return f"Error: {response.status}, {await response.text()}"
            json_response = await response.json()            
            return json_response['choices'][0]['message']['content']

async def get_prompt(role,text):
    prompt = {'role': role,  'content': f"{text}"}
    return prompt

def hello():
    message = get_prompt("assistant", "Olá, como eu posso ajudar você?")
    return message
    
    
def how_to_use_training():
    message = get_prompt("system", f"""Você vai auxiliar o usuário a utilizar o Orbit Classifier seguindo e seguinte manual:
                        1. O Orbit Classifier foi desenvolvido utilizando os melhores modelos LLM de IA para analisar os sentimentos e categorizar uma lista de comentários de redes sociais inseridas pelo usuário.
                        2. O Classifier pode ser utilizado de 2 maneiras. A primeira é por meio do nosso Tweets Search, onde o usuário poderá procurar os tweets dos últimos 7 dias relacionados a sua palavra chave de busca. O usuário pode escolher em qual conta do twitter a busca estará relacionada, mas atenção porque cada conta em um limite de 10.00 tweets por mês.
                        A segunda maneira de utilizar o Classifier é fazendo o upload diretamente de uma base de dados no formato .xlsx . Atenção, os textos a serem analisados devem estar em uma coluna chamada "Texto" no arquivo, não importando se existem outros dados nas outras colunas. 
                        3. O Usuário pode fazer o download dos arquivos gerados pelo classifier clicando nos botões que aparecerão logo abaixo das tabelas.                        
                        Dê uma saudação e pergunto no que voce pode ajudar.
                        Dê respostas resumidas.                        
                        """)
    
    return message

def initial_training(dataset):    
    message = get_prompt('system', f"""Você é um analista de dados especializado em social media. 
                         Seu trabalho é auxiliar o usuário respondendo perguntas a respeito do Dataset informado. 
                         Seja gentil, claro e objetivo nas respostas.
                         Caso o dataset esteja vazio. Informe ao usuário que ele deve primeiramente iniciar a Classificação de Textos.
                         Dataset: {dataset}""")
    return message


def dataset_training():
    classification_df = pd.read_excel("outputs/text_classification_output.xlsx", sheet_name = 1)    
    message = get_prompt('system', f"QUando o usuário perguntar, leia e interprete os seguintes dados: {classification_df.to_string()}")

    return message