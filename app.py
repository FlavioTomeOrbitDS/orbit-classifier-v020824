from flask import Flask, request, jsonify,send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
from tweets_search import tweets_search
import pandas as pd
import utils_conf
import pandas as pd
import aiohttp
import asyncio
import re
import os
import datetime
import difflib


def get_current_datetime():
  """Returns the current datetime as a string."""
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_text(text):
    # Remove a sequência "RT :"
    text = text.replace("RT", "").strip()
    # Remove a palavra "sentimento"
    text = text.replace("sentimento", "").strip()
    # Remove "Comentário" e suas variações
    text = re.sub(r'^Comentário\s*\d*:\s*', '', text)
    # Remove números iniciais
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove aspas simples e duplas
    text = text.replace("'", "").replace('"', "")
    # Remove pontuações iniciais
    text = text.lstrip(".'\"")
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove menções
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove emojis e outros caracteres não alfanuméricos, exceto espaços, vírgulas
    text = re.sub(r'[^\w\s,]', '', text)
    # Remove espaços extras no início e no final do texto
    text = text.strip()

    return text

async def make_api_call_to_gpt(prompt, api_key):    
    api_key = utils_conf.get_config_value('OPENAI_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    async with aiohttp.ClientSession() as session:                
        payload = {
            "model": "gpt-4-turbo",
            "messages": prompt,
            "temperature": 0,
            "max_tokens": 2000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(payload)) as response:
            if response.status == 200:
                resp_json = await response.json()
                #print(resp_json)
                return resp_json['choices'][0]['message']['content']
            else:
                print(f"##### ERROR {response.status}")
                print(response)
                return f"Error: {response.status}"


async def openAiApiCall(messages):    
    api_key = utils_conf.get_config_value('OPENAI_KEY')
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": "gpt-4", "messages": messages, "temperature": 0.1}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                return f"Error: {response.status}, {await response.text()}"
            json_response = await response.json()            
            return json_response['choices'][0]['message']['content']


async def get_prompt(role,text):
    prompt = [{'role': role,  'content': f"Analise o sentimento do texto a seguir. Retorne positivo, negativo ou neutro, sem nenhuma outra informaçã adicional. Caso não seja possível definir o sentimento, retorne indeterminado: {text}"}]
    return prompt

def dividir_lista(lista, tamanho_sublista):
    # Usando list comprehension para criar sub-listas com o tamanho especificado
    return [lista[i:i + tamanho_sublista] for i in range(0, len(lista), tamanho_sublista)]


#******************************* TAGS ********************************************************
def geraMensagens_tags(lista_de_comentarios, tema_principal, contexto, sentimento):
    messages = [
        {"role": "system", "content": f"""Tema Principal: {tema_principal}. 
Contexto: {contexto}
Sentimento Principal: {sentimento}
 Objetivo: Criar 10 tags baseadas na lista de Textos, no tema principal, no contexto e no sentimento principal informados. 
Instruções: 
1. As tags devem ser frases de no máximo 4 palavras que melhor sintetizem o conjunto de textos, expressando os sentimentos contidos neles.; 

2. Caso o Sentimento Principal seja "negativo" ou "positivo", utilize palavras que reforcem o sentimento das frases, como por exemplo, Críticas, Elogios, Apoio, Rejeição, Satisfação, Insatisfação, Aprovação, Desaprovação, Alegria, Tristeza, Amei, Odiei, Gostei, Não gostei, Positivo, Negativo, etc... 
3. Caso o sentimento principal seja neutro, não utilize palavras relacionadas aos sentimentos positivo ou negativo e nem as palavras listadas na instrução 2
Lista de textos:{lista_de_comentarios}
"""}
   ]
    return messages

def normalize_text(text):
    """ Normaliza o texto removendo espaços extras, convertendo para minúsculas e removendo pontuações comuns. """
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def find_unique_tags(tags):
    """ Encontra categorias únicas em uma lista, considerando semelhanças de texto simples. """
    unique_categories = []
    seen_categories = set()

    for tag in tags:
        norm_category = normalize_text(tag)
        
        # Verificar se a categoria normalizada já foi vista ou é muito similar a uma existente
        if not any(difflib.SequenceMatcher(None, norm_category, normalize_text(existing)).ratio() > 0.8 for existing in seen_categories):
            unique_categories.append(tag)
            seen_categories.add(norm_category)

    return unique_categories

def flatten_tags_list(tags_groups):
    # Lista para armazenar todas as categorias unificadas
    all_tags = []
    
    # Itera sobre cada grupo de categorias na lista fornecida
    for group in tags_groups:
        # Divide o grupo em linhas individuais (categorias)
        tags = group.split('\n')
        # Remove a numeração de cada categoria e adiciona à lista unificada
        for tag in tags:
            # Extrai a parte da categoria após o número e o ponto
            cleaned_tag = tag.split('. ', 1)[1] if '. ' in tag else tag
            all_tags.append(cleaned_tag.strip())
    
    return all_tags



async def tags_analysis(lista_de_comentarios, tema_principal, contexto, sentimento):
    print(f"##### Criando as tags...{get_current_datetime()}")
    # Dividir a lista principal de comentários em uma lista menor para fazer as chamadas à API
    aux_list = dividir_lista(lista_de_comentarios, 100)
    
    # Gera as mensagens para a API para cada item das listas menores
    lista_mensagens = []
    for i in aux_list:
        lista_mensagens.append(geraMensagens_tags(i, tema_principal, contexto, sentimento))
    
    # Gera a lista de tarefas para fazer as chamadas assíncronas para a API
    tasks = []
    for mensagem in lista_mensagens:
        tasks.append(openAiApiCall(mensagem))
    

    # Lista para armazenar as respostas
    respostas = []
    
    # Executando as tarefas e atualizando a barra de progresso
    for i, task in enumerate(asyncio.as_completed(tasks), start=1):
        resposta = await task
        respostas.append(resposta)    
    
    # Remove as categorias duplicadas
    flattened_tags = flatten_tags_list(respostas)
    unique_tags = find_unique_tags(flattened_tags)
    df_tags = pd.DataFrame(data=(unique_tags), columns=['Tag'])    
    
    # Filtra o DataFrame para remover as linhas com 'Tag' == 'Indefinido'
    df_tags = df_tags[df_tags['Tag'] != 'Indefinido']

    
    return df_tags

#******************************* CATEGORIAS ********************************************************
def geraMensagens_categorias(lista_de_tags, tema_principal, sentimento):
    messages = [
        {"role": "system", "content": f"""Tema principal: {tema_principal}
Objetivo: Gerar categorias para o conjunto de textos informados pelo usuário.
Metodologia: Leia a lista de textos e classifique cada texto conforme as categorias criadas.
Regras: 1. As categorias devem ter no máximo 3 palavras. 
2. Evite nomear uma categoria com o tema principal.
3. Não crie mais de 6 categorias.
4. Retorne somente os textos classificados
5. As categorias devem estar relacionadas ao sentimento {sentimento}
6. Não utilize a palavra 'Categoria' ou alguma numeração.
Formato de saída : JSON. chaves:categorias; valores: tags 
Lista de tags: {lista_de_tags}
"""}
    ]
    return messages

def geraMensagens_classificação_refinamento(texto):
    messages = [
        {"role": "system", "content": f"""Leia o JSON abaixo que está configurada na forma : chave:categoria, valor: tags
1. Caso existam mais de 6 categorias, substitua algumas de forma que existam no máximo 10. 
3. Se necessário gere novas categorias de forma que consiga agrupar outras. 
4. Substitua as categorias no tageamento e retorne uma string no mesmo formato.
Formato de saída: 
Um JSON no mesmo formato
Observação: retorne somente o novo tageamento, sem nenhuma outra informação 
JSON para analise:{texto} 
"""}
    ]
    return messages

def formata_categorias(json_str):
    # caso a string venha com a palavra json no início
    json_str = json_str.replace("json", "")
    # Parse the JSON string
    data = json.loads(json_str)
    
    # Prepare a list to store the rows of the DataFrame
    rows = []

    # Loop through each category and its corresponding list of items
    for category, items in data.items():
        for item in items:
            rows.append({"Categoria": category, "Tag": item})
    
    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows)
    
    print("formata_categorias resultado:")
    print(df)
    
    return df

async def refina_categorias(texto):
    print(f"##### Refinando as Categorias...{get_current_datetime()}")
    mensagem = geraMensagens_classificação_refinamento(texto)
    
    resposta = await openAiApiCall(mensagem)
    print(f"#### CAtegorias: {resposta}")
    
    df = formata_categorias(resposta)
    
    return df

async def category_analysis(lista_de_tags, tema_principal, sentimento): 
    print(f"##### Gerando as Categorias...{get_current_datetime()}")       
    mensagem = geraMensagens_categorias(lista_de_tags, tema_principal, sentimento)    
    resposta = await openAiApiCall(mensagem)                
    
    df = await refina_categorias(resposta)
    
    return df

#******************************* CLASSIFICAÇÃO ********************************************************
def geraMensagens_classificação(lista_de_comentarios,lista_de_categorias):
    messages = [
        {"role": "system", "content": f"""Objetivo da tarefa: Classificar os comentários.

        Classifique a lista de comentário de acordo com a categoria que melhor enfatizar o sentido do texto.
        Caso um comantério não possa ser classificado com as categorias existentes, classifique- o como : Indefinido.

        Formato de saída:  JSON do tipo chave:categoria, valor: comentarios
        
        Lista de Categorias: {lista_de_categorias}
         
        Lista de Comentários: {lista_de_comentarios}        
"""}
    ]
    return messages

def clean_json_string(json_string):
    
    # Substitui aspas simples por aspas duplas
    json_string = json_string.replace("'", '"')
    
    # Remove caracteres de nova linha e tabulações
    json_string = json_string.replace("\n", " ").replace("\t", " ")
    
    # Remove barras extras
    json_string = json_string.replace("\\", "")
    
    # Remove vírgulas finais antes de fechar listas ou dicionários
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    
    # Corrige problemas de formato de JSON, como substituição de \n
    json_string = json_string.replace("\\n", "\n")

    # Remove aspas adicionais desnecessárias
    json_string = json_string.replace('""', '"')
    
     # Remove espaços em branco extras entre palavras
    json_string = re.sub(r'\s+', ' ', json_string).strip()

    return json_string

def formata_classificacao(json_list):
    print(f"formata_classificacao: {json_list}")
    
    rows = []
    
    for json_obj in json_list:
        # Limpa e corrige o JSON string antes de processar
        json_obj = clean_json_string(json_obj)
        try:
            data = json.loads(json_obj)
            print(data)
            for tag, texts in data.items():
                for text in texts:
                    rows.append({'Tag': tag, 'Texto': text})
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
    
    df = pd.DataFrame(rows)
    print(f"formata_classificacao dataframe: {df}")
    return df

# 
# def formata_classificacao(json_list):
#     print(f"formata_classificacao: {json_list}")
    
#     rows = []
    
#     for json_obj in json_list:
#         json_obj = json_obj.replace("'", '"')
#         data = json.loads(json_obj)
#         for tag, texts in data.items():
#             for text in texts:
#                 rows.append({'Tag': tag, 'Texto': text})
    
#     df = pd.DataFrame(rows)
#     print(f"formata_classificacao dataframe: {df}")
#     return df
    
    
async def text_classification(lista_de_comentarios,lista_de_tags):
    print(f"##### Fazendo a Classificação...{get_current_datetime()}")
    #print(lista_de_comentarios)
    #print(lista_de_tags)
    # Dividir a lista principal de comentários em uma lista menor para fazer as chamadas à API
    aux_list = dividir_lista(lista_de_comentarios, 10)
    
    # Gera as mensagens para a API para cada item das listas menores
    lista_mensagens = []
    for i in aux_list:
        lista_mensagens.append(geraMensagens_classificação(i, lista_de_tags))
    
    # Gera a lista de tarefas para fazer as chamadas assíncronas para a API
    tasks = []
    for mensagem in lista_mensagens:
        tasks.append(openAiApiCall(mensagem))
        
    # Lista para armazenar as respostas
    respostas = []
    
    # Executando as tarefas e atualizando a barra de progresso
    for i, task in enumerate(asyncio.as_completed(tasks), start=1):
        resposta = await task        
        if resposta != 'Error: 502, <html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n<center><h1>502 Bad Gateway</h1></center>\r\n<hr><center>cloudflare</center>\r\n</body>\r\n</html>\r\n':
            respostas.append(resposta)  
                
    df = formata_classificacao(respostas)       
        
    return df

import json


#*********************** ANALISE DE SENTIMENTOS ***********************************************
async def sentiment_analysis(df, api_key, max_per_call=100):
    print("Analisando sentimentos...")
    sentiments = []
    for start in range(0, len(df), max_per_call):
        end = start + max_per_call
        tasks = [make_api_call_to_gpt(await get_prompt('system',text), api_key) for text in df['texto_limpo'][start:end]]
        sentiments.extend(await asyncio.gather(*tasks))
     
    return sentiments

import numpy as np

def split_dataframe(df, max_items=50):
    """
    Splits a DataFrame into multiple smaller DataFrames, each with a maximum of 'max_items' rows.
    """
    num_splits = int(np.ceil(len(df) / max_items))
    return np.array_split(df, num_splits)


async def process_sentiments(df):     
    #Separa o dataframe em grupos de 100 
    df_list = split_dataframe(df)
    final_result = []
    #para cada bloco de 100, analisa os sentimentos de forma assíncrona usando a API
    for df_part in df_list:
         results = await sentiment_analysis(df_part, "")
         final_result.append(results)        
                      
    flat_list = [item for sublist in final_result for item in sublist]
    
    df['Sentimento'] = flat_list    
    return df


async def analise_de_sentimentos(df):                        
    #Análise de sentimentos
    print(f"##### Fazendo a analise de sentimentos...{get_current_datetime()}")
    df = await (process_sentiments(df))    
    df['Sentimento'] = df['Sentimento'].replace("Negative", "negativo")
    df['Sentimento'] = df['Sentimento'].replace("Positive", "positivo")
    df['Sentimento'] = df['Sentimento'].replace("Neutral", "neutro")
    #df.to_excel("teste _sentimentos_2.xlsx")     
    
    return df                       

def gera_df_final(df_classific,df_categorias):
    print(f"##### Gerando arquivo final...{get_current_datetime()}")
    # Primeiro, garanta que a coluna 'Tag' no DataFrame df_classific não tenha espaços extras
    df_classific['Tag'] = df_classific['Tag'].str.strip()
    df_categorias['Tag'] = df_categorias['Tag'].str.strip()

    # Realizar uma junção (merge) para adicionar a coluna 'Categoria' baseada na correspondência de 'Tag'
    df_classific = pd.merge(df_classific, df_categorias, on='Tag', how='left')

    #df_classific = df_classific.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])

    # O DataFrame df_classific agora possui uma coluna 'Categoria' correspondente
    return df_classific


def processar_incidencias(data, df_final, file_path):
    try:
        # Exibindo uma mensagem com o horário atual do processamento
        print(f"##### Calculando Incidências...{get_current_datetime()}")
        
        

        # Verificar se as colunas necessárias estão presentes
        required_columns = ['Categoria', 'Tag', 'Sentimento']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("As colunas necessárias não estão todas presentes no DataFrame.")

        #remove todos os valores vazios do dataframe
        data = data.dropna(subset=required_columns)
        
        # Agrupar os dados por Sentimento, Categoria e Tag e calcular a incidência
        grouped_data = data.groupby(required_columns).size().reset_index(name='Incidência')

        # Calcular a incidência total por categoria e sentimento
        total_por_categoria_sentimento = data.groupby(['Sentimento', 'Categoria']).size().reset_index(name='Total Categoria')

        # Calcular a incidência total geral
        total_geral = data.shape[0]

        # Unir os dados do agrupamento com os totais por categoria e sentimento para calcular as porcentagens
        final_data = pd.merge(grouped_data, total_por_categoria_sentimento, on=['Sentimento', 'Categoria'])
        final_data['% na Categoria'] = (final_data['Incidência'] / final_data['Total Categoria']) * 100
        final_data['% no Arquivo Geral'] = (final_data['Incidência'] / total_geral) * 100

        # Verificar se o arquivo existe e excluí-lo se existir
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Arquivo existente '{file_path}' removido.")

        # Criar uma nova aba no arquivo Excel para salvar os resultados do processamento de incidências e df_final
        with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
            final_data.to_excel(writer, sheet_name='Incidência', index=False)
            data.to_excel(writer, sheet_name='Dados Finais', index=False)

        return final_data

    except Exception as e:
        print(f"Erro ao processar incidências: {e}")
        # Retornando um DataFrame vazio em caso de erro
        return pd.DataFrame()
        
async def start_classification(lista_de_comentários, tema, context, sentimento):                    
    #Gera as Tags
    try:        
        df_tags = ( await tags_analysis(lista_de_comentários, tema, context, sentimento))            
        print(f"##### TAGS {df_tags}")
    except Exception as e:
        print(f"##### Erro ao gerar Tag: {e}")
    #Gera as Categorias
    #try:
    df_categorias = await category_analysis(df_tags['Tag'].to_list(), tema, sentimento)       
    print(f"##### Categorias: {df_categorias}")
    #except Exception as e:
    #    print(f"##### Erro ao as Categorias: {e}")
    
    #Classificação dos textos
        
    df_classificado = ( await text_classification(lista_de_comentários, df_categorias['Tag'].to_list()))            
    print(f"##### Classificação: {df_classificado}")
    # except Exception as e:
    #     print(f"##### Erro ao Classificar: {e}")
    
    #GEra arquuivo final 
    try:
        df_final = gera_df_final(df_classificado,df_categorias)        
        df_final["Sentimento"] = sentimento        
        #df_final.to_excel("outputs/text_classification_output.xlsx")                
    except Exception as e:
        print(f"##### Erro ao gerar o arquivo final: {e}")
            
    return df_final        
    
async def classificacao_com_sentimento(df, tema, context):    
    #Analise de sentimentos
    df = await analise_de_sentimentos(df)

    print("### Sentimentos:")                    
    print(df)
    
    sentimentos = ["positivo", "neutro", "negativo"]
    all_data = []
    
    # Processamento de dados por sentimento
    for sentimento in sentimentos:
        print(f"##### Processando Sentimento: {sentimento}")
        
        # Filtrar e preparar dados para o sentimento atual
        df_sentimento = df[df["Sentimento"] == sentimento].copy()
        df_sentimento['texto_limpo'] = df_sentimento['Texto'].apply(clean_text)
        lista_de_comentários = df_sentimento['texto_limpo'].to_list()
        
        # Classificação dos textos
        df_classificado = await start_classification(lista_de_comentários, tema, context, sentimento)
        
        
        # Adiciona os dados classificados a uma lista
        all_data.append(df_classificado)

    # Concatenar todos os dados classificados em um único DataFrame
    df_concatenado = pd.concat(all_data, ignore_index=True)
    #print("### DF CONCATENADO")
    #print(df_concatenado)
    
    final_df = pd.DataFrame()
    # Processar incidências
    final_df = processar_incidencias(df_concatenado, final_df,"outputs/text_classification_output.xlsx")
        
    return final_df    


def generate_js_dictionary(file_path='outputs/text_classification_output.xlsx', sheet_name="Incidência"):
    # Carregar a aba especificada do arquivo Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Agrupar os dados por Categoria, construindo a estrutura necessária
    grouped = df.groupby('Categoria').apply(lambda x: x[['Tag', 'Incidência', '% na Categoria', '% no Arquivo Geral']].to_dict('records')).to_dict()

    # Construir a estrutura do dicionário conforme o modelo solicitado
    data_dict = {
        "name": "Tema Principal",
        "children": []
    }

    for category, details in grouped.items():
        category_dict = {
            "name": category,
            "children": [
                {"name": tag['Tag'], "value": tag['Incidência'],"percent_categoria": round(tag['% na Categoria'], 1),
                        "percent_geral": round(tag['% no Arquivo Geral'], 1)} for tag in details
            ]
        }
        data_dict['children'].append(category_dict)

    # Converter o dicionário para uma string formatada em JSON para ser usada em JavaScript    
    return json.dumps(data_dict, indent=2, ensure_ascii=False)

def generate_js_dictionary_with_sentiment(file_path='outputs/text_classification_output.xlsx', sheet_name="Incidência"):
    # Carregar a aba especificada do arquivo Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Agrupar os dados por Sentimento e Categoria, e incluir as Tags e Incidências
    grouped = df.groupby(['Sentimento', 'Categoria']).apply(
       lambda x: x[['Tag', 'Incidência', '% na Categoria', '% no Arquivo Geral']].to_dict('records')
    ).reset_index().groupby('Sentimento').apply(
        lambda x: x[['Categoria', 0]].set_index('Categoria').to_dict(orient='index')
    ).to_dict()

    # Construir a estrutura do dicionário conforme o modelo solicitado
    data_dict = {
        "name": "Tema Principal",
        "children": []
    }

    for sentiment, categories in grouped.items():
        sentiment_dict = {
            "name": f"sentimento {sentiment}",
            "children": []
        }
        for category, details in categories.items():
            category_dict = {
                "name": category,
                # "children": [
                #     {"name": tag['Tag'], "value": tag['Incidência']} for tag in details[0]
                "children": [
                {"name": tag['Tag'], "value": tag['Incidência'],"percent_categoria": round(tag['% na Categoria'], 1),
                        "percent_geral": round(tag['% no Arquivo Geral'], 1)} for tag in details[0]
            ]
                    
                    
                    
                
            }
            sentiment_dict['children'].append(category_dict)
        data_dict['children'].append(sentiment_dict)

    # Converter o dicionário para uma string formatada em JSON para ser usada em JavaScript
    # Usando ensure_ascii=False para manter caracteres acentuados corretamente
    #js_string = 'const data = ' + json.dumps(data_dict, indent=2, ensure_ascii=False) + ';'

    #return js_string
     
    return json.dumps(data_dict, indent=2, ensure_ascii=False)



app = Flask(__name__)
CORS(app)

#-------------------------------- ROUTES ------------------------------------------------------
@app.route('/')
def home():    
    
    return 'Text Classification Backend!'
    

#--------------------------------TWEETS SEARCH------------------------------------------------------
@app.route('/api/tweetssearch', methods=['POST'])
def tweetssearch():
    if request.is_json:
        # Get the JSON data
        data = request.get_json()    
        query = data.get('query', None)        
        max_results = data.get('max_tweets', None)                          
        b_token = utils_conf.get_config_value('bearer_token')                        
        df = tweets_search(query,b_token,"lang:pt" ,max_results)    
        json_data = df.to_json(orient='records')        
                
        return jsonify(json_data=json_data)            
        
    
    return None

#-------------------------------TEXT CLASSIFICATION -------------------------------------------------------
@app.route('/api/getclassifications', methods=['POST'])
async def getclassifications():    
    data = request.get_json()    
    context = data.get('context', None)
    tema = data.get('tema', None)                
    
    df_input = pd.read_excel('outputs/tweets_search_output.xlsx')                                                            
    if 'Texto' in df_input.columns:                
        df_input = df_input.head(2000)                
        df_input['texto_limpo']  = df_input['Texto'].apply(clean_text)
        
        final_df = await classificacao_com_sentimento(df_input,tema, context)
        
        print("### RESULTADO FINAL DA CLASSIFICAÇÃO:")
        print(final_df)       
                                                    
        print(f"##### Processo finalizado...{get_current_datetime()}")
                        
        return jsonify(json_data=final_df.to_json(orient = 'records')), 200
    else:
        return jsonify({"error": "Column 'Texto' not found"}), 400

    # if request.is_json:    
    #     data = request.get_json()    
    #     context = data.get('context', None)
    #     tema = data.get('tema', None)        
    #     # # Get the JSON data       
    #     df = pd.read_excel('outputs/tweets_search_output.xlsx')    
    #     #result = text_classification(df, context, tema)           
    #     if 'Texto' in df.columns:
    #         textos = df['Texto']
    #         df = pd.DataFrame(textos)                      
    #         df['texto_limpo']  = df['Texto'].apply(clean_text)
    #         lista_de_comentários = df['texto_limpo'].to_list()                                        
            
    #         json_data = await start_classification(lista_de_comentários, tema, context)
                                
    #         return jsonify(json_data=json_data), 200
        
        
    # return jsonify({"error": "Erro"}), 400      

#--------------------------------------------------------------------------------------
@app.route('/api/getclassificationsbyfilenew', methods=['POST'])
async def getclassificationsbyfilenew():           
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    tema = request.form.get('tema')
    context = request.form.get('context')    
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:        
            df_input = pd.read_excel(file)                                                            
            if 'Texto' in df_input.columns:                
                df_input = df_input.head(2000)                
                df_input['texto_limpo']  = df_input['Texto'].apply(clean_text)
                
                final_df = await classificacao_com_sentimento(df_input,tema, context)
                
                print("### RESULTADO FINAL DA CLASSIFICAÇÃO:")
                print(final_df)       
                                                            
                print(f"##### Processo finalizado...{get_current_datetime()}")
                                
                return jsonify(json_data=final_df.to_json(orient = 'records')), 200
            else:
                return jsonify({"error": "Column 'Texto' not found"}), 400
       
#---------------------------- CHART GENERATION -----------------------------------------------------
@app.route('/api/getchartdata', methods=['GET'])
def getchartdata():        
    json_data = generate_js_dictionary()
    
    return jsonify(json_data=json_data)            

@app.route('/api/getchartdatagrouped', methods=['GET'])
def getchartdatagrouped():            
    json_data = generate_js_dictionary_with_sentiment()
    return jsonify(json_data=json_data)            


#-----------------------------SET AND GET KEYS ---------------------------------------------------------
@app.route('/api/getkeys', methods=['GET'])
def getkeys():
    try:
        keys = []
        #keys.append(utils_conf.get_config_value('bearer_token'))
        #keys.append(utils_conf.get_config_value('OPENAI_KEY'))
        keys.append(utils_conf.get_config_value('max_tweets'))
        #keys.append(utils_conf.get_config_value('max_len_tags'))
        #keys.append(utils_conf.get_config_value('max_len_class'))
        #keys.append(utils_conf.get_config_value('tagQTD'))
        #keys.append(utils_conf.get_config_value('classificQTD'))
        
        print(f"#### Data sent {keys}")
    except:
        return jsonify("Erro ao carregar as APIs")            
        
    return jsonify(keys)          

@app.route('/api/setkeys', methods=['POST'])
def setkeys():
    if request.is_json:
        # Get the JSON data
        data = request.get_json()    
        print(f"##### Config received: {data}")
        #twitter_key = data.get('bearer_token', None)
        #openai_key = data.get('openai_key', None)
        max_tweets = data.get('max_tweets', None)
        #max_len_tags = data.get('max_len_tags', None)
        #max_len_class = data.get('max_len_class', None)
        #tagQTD = data.get('tagQTD', None)
        #classificQTD = data.get('classificQTD', None)
        
        #utils_conf.update_config_file('bearer_token', twitter_key)
        #utils_conf.update_config_file('OPENAI_KEY', openai_key)
        utils_conf.update_config_file('max_tweets', str(max_tweets))
        #utils_conf.update_config_file('max_len_tags', str(max_len_tags))
        #utils_conf.update_config_file('max_len_class', str(max_len_class))
        #utils_conf.update_config_file('tagQTD', str(tagQTD))
        #utils_conf.update_config_file('classificQTD', str(classificQTD))                
        
        print("##### Config Updated!")
        return jsonify('200')            
        
    
    return None

#---------------------------- XLSX FILES DOWNLOAD ----------------------------------------------------------
@app.route('/api/downloadsearch')
def downloadsearch(filename='tweets_search_output.xlsx'):
    return send_from_directory('outputs', filename, as_attachment=True)

@app.route('/api/downloadclassification')
def downloadclassification(filename='text_classification_output.xlsx'):
    return send_from_directory('outputs', filename, as_attachment=True)
  
#--------------------------------------------------------------------------------------    
@app.route('/api/getsummary', methods=['GET'])
async def getsummary():       
    print(f"##### Gerando Resumo...{get_current_datetime()}") 
    #Le o dataframe
    df = pd.read_excel("outputs/text_classification_output.xlsx", sheet_name=1)
    #Transforma em texto (ou json)
    texto_total = df.to_string()
    
    #Envia para a api resumir
    message = [
        {"role": "system", "content": f"""Leia o dataframe no formato de texto abaixo e me retorne um resumo de cada Categoria.
         Separe as categorias em Tópicos, informando as palavras chave e o sentimento predominante na categoria
         Dataframe:
         {texto_total}
"""}
   ]
    
    resultado = await openAiApiCall(message)                    
    print(resultado)
    #Retorna o resumo como texto ao frontckend      
    print(f"##### Enviando resumo...{get_current_datetime()}")   
    return resultado

#--------------------------------------------------------------------------------------    


if __name__ == '__main__':
    app.run(debug=True, port=8080)
