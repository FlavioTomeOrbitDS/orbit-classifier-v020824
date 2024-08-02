def update_config_file(key, value, filename="conf/config.txt"):
    # Tentar abrir o arquivo existente para ler as configurações atuais
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    # Criar um dicionário a partir das linhas existentes
    config_dict = {}
    for line in lines:
        if ':' in line:
            k, v = line.strip().split(':', 1)
            config_dict[k.strip()] = v.strip()

    # Atualizar o dicionário com a nova chave e valor
    config_dict[key] = value

    # Escrever o dicionário atualizado de volta ao arquivo
    with open(filename, 'w') as file:
        for k, v in config_dict.items():
            file.write(f"{k} : {v}\n")
            


def get_config_value(key, filename="conf/config.txt"):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("Arquivo de configuração não encontrado.")
        return None

    # Criar um dicionário a partir das linhas existentes
    config_dict = {}
    for line in lines:
        if ':' in line:
            k, v = line.strip().split(':', 1)
            config_dict[k.strip()] = v.strip()

    # Retornar o valor para a chave especificada, se existir
    if key in config_dict:
        return config_dict[key]
    else:
        print(f"Chave {key} não encontrada.")
        return None
