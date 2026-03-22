# Usa uma imagem oficial leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos locais para dentro da imagem
COPY . /app

# Atualiza pacotes básicos e instala as dependências
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Configura as permissões para o usuário 1000 (Obrigatório para o Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Define as variáveis de ambiente necessárias
ENV HOST=0.0.0.0
ENV PORT=7860

# Expõe a porta 7860 exigida pelo Hugging Face e outras nuvens
EXPOSE 7860

# Inicia a aplicação usando 2 processos e tolerância de 5 minutos (300 segundos) para leitura pesada de Excel
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "300"]
