FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

WORKDIR /app

#Usuário sem privilégios de root para rodar o Jupyter
RUN useradd --create-home --uid 1000 app

# Dependências primeiro, para aproveitar o cache de camadas
# requirements.txt tbm instala o pacote local (-e .), por isso pyproject.toml e src/ entram aqui.
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install -r requirements.txt

COPY . .
RUN chown -R app:app /app
USER app

EXPOSE 8888

#Padrão: Abre o JupyterLab na raiz do projeto; o token de acesso aparece nos logs
#Para somente executar os notebooks, veja o README (seção "Com Docker").
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
