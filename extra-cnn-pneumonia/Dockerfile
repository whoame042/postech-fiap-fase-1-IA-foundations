# Imagem com Python 3.13, a mesma versão usada no desenvolvimento
FROM python:3.13-slim

# PYTHONUNBUFFERED: os prints aparecem na hora, sem ficar "presos" em um buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copia só o requirements.txt primeiro e instala. Assim, se depois eu mudar um
# arquivo do projeto mas não as dependências, o Docker reaproveita esta camada
# em vez de reinstalar tudo de novo (fica bem mais rápido para reconstruir a imagem).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto. data/ e models/ ficam de fora de propósito: são
# grandes e não fazem parte da imagem — entram depois como volume (veja o
# README.md, seção "Executar com Docker").
COPY src/ src/
COPY notebooks/ notebooks/

EXPOSE 8888

# Sobe o Jupyter Lab acessível de fora do container, sem pedir token/senha.
# Isso é adequado só para rodar localmente na sua máquina — nunca exponha essa
# porta na internet assim. --allow-root é necessário porque o container roda
# como root por padrão (não tem relação com privilégios na sua máquina).
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password="]
