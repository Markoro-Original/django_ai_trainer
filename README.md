# django_ai_trainer

## **Rodando o projeto**

#### Crie e ative o ambiente virtual

Windows
```bash
python -m venv nome_da_venv
nome_da_venv\Scripts\activate     # Para Windows
```

Linux
```bash
python -m venv nome_da_venv
source nome_da_venv/bin/activate  # Para Linux/macOS
```

### **Com o ambiente virtual ativado**:
- Instale as dependências:
```bash
pip install -r requirements.txt
```
- Faça as migrações necessárias:
```bash
python manage.py migrate
```
- Rode o servidor localmente:
```bash
python manage.py runserver
```

## **Credenciais padrão**

Para acessar a aplicação, utilize as seguintes credenciais:

- **Usuário:** admin
- **Senha:** admin
