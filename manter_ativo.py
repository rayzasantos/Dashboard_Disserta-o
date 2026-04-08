import time
import urllib.request
from datetime import datetime

print("="*60)
print(" SCRIPT PARA MANTER O DASHBOARD ATIVO (HUGGING FACE)")
print("="*60)
print()
print("Dica: Deixe esta janela preta aberta no seu computador")
print("durante a semana da banca ou no dia da apresentação.")
print("Ela vai acessar o seu site de tempos em tempos para evitar")
print("que ele 'durma'.\n")

# Pede a URL
url = input("Cole o link completo do seu Dashboard no Hugging Face (ex: https://seu-usuario-seu-espaco.hf.space): ").strip()

if not url.startswith("http"):
    url = "https://" + url

print(f"\nIniciando o monitoramento de: {url}")
print("Pressione 'Ctrl + C' se quiser parar o script.\n")

while True:
    try:
        # Acessa a URL
        resposta = urllib.request.urlopen(url)
        agora = datetime.now().strftime("%H:%M:%S")
        
        if resposta.getcode() == 200:
            print(f"[{agora}] Sucesso! O dashboard está online e ativo.")
        else:
            print(f"[{agora}] Aviso: O site retornou o código {resposta.getcode()}...")
            
    except Exception as e:
        agora = datetime.now().strftime("%H:%M:%S")
        print(f"[{agora}] Erro ao tentar acessar o dashboard: {e}")
        print("Verifique se o link está correto e se o seu computador tem internet.")
        
    # Espera 5 minutos (300 segundos) antes de tentar de novo
    time.sleep(300)
